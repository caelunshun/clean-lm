mod common;

use cleandl::{
    context::Context,
    execgraph::{
        BuildOptions, ExecGraph, GraphBuilder, GraphOutput, Input, SymTensor,
        ops::attention::{
            AttentionFwdOpts, AttentionMask, AttentionOpts, Fp8Descale, KvLayout, QueryLayout,
        },
        sym::Monomial,
    },
    tensor::Tensor,
};
use common::{assert_all_close, download, seeded_rng, test_context};
use float8::F8E4M3;
use half::{bf16, f16};
use rand::{Rng, seq::SliceRandom};
use rand_distr::{Distribution, Normal, Uniform};
use rayon::prelude::*;
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType},
    shape::Shape,
};
use std::sync::OnceLock;

fn err<T>(r: anyhow::Result<T>) -> String {
    format!("{:#}", r.err().expect("an error"))
}

const BF16: ElementType = ElementType::Float(FloatType::Bf16);
const F16: ElementType = ElementType::Float(FloatType::F16);
const F8: ElementType = ElementType::Float(FloatType::F8E4M3);
const F32: ElementType = ElementType::Float(FloatType::F32);
const I32: ElementType = ElementType::SInt(SIntType::S32);

fn cx() -> &'static Context {
    static CX: OnceLock<Context> = OnceLock::new();
    CX.get_or_init(test_context)
}

/// Device 0 is one flash-attention-rs supports.
fn supported(cx: &Context) -> bool {
    let d = cx.cuda();
    let cc = d.compute_capability_major() * 10 + d.compute_capability_minor();
    cc == 89 || cc == 90 || cc >= 100
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum QKind {
    Dense,
    /// Packed with this many unused rows at the end.
    Packed {
        padding: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum KKind {
    Dense,
    Packed { padding: usize },
    Paged { page_size: usize },
}

#[derive(Clone, Debug)]
struct Case {
    ty: ElementType,
    out_type: Option<ElementType>,
    head_dim: usize,
    heads: (usize, usize),
    /// Allocated lengths per sequence; dense layouts need equal lengths.
    seqlens_q: Vec<usize>,
    seqlens_k: Vec<usize>,
    q_kind: QKind,
    k_kind: KKind,
    seqused_q: Option<Vec<usize>>,
    seqused_k: Option<Vec<usize>>,
    /// Pass `max_seqlen` bounds for packed layouts instead of the totals.
    max_seqlen: bool,
    mask: AttentionMask,
    softcap: f32,
    softmax_scale: Option<f32>,
    lse: bool,
    num_splits: Option<u32>,
    backward: bool,
    /// Batch and sequence dimensions are dynamic, with hints of these multiples of the actual
    /// sizes.
    dynamic: Option<f64>,
    /// Dense Q, K and V are slices of one `[batch, seqlen, heads_q + 2 * heads_kv, head_dim]`
    /// input.
    fused_qkv: bool,
}

impl Case {
    fn dense(batch: usize, sq: usize, sk: usize, heads: (usize, usize), head_dim: usize) -> Self {
        Self {
            ty: BF16,
            out_type: None,
            head_dim,
            heads,
            seqlens_q: vec![sq; batch],
            seqlens_k: vec![sk; batch],
            q_kind: QKind::Dense,
            k_kind: KKind::Dense,
            seqused_q: None,
            seqused_k: None,
            max_seqlen: false,
            mask: AttentionMask::None,
            softcap: 0.0,
            softmax_scale: None,
            lse: true,
            num_splits: None,
            backward: false,
            dynamic: None,
            fused_qkv: false,
        }
    }

    fn varlen(seqlens_q: Vec<usize>, seqlens_k: Vec<usize>, heads: (usize, usize)) -> Self {
        Self {
            seqlens_q,
            seqlens_k,
            q_kind: QKind::Packed { padding: 5 },
            k_kind: KKind::Packed { padding: 3 },
            ..Self::dense(0, 0, 0, heads, 128)
        }
    }

    fn with(mut self, f: impl FnOnce(&mut Self)) -> Self {
        f(&mut self);
        self
    }

    fn batch(&self) -> usize {
        self.seqlens_q.len()
    }

    fn used_q(&self, b: usize) -> usize {
        self.seqused_q.as_ref().map_or(self.seqlens_q[b], |u| u[b])
    }

    /// Keys of sequence `b`: for paged keys without `seqused_k`, `max_seqlen_k` (the longest
    /// sequence with `max_seqlen`, otherwise the page table's capacity).
    fn used_k(&self, b: usize) -> usize {
        match (&self.seqused_k, self.k_kind) {
            (Some(u), _) => u[b],
            (None, KKind::Paged { page_size }) if !self.max_seqlen => {
                let max = self.seqlens_k.iter().max().unwrap();
                max.div_ceil(page_size) * page_size
            }
            (None, KKind::Paged { .. }) => *self.seqlens_k.iter().max().unwrap(),
            (None, _) => self.seqlens_k[b],
        }
    }

    fn mask_range(&self, i: usize, sq: usize, sk: usize) -> (usize, usize) {
        let center = i as i64 + sk as i64 - sq as i64;
        let (left, right) = match self.mask {
            AttentionMask::None => (None, None),
            AttentionMask::Causal => (None, Some(0)),
            AttentionMask::SlidingWindow { left, right } => (left, right),
        };
        let hi = right.map_or(sk as i64, |r| (center + i64::from(r) + 1).min(sk as i64));
        let lo = left.map_or(0, |l| (center - i64::from(l)).max(0));
        if hi <= lo {
            (0, 0)
        } else {
            (lo as usize, hi as usize)
        }
    }

    fn scale(&self) -> f64 {
        f64::from(
            self.softmax_scale
                .unwrap_or(1.0 / (self.head_dim as f32).sqrt()),
        )
    }
}

fn cumsum(lens: &[usize]) -> Vec<usize> {
    let mut cu = vec![0];
    for &l in lens {
        cu.push(cu.last().unwrap() + l);
    }
    cu
}

/// Host-side data and how logical rows map to rows of the device tensors.
struct Data {
    /// Rounded values, in the device layouts.
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    d_out: Vec<f32>,
    /// Device row of query `i` / key `j` of sequence `b`.
    q_rows: Vec<Vec<usize>>,
    k_rows: Vec<Vec<usize>>,
    total_q: usize,
    /// Rows of the K and V tensors (pages * page_size when paged).
    total_k: usize,
    page_table: Option<(Vec<i32>, usize)>,
    descale: Option<[Vec<f32>; 3]>,
}

fn round(ty: ElementType, x: f32) -> f32 {
    match ty {
        BF16 => bf16::from_f32(x).to_f32(),
        F16 => f16::from_f32(x).to_f32(),
        F8 => F8E4M3::from_f32(x).to_f32(),
        F32 => x,
        _ => unreachable!(),
    }
}

fn encode(ty: ElementType, values: &[f32]) -> Vec<u8> {
    match ty {
        BF16 => bytemuck::cast_slice(
            &values
                .iter()
                .map(|&x| bf16::from_f32(x))
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        F16 => bytemuck::cast_slice(&values.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>())
            .to_vec(),
        F8 => values
            .iter()
            .map(|&x| F8E4M3::from_f32(x).to_bits())
            .collect(),
        F32 => bytemuck::cast_slice(values).to_vec(),
        _ => unreachable!(),
    }
}

fn decode(ty: ElementType, t: &Tensor, cx: &Context) -> Vec<f32> {
    match ty {
        BF16 => download::<bf16>(t, cx)
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect(),
        F16 => download::<f16>(t, cx)
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect(),
        F32 => download::<f32>(t, cx).unwrap(),
        _ => unreachable!(),
    }
}

fn random(ty: ElementType, n: usize, rng: &mut impl Rng) -> Vec<f32> {
    let distr = Normal::new(0.0f32, 1.0).unwrap();
    (0..n).map(|_| round(ty, distr.sample(rng))).collect()
}

impl Data {
    fn new(case: &Case, seed: u64) -> Self {
        let mut rng = seeded_rng(seed);
        let (hq, hk) = case.heads;
        let d = case.head_dim;
        let b = case.batch();
        let (q_rows, total_q) = match case.q_kind {
            QKind::Dense => {
                let s = case.seqlens_q[0];
                assert!(case.seqlens_q.iter().all(|&l| l == s));
                (
                    (0..b)
                        .map(|b| (0..s).map(|i| b * s + i).collect())
                        .collect(),
                    b * s,
                )
            }
            QKind::Packed { padding } => {
                let cu = cumsum(&case.seqlens_q);
                (
                    (0..b).map(|b| (cu[b]..cu[b + 1]).collect()).collect(),
                    cu[b] + padding,
                )
            }
        };
        let mut page_table = None;
        let (k_rows, total_k) = match case.k_kind {
            KKind::Dense => {
                let s = case.seqlens_k[0];
                assert!(case.seqlens_k.iter().all(|&l| l == s));
                (
                    (0..b)
                        .map(|b| (0..s).map(|j| b * s + j).collect())
                        .collect(),
                    b * s,
                )
            }
            KKind::Packed { padding } => {
                let cu = cumsum(&case.seqlens_k);
                (
                    (0..b).map(|b| (cu[b]..cu[b + 1]).collect()).collect(),
                    cu[b] + padding,
                )
            }
            KKind::Paged { page_size } => {
                // Every slot of the table gets its own page: without `seqused_k`, every sequence
                // has `max_seqlen_k` keys, which defaults to the table's capacity.
                let max_pages = case
                    .seqlens_k
                    .iter()
                    .map(|l| l.div_ceil(page_size))
                    .max()
                    .unwrap();
                // Shuffled pages, plus spare ones nothing points to.
                let num_pages = b * max_pages + 3;
                let mut order: Vec<usize> = (0..num_pages).collect();
                order.shuffle(&mut rng);
                let table: Vec<i32> = order[..b * max_pages].iter().map(|&p| p as i32).collect();
                let rows = (0..b)
                    .map(|bi| {
                        (0..max_pages * page_size)
                            .map(|j| {
                                table[bi * max_pages + j / page_size] as usize * page_size
                                    + j % page_size
                            })
                            .collect()
                    })
                    .collect();
                page_table = Some((table, max_pages));
                (rows, num_pages * page_size)
            }
        };
        let descale = (case.ty == F8).then(|| {
            let distr = Uniform::new(0.5f32, 2.0).unwrap();
            [(); 3].map(|_| (0..b * hk).map(|_| distr.sample(&mut rng)).collect())
        });
        let out_ty = case.out_type.unwrap_or(case.ty);
        Self {
            q: random(case.ty, total_q * hq * d, &mut rng),
            k: random(case.ty, total_k * hk * d, &mut rng),
            v: random(case.ty, total_k * hk * d, &mut rng),
            d_out: random(out_ty, total_q * hq * d, &mut rng),
            q_rows,
            k_rows,
            total_q,
            total_k,
            page_table,
            descale,
        }
    }
}

struct Expected {
    out: Vec<f32>,
    lse: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

/// f64 reference over the rounded inputs.
fn reference(case: &Case, data: &Data) -> Expected {
    let (hq, hk) = case.heads;
    let group = hq / hk;
    let d = case.head_dim;
    let scale = case.scale();
    let softcap = f64::from(case.softcap);
    let packed_q = matches!(case.q_kind, QKind::Packed { .. });
    let sq_alloc = case.seqlens_q[0];
    let lse_index = |b: usize, h: usize, i: usize, row: usize| {
        if packed_q {
            h * data.total_q + row
        } else {
            (b * hq + h) * sq_alloc + i
        }
    };
    struct Head {
        out: Vec<(usize, Vec<f64>)>,
        lse: Vec<(usize, f64)>,
        dq: Vec<(usize, Vec<f64>)>,
        dk: Vec<(usize, Vec<f64>)>,
        dv: Vec<(usize, Vec<f64>)>,
    }
    let heads: Vec<(usize, usize)> = (0..case.batch())
        .flat_map(|b| (0..hq).map(move |h| (b, h)))
        .collect();
    let results: Vec<Head> = heads
        .par_iter()
        .map(|&(b, h)| {
            let kh = h / group;
            let (dq_s, dk_s, dv_s) = match &data.descale {
                Some([q, k, v]) => (q[b * hk + kh], k[b * hk + kh], v[b * hk + kh]),
                None => (1.0, 1.0, 1.0),
            };
            let (sq, sk) = (case.used_q(b), case.used_k(b));
            let q = |i: usize| {
                let base = (data.q_rows[b][i] * hq + h) * d;
                data.q[base..base + d].iter().map(|&x| f64::from(x * dq_s))
            };
            let kv = |src: &[f32], s: f32, j: usize| -> Vec<f64> {
                let base = (data.k_rows[b][j] * hk + kh) * d;
                src[base..base + d]
                    .iter()
                    .map(|&x| f64::from(x * s))
                    .collect()
            };
            let ks: Vec<Vec<f64>> = (0..sk).map(|j| kv(&data.k, dk_s, j)).collect();
            let vs: Vec<Vec<f64>> = (0..sk).map(|j| kv(&data.v, dv_s, j)).collect();
            let mut head = Head {
                out: vec![],
                lse: vec![],
                dq: vec![],
                dk: vec![],
                dv: vec![],
            };
            let mut dk = vec![vec![0.0; d]; sk];
            let mut dv = vec![vec![0.0; d]; sk];
            for i in 0..sq {
                let qi: Vec<f64> = q(i).collect();
                let row = data.q_rows[b][i];
                let (lo, hi) = case.mask_range(i, sq, sk);
                // Scores and the derivative of the score by the raw dot product.
                let (s, ds): (Vec<f64>, Vec<f64>) = (lo..hi)
                    .map(|j| {
                        let x = scale * qi.iter().zip(&ks[j]).map(|(a, b)| a * b).sum::<f64>();
                        if softcap > 0.0 {
                            let t = (x / softcap).tanh();
                            (softcap * t, scale * (1.0 - t * t))
                        } else {
                            (x, scale)
                        }
                    })
                    .unzip();
                let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = s.iter().map(|x| (x - max).exp()).sum();
                let lse = if lo == hi {
                    f64::NEG_INFINITY
                } else {
                    max + sum.ln()
                };
                let p: Vec<f64> = s.iter().map(|x| (x - lse).exp()).collect();
                let mut o = vec![0.0; d];
                for (pj, j) in p.iter().zip(lo..hi) {
                    for (o, v) in o.iter_mut().zip(&vs[j]) {
                        *o += pj * v;
                    }
                }
                head.lse.push((lse_index(b, h, i, row), lse));
                if case.backward {
                    let base = (row * hq + h) * d;
                    let d_o: Vec<f64> = data.d_out[base..base + d]
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect();
                    let dpsum: f64 = d_o.iter().zip(&o).map(|(a, b)| a * b).sum();
                    let mut dq = vec![0.0; d];
                    for (idx, j) in (lo..hi).enumerate() {
                        let dp: f64 = d_o.iter().zip(&vs[j]).map(|(a, b)| a * b).sum();
                        let dqk = p[idx] * (dp - dpsum) * ds[idx];
                        for x in 0..d {
                            dv[j][x] += p[idx] * d_o[x];
                            dq[x] += dqk * ks[j][x];
                            dk[j][x] += dqk * qi[x];
                        }
                    }
                    head.dq.push((row, dq));
                }
                head.out.push((row, o));
            }
            if case.backward {
                for j in 0..sk {
                    head.dk
                        .push((data.k_rows[b][j], std::mem::take(&mut dk[j])));
                    head.dv
                        .push((data.k_rows[b][j], std::mem::take(&mut dv[j])));
                }
            }
            head
        })
        .collect();

    let mut e = Expected {
        out: vec![0.0; data.total_q * hq * d],
        lse: vec![f32::NEG_INFINITY; data.total_q * hq],
        dq: vec![0.0; data.total_q * hq * d],
        dk: vec![0.0; data.total_k * hk * d],
        dv: vec![0.0; data.total_k * hk * d],
    };
    let mut dk = vec![0.0f64; e.dk.len()];
    let mut dv = vec![0.0f64; e.dv.len()];
    for (head, &(_, h)) in results.iter().zip(&heads) {
        let kh = h / group;
        for (row, o) in &head.out {
            let base = (row * hq + h) * d;
            for (x, &o) in o.iter().enumerate() {
                e.out[base + x] = o as f32;
            }
        }
        for &(i, lse) in &head.lse {
            e.lse[i] = lse as f32;
        }
        for (row, g) in &head.dq {
            let base = (row * hq + h) * d;
            for (x, &g) in g.iter().enumerate() {
                e.dq[base + x] = g as f32;
            }
        }
        for ((row, gk), (_, gv)) in head.dk.iter().zip(&head.dv) {
            let base = (row * hk + kh) * d;
            for x in 0..d {
                dk[base + x] += gk[x];
                dv[base + x] += gv[x];
            }
        }
    }
    e.dk = dk.iter().map(|&x| x as f32).collect();
    e.dv = dv.iter().map(|&x| x as f32).collect();
    e
}

/// A flat device tensor and the shape it is bound with (`Tensor::from_bytes` takes only shapes
/// it considers contiguous).
struct Upload(Tensor, Vec<usize>);

fn upload_bytes(bytes: &[u8], ty: ElementType, shape: Vec<usize>, cx: &Context) -> Upload {
    let n = shape.iter().product::<usize>();
    let t = Tensor::from_bytes(bytes, ty, [n], cx.default_compute_stream(), cx).unwrap();
    Upload(t, shape)
}

fn upload(ty: ElementType, values: &[f32], shape: Vec<usize>, cx: &Context) -> Upload {
    upload_bytes(&encode(ty, values), ty, shape, cx)
}

fn upload_i32(values: &[i32], shape: Vec<usize>, cx: &Context) -> Upload {
    upload_bytes(bytemuck::cast_slice(values), I32, shape, cx)
}

fn upload_lens(values: &[usize], cx: &Context) -> Upload {
    let v: Vec<i32> = values.iter().map(|&x| x as i32).collect();
    upload_i32(&v, vec![v.len()], cx)
}

/// Fails unless `actual` is within `atol * max|expected| + rtol * |expected|` of `expected`.
#[track_caller]
fn check(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    // -inf LSEs compare equal.
    let fin = |x: &[f32]| -> Vec<f32> {
        x.iter()
            .map(|&x| if x == f32::NEG_INFINITY { -1e30 } else { x })
            .collect()
    };
    let scale = expected
        .iter()
        .filter(|x| x.is_finite())
        .fold(0f32, |m, x| m.max(x.abs()))
        .max(1e-3);
    assert_all_close(name, &fin(actual), &fin(expected), move |_, e| {
        atol * scale + rtol * e.abs()
    });
}

/// Builds and runs the case's graph and checks it against the reference.
fn run(case: &Case) {
    let cx = cx();
    if !supported(cx) {
        eprintln!("SKIPPED: flash attention does not support this device");
        return;
    }
    let name = format!("{case:?}");
    let data = Data::new(case, 7);
    let expected = reference(case, &data);
    let (hq, hk) = case.heads;
    let d = case.head_dim;
    let b = case.batch();
    let out_ty = case.out_type.unwrap_or(case.ty);

    let gb = GraphBuilder::new();
    let dim = |name: &str, actual: usize| -> Monomial {
        match case.dynamic {
            Some(f) => gb
                .dynamic_dim(name, ((actual as f64 * f) as usize).max(1), 0)
                .into(),
            None => actual.into(),
        }
    };
    let mut inputs: Vec<Upload> = Vec::new();
    let mut add = |name: &str, ty, shape: Vec<Monomial>, t: Upload| {
        inputs.push(t);
        gb.input(name, ty, shape).unwrap()
    };
    let batch = dim("batch", b);
    let fixed = |n: usize| Monomial::fixed(n);
    let (q, k, v) = if case.fused_qkv {
        assert!(case.q_kind == QKind::Dense && case.k_kind == KKind::Dense);
        let s = case.seqlens_q[0];
        assert_eq!(s, case.seqlens_k[0]);
        let width = hq + 2 * hk;
        let mut fused = vec![0f32; b * s * width * d];
        for r in 0..b * s {
            for (src, heads, off) in [(&data.q, hq, 0), (&data.k, hk, hq), (&data.v, hk, hq + hk)] {
                let dst = (r * width + off) * d;
                fused[dst..dst + heads * d]
                    .copy_from_slice(&src[r * heads * d..(r + 1) * heads * d]);
            }
        }
        let seq = dim("seq", s);
        let qkv = add(
            "qkv",
            case.ty,
            vec![batch.clone(), seq, fixed(width), fixed(d)],
            upload(case.ty, &fused, vec![b, s, width, d], cx),
        );
        (
            qkv.sliced(2, 0, hq).unwrap(),
            qkv.sliced(2, hq, hk).unwrap(),
            qkv.sliced(2, hq + hk, hk).unwrap(),
        )
    } else {
        let q_shape: Vec<Monomial> = match case.q_kind {
            QKind::Dense => vec![
                batch.clone(),
                dim("seq_q", case.seqlens_q[0]),
                fixed(hq),
                fixed(d),
            ],
            QKind::Packed { .. } => vec![dim("total_q", data.total_q), fixed(hq), fixed(d)],
        };
        let k_shape: Vec<Monomial> = match case.k_kind {
            KKind::Dense => vec![
                batch.clone(),
                dim("seq_k", case.seqlens_k[0]),
                fixed(hk),
                fixed(d),
            ],
            KKind::Packed { .. } => vec![dim("total_k", data.total_k), fixed(hk), fixed(d)],
            KKind::Paged { page_size } => vec![
                dim("pages", data.total_k / page_size),
                fixed(page_size),
                fixed(hk),
                fixed(d),
            ],
        };
        let q_dims: Vec<usize> = match case.q_kind {
            QKind::Dense => vec![b, case.seqlens_q[0], hq, d],
            QKind::Packed { .. } => vec![data.total_q, hq, d],
        };
        let k_dims: Vec<usize> = match case.k_kind {
            KKind::Dense => vec![b, case.seqlens_k[0], hk, d],
            KKind::Packed { .. } => vec![data.total_k, hk, d],
            KKind::Paged { page_size } => vec![data.total_k / page_size, page_size, hk, d],
        };
        (
            add("q", case.ty, q_shape, upload(case.ty, &data.q, q_dims, cx)),
            add(
                "k",
                case.ty,
                k_shape.clone(),
                upload(case.ty, &data.k, k_dims.clone(), cx),
            ),
            add("v", case.ty, k_shape, upload(case.ty, &data.v, k_dims, cx)),
        )
    };

    let batch_plus_one = dim("batch_plus_one", b + 1);
    let mut opts = AttentionOpts {
        softmax_scale: case.softmax_scale,
        mask: case.mask,
        softcap: case.softcap,
        ..Default::default()
    };
    let max_q = case.seqlens_q.iter().copied().max().unwrap();
    let max_k = case.seqlens_k.iter().copied().max().unwrap();
    if let QKind::Packed { .. } = case.q_kind {
        let cu = add(
            "cu_seqlens_q",
            I32,
            vec![batch_plus_one.clone()],
            upload_lens(&cumsum(&case.seqlens_q), cx),
        );
        opts.q_layout = QueryLayout::Packed {
            cu_seqlens: cu,
            max_seqlen: case.max_seqlen.then(|| Monomial::fixed(max_q)),
        };
    }
    match case.k_kind {
        KKind::Dense => {}
        KKind::Packed { .. } => {
            let cu = add(
                "cu_seqlens_k",
                I32,
                vec![batch_plus_one.clone()],
                upload_lens(&cumsum(&case.seqlens_k), cx),
            );
            opts.kv_layout = KvLayout::Packed {
                cu_seqlens: cu,
                max_seqlen: case.max_seqlen.then(|| Monomial::fixed(max_k)),
            };
        }
        KKind::Paged { .. } => {
            let (table, max_pages) = data.page_table.as_ref().unwrap();
            let t = upload_i32(table, vec![b, *max_pages], cx);
            let pt = add(
                "page_table",
                I32,
                vec![batch.clone(), dim("max_pages", *max_pages)],
                t,
            );
            opts.kv_layout = KvLayout::Paged {
                page_table: pt,
                max_seqlen: case.max_seqlen.then(|| Monomial::fixed(max_k)),
            };
        }
    }
    if let Some(used) = &case.seqused_q {
        opts.seqused_q = Some(add(
            "seqused_q",
            I32,
            vec![batch.clone()],
            upload_lens(used, cx),
        ));
    }
    if let Some(used) = &case.seqused_k {
        opts.seqused_k = Some(add(
            "seqused_k",
            I32,
            vec![batch.clone()],
            upload_lens(used, cx),
        ));
    }
    let fp8_descale = data.descale.as_ref().map(|[dq, dk, dv]| {
        let mut desc = |name: &str, x: &[f32]| {
            add(
                name,
                F32,
                vec![batch.clone(), fixed(hk)],
                upload(F32, x, vec![b, hk], cx),
            )
        };
        Fp8Descale {
            q: desc("q_descale", dq),
            k: desc("k_descale", dk),
            v: desc("v_descale", dv),
        }
    });
    let fwd = q
        .attention_fwd(
            k,
            v,
            &opts,
            AttentionFwdOpts {
                fp8_descale,
                out_type: case.out_type,
                lse: case.lse || case.backward,
                num_splits: case.num_splits,
            },
        )
        .unwrap();
    gb.output(fwd.out).unwrap();
    if case.lse || case.backward {
        gb.output(fwd.lse.unwrap()).unwrap();
    }
    if case.backward {
        let d_out_t = upload(out_ty, &data.d_out, fwd_shape(case, &data), cx);
        let d_out: SymTensor<'_> = add("d_out", out_ty, fwd.out.shape().dims().to_vec(), d_out_t);
        let grads = q
            .attention_bwd(k, v, fwd.out, fwd.lse.unwrap(), d_out, &opts)
            .unwrap();
        for t in [grads.dq, grads.dk, grads.dv] {
            gb.output(t).unwrap();
        }
    }
    let graph = gb.finish().unwrap();
    let exec = ExecGraph::build(graph, cx, BuildOptions::default())
        .unwrap_or_else(|e| panic!("{name}: build failed: {e:#}"));
    let mut inputs: Vec<Input<'_>> = inputs
        .iter()
        .map(|Upload(t, shape)| Input::from(t.reshaped(Shape::new(shape.iter().copied()))))
        .collect();
    let outputs = exec
        .execute_default(&mut inputs, cx.default_compute_stream())
        .unwrap_or_else(|e| panic!("{name}: execute failed: {e:#}"));
    cx.default_compute_stream().block_on().unwrap();
    let mut outputs = outputs.into_iter().map(GraphOutput::into_plain);

    let (atol, rtol) = match (case.ty, out_ty) {
        (F8, _) => (4e-2, 8e-2),
        (BF16, _) => (1e-2, 1e-2),
        _ => (2e-3, 2e-3),
    };
    let out = outputs.next().unwrap();
    let got_out = decode(out_ty, &out, cx);
    check(&format!("{name} out"), &got_out, &expected.out, atol, rtol);
    if case.lse || case.backward {
        let lse = outputs.next().unwrap();
        check(
            &format!("{name} lse"),
            &decode(F32, &lse, cx),
            &expected.lse,
            1e-4,
            1e-3,
        );
    }
    if case.backward {
        let (gatol, grtol) = if case.ty == BF16 {
            (2e-2, 2e-2)
        } else {
            (4e-3, 4e-3)
        };
        for (what, exp) in [
            ("dq", &expected.dq),
            ("dk", &expected.dk),
            ("dv", &expected.dv),
        ] {
            let t = outputs.next().unwrap();
            check(
                &format!("{name} {what}"),
                &decode(case.ty, &t, cx),
                exp,
                gatol,
                grtol,
            );
        }
    }
}

fn fwd_shape(case: &Case, data: &Data) -> Vec<usize> {
    let (hq, _) = case.heads;
    match case.q_kind {
        QKind::Dense => vec![case.batch(), case.seqlens_q[0], hq, case.head_dim],
        QKind::Packed { .. } => vec![data.total_q, hq, case.head_dim],
    }
}

const WINDOW: AttentionMask = AttentionMask::SlidingWindow {
    left: Some(70),
    right: Some(20),
};

#[test]
fn forward_masks() {
    for mask in [AttentionMask::None, AttentionMask::Causal, WINDOW] {
        run(&Case::dense(2, 300, 300, (8, 2), 128).with(|c| c.mask = mask));
        // More keys than queries, then more queries than keys (fully masked causal rows).
        run(&Case::dense(1, 200, 333, (4, 4), 128).with(|c| c.mask = mask));
        run(&Case::dense(2, 333, 129, (4, 1), 64).with(|c| c.mask = mask));
    }
    run(&Case::dense(1, 250, 250, (4, 2), 128).with(|c| {
        c.mask = AttentionMask::SlidingWindow {
            left: Some(50),
            right: Some(0),
        }
    }));
}

#[test]
fn forward_dtypes_and_head_dims() {
    for head_dim in [64, 96, 128, 256] {
        run(&Case::dense(2, 150, 190, (4, 2), head_dim).with(|c| c.mask = AttentionMask::Causal));
        run(&Case::dense(1, 170, 170, (2, 2), head_dim).with(|c| {
            c.ty = F16;
            c.lse = false;
        }));
    }
    run(&Case::dense(2, 200, 200, (4, 4), 64).with(|c| {
        c.ty = F16;
        c.softcap = 5.0;
        c.softmax_scale = Some(0.3);
        c.mask = WINDOW;
    }));
}

#[test]
fn forward_fp8() {
    for head_dim in [64, 128, 256] {
        for out_type in [BF16, F16] {
            run(&Case::dense(2, 200, 230, (4, 2), head_dim).with(|c| {
                c.ty = F8;
                c.out_type = Some(out_type);
                c.mask = AttentionMask::Causal;
            }));
        }
    }
    run(
        &Case::varlen(vec![100, 0, 77], vec![90, 40, 200], (4, 1)).with(|c| {
            c.ty = F8;
            c.out_type = Some(BF16);
            c.mask = WINDOW;
        }),
    );
}

#[test]
fn forward_paged() {
    for page_size in [16, 64, 256] {
        run(&Case::dense(3, 120, 0, (4, 2), 128).with(|c| {
            c.seqlens_k = vec![300, 129, 64];
            c.k_kind = KKind::Paged { page_size };
            c.mask = AttentionMask::Causal;
        }));
    }
    // Pages longer than the sequences, with `seqused_k`.
    run(&Case::dense(2, 1, 0, (8, 1), 128).with(|c| {
        c.seqlens_k = vec![512, 512];
        c.seqused_k = Some(vec![500, 3]);
        c.k_kind = KKind::Paged { page_size: 32 };
        c.max_seqlen = true;
    }));
    run(&Case::dense(2, 64, 0, (4, 2), 64).with(|c| {
        c.ty = F8;
        c.out_type = Some(F16);
        c.seqlens_k = vec![200, 90];
        c.k_kind = KKind::Paged { page_size: 16 };
        c.mask = WINDOW;
    }));
    run(
        &Case::varlen(vec![30, 50], vec![100, 300], (4, 2)).with(|c| {
            c.k_kind = KKind::Paged { page_size: 64 };
            c.mask = AttentionMask::Causal;
        }),
    );
}

#[test]
fn forward_varlen() {
    run(&Case::varlen(
        vec![100, 0, 257, 33],
        vec![120, 64, 257, 1],
        (4, 2),
    ));
    run(
        &Case::varlen(vec![100, 0, 257, 33], vec![120, 64, 257, 1], (4, 2)).with(|c| {
            c.mask = AttentionMask::Causal;
            c.max_seqlen = true;
        }),
    );
    run(
        &Case::varlen(vec![300, 40], vec![200, 380], (2, 2)).with(|c| {
            c.mask = WINDOW;
            c.seqused_q = Some(vec![250, 3]);
            c.seqused_k = Some(vec![150, 380]);
            c.ty = F16;
        }),
    );
    // Dense queries with seqused, and packed queries over dense keys.
    run(&Case::dense(3, 200, 200, (4, 4), 128).with(|c| {
        c.seqused_q = Some(vec![200, 1, 90]);
        c.seqused_k = Some(vec![17, 200, 0]);
        c.mask = AttentionMask::Causal;
    }));
    run(&Case::varlen(vec![64, 64], vec![128, 128], (4, 2)).with(|c| c.k_kind = KKind::Dense));
}

#[test]
fn forward_splits() {
    for num_splits in [1, 3, 8] {
        run(&Case::dense(1, 16, 2000, (4, 2), 128).with(|c| c.num_splits = Some(num_splits)));
    }
    run(
        &Case::varlen(vec![5, 1], vec![900, 1500], (8, 1)).with(|c| {
            c.num_splits = Some(4);
            c.mask = AttentionMask::Causal;
        }),
    );
}

#[test]
fn forward_dynamic_and_strided() {
    for factor in [0.5, 3.0] {
        run(&Case::dense(2, 150, 250, (4, 2), 128).with(|c| {
            c.dynamic = Some(factor);
            c.mask = AttentionMask::Causal;
        }));
        run(&Case::varlen(vec![100, 30], vec![90, 120], (4, 2)).with(|c| c.dynamic = Some(factor)));
        run(&Case::dense(2, 64, 0, (4, 2), 128).with(|c| {
            c.seqlens_k = vec![100, 300];
            c.k_kind = KKind::Paged { page_size: 64 };
            c.dynamic = Some(factor);
        }));
    }
    run(&Case::dense(2, 200, 200, (8, 2), 128).with(|c| {
        c.fused_qkv = true;
        c.mask = AttentionMask::Causal;
        c.backward = true;
    }));
}

#[test]
fn backward() {
    for mask in [AttentionMask::None, AttentionMask::Causal, WINDOW] {
        run(&Case::dense(2, 200, 200, (8, 2), 128).with(|c| {
            c.mask = mask;
            c.backward = true;
        }));
        run(&Case::dense(1, 150, 260, (4, 4), 64).with(|c| {
            c.mask = mask;
            c.ty = F16;
            c.backward = true;
        }));
    }
    for head_dim in [96, 256] {
        run(&Case::dense(1, 130, 130, (2, 1), head_dim).with(|c| {
            c.mask = AttentionMask::Causal;
            c.backward = true;
        }));
    }
    run(&Case::dense(2, 128, 128, (4, 2), 64).with(|c| {
        c.softcap = 4.0;
        c.backward = true;
        c.dynamic = Some(2.0);
    }));
}

#[test]
fn backward_varlen() {
    run(
        &Case::varlen(vec![100, 0, 257], vec![120, 64, 200], (4, 2)).with(|c| {
            c.mask = AttentionMask::Causal;
            c.backward = true;
        }),
    );
    run(
        &Case::varlen(vec![150, 90], vec![150, 200], (4, 4)).with(|c| {
            c.mask = WINDOW;
            c.seqused_q = Some(vec![120, 90]);
            c.seqused_k = Some(vec![150, 33]);
            c.ty = F16;
            c.backward = true;
            c.max_seqlen = true;
        }),
    );
    run(&Case::dense(2, 160, 160, (4, 2), 128).with(|c| {
        c.seqused_q = Some(vec![100, 160]);
        c.seqused_k = Some(vec![160, 7]);
        c.backward = true;
    }));
}

#[test]
fn rejected() {
    let gb = GraphBuilder::new();
    let t = |name: &str, ty, shape: [usize; 4]| gb.input(name, ty, shape.to_vec()).unwrap();
    let (q, k, v) = (
        t("q", BF16, [1, 64, 4, 128]),
        t("k", BF16, [1, 64, 2, 128]),
        t("v", BF16, [1, 64, 2, 128]),
    );
    let fwd = AttentionFwdOpts::default;

    let odd = t("odd", BF16, [1, 64, 4, 80]);
    assert!(
        err(odd.attention_fwd(odd, odd, &AttentionOpts::default(), fwd()))
            .contains("head dimension")
    );
    let k3 = t("k3", BF16, [1, 64, 3, 128]);
    assert!(err(q.attention_fwd(k3, k3, &AttentionOpts::default(), fwd())).contains("multiple"));
    let k8 = t("k8", F8, [1, 64, 2, 128]);
    assert!(err(q.attention_fwd(k8, k8, &AttentionOpts::default(), fwd())).contains("type"));
    let q8 = t("q8", F8, [1, 64, 4, 128]);
    assert!(
        err(q8.attention_fwd(k8, k8, &AttentionOpts::default(), fwd())).contains("output type")
    );
    let no_descale = AttentionFwdOpts {
        out_type: Some(BF16),
        ..fwd()
    };
    assert!(
        err(q8.attention_fwd(k8, k8, &AttentionOpts::default(), no_descale)).contains("descale")
    );
    let transposed = q.transposed(2, 3).unwrap();
    assert!(
        err(transposed.attention_fwd(k, v, &AttentionOpts::default(), fwd()))
            .contains("contiguous")
    );

    let out = q
        .attention_fwd(
            k,
            v,
            &AttentionOpts::default(),
            AttentionFwdOpts { lse: true, ..fwd() },
        )
        .unwrap();
    let pt = gb.input("pt", I32, vec![1, 4]).unwrap();
    let paged = AttentionOpts {
        kv_layout: KvLayout::Paged {
            page_table: pt,
            max_seqlen: None,
        },
        ..Default::default()
    };
    let lse = out.lse.unwrap();
    assert!(err(q.attention_bwd(k, v, out.out, lse, out.out, &paged)).contains("paged"));
    let wrong_lse = gb.input("lse2", F32, vec![1, 4, 32]).unwrap();
    assert!(
        err(q.attention_bwd(k, v, out.out, wrong_lse, out.out, &AttentionOpts::default()))
            .contains("LSE")
    );
    let unbound = gb.dynamic_dim("unbound", 10, 0);
    let cu = gb.input("cu", I32, vec![2]).unwrap();
    let qp = gb.input("qp", BF16, vec![64, 4, 128]).unwrap();
    let bad_max = AttentionOpts {
        q_layout: QueryLayout::Packed {
            cu_seqlens: cu,
            max_seqlen: Some(unbound.into()),
        },
        ..Default::default()
    };
    assert!(err(qp.attention_fwd(k, v, &bad_max, fwd())).contains("cannot be inferred"));
}
