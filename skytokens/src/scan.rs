use crate::vocab::TokenId;
use std::iter;

/// Builds a list of all positions `i` in the given token buffer
/// where `tokens[i] == first && tokens[i+1] == second`. The provided
/// positions list must have sufficient capacity to hold the maximum
/// number of possible indexes. Return match count.
pub(crate) fn scan_merge_matches(
    tokens: &[TokenId],
    matches: &mut [u32],
    first: TokenId,
    second: TokenId,
) -> usize {
    let max_matches = tokens.len().saturating_sub(1);
    assert!(matches.len() >= max_matches);
    assert!(tokens.len() < u32::MAX as usize);

    let tokens: &[u32] = bytemuck::cast_slice(tokens);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return unsafe { scan_merge_matches_avx512(tokens, matches, first.idx(), second.idx()) };
    }

    scan_merge_matches_fallback(tokens, matches, first.idx(), second.idx())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scan_merge_matches_avx512(
    tokens: &[u32],
    matches: &mut [u32],
    first: u32,
    second: u32,
) -> usize {
    use core::arch::x86_64::*;

    let mut match_count = 0;

    let (head, middle, tail) = unsafe { tokens.align_to::<__m512i>() };
    if middle.is_empty() {
        return scan_merge_matches_fallback(tokens, matches, first, second);
    }

    if !head.is_empty() {
        for (i, [a, b]) in head
            .array_windows::<2>()
            .copied()
            .chain(iter::once([
                head.last().copied().unwrap(),
                tokens[head.len()],
            ]))
            .enumerate()
        {
            if a == first && b == second {
                matches[match_count] = i as u32;
                match_count += 1;
            }
        }
    }

    // SIMD loop
    unsafe {
        let v_first = _mm512_set1_epi32(first as i32);
        let v_second = _mm512_set1_epi32(second as i32);

        for (i, &v_toks) in middle.iter().enumerate() {
            let tail_match = if i < middle.len() - 1 || !tail.is_empty() {
                tokens[head.len() + (i + 1) * 16] == second
            } else {
                false
            };

            let m1 = _mm512_cmpeq_epi32_mask(v_toks, v_first);
            let m2 = _mm512_cmpeq_epi32_mask(v_toks, v_second) >> 1 | ((tail_match as u16) << 15);

            let m_match = m1 & m2;

            let base = (head.len() + i * 16) as u32 as i32;
            let indexes = _mm512_setr_epi32(
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
                base + 8,
                base + 9,
                base + 10,
                base + 11,
                base + 12,
                base + 13,
                base + 14,
                base + 15,
            );

            let v_match_count = m_match.count_ones();
            let store_mask = ((1 << v_match_count) - 1) as u16;

            let compressed_indexes = _mm512_maskz_compress_epi32(m_match, indexes);

            _mm512_mask_storeu_epi32(
                matches.as_mut_ptr().add(match_count).cast(),
                store_mask,
                compressed_indexes,
            );
            match_count += v_match_count as usize;
        }
    }

    if !tail.is_empty() {
        for (i, [a, b]) in tail.array_windows::<2>().copied().enumerate() {
            let i = i + head.len() + 16 * middle.len();

            if a == first && b == second {
                matches[match_count] = i as u32;
                match_count += 1;
            }
        }
    }

    match_count
}

fn scan_merge_matches_fallback(
    tokens: &[u32],
    matches: &mut [u32],
    first: u32,
    second: u32,
) -> usize {
    let mut match_count = 0;
    for (i, &[a, b]) in tokens.array_windows::<2>().enumerate() {
        if a == first && b == second {
            matches[match_count] = i as u32;
            match_count += 1;
        }
    }
    match_count
}

#[cfg(test)]
mod tests {
    // --- LLM-generated --- //
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    const SENTINEL: u32 = 0xefef_efef;

    fn reference(tokens: &[u32], first: u32, second: u32) -> Vec<u32> {
        (0..tokens.len().saturating_sub(1))
            .filter(|&i| tokens[i] == first && tokens[i + 1] == second)
            .map(|i| i as u32)
            .collect()
    }

    fn run(
        tokens: &[u32],
        first: u32,
        second: u32,
        scan: impl FnOnce(&[u32], &mut [u32], u32, u32) -> usize,
    ) -> Vec<u32> {
        // Pad the match buffer to check that nothing is written past the match count.
        let mut matches = vec![SENTINEL; tokens.len() + 32];
        let count = scan(tokens, &mut matches, first, second);
        assert!(
            matches[count..].iter().all(|&m| m == SENTINEL),
            "wrote past match count"
        );
        matches.truncate(count);
        matches
    }

    /// Checks every implementation against the reference.
    fn check(tokens: &[u32], first: u32, second: u32) {
        let expected = reference(tokens, first, second);

        let fallback = run(tokens, first, second, scan_merge_matches_fallback);
        assert_eq!(fallback, expected, "fallback mismatch on {tokens:?}");

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            let avx512 = run(tokens, first, second, |t, m, f, s| unsafe {
                scan_merge_matches_avx512(t, m, f, s)
            });
            assert_eq!(avx512, expected, "avx512 mismatch on {tokens:?}");
        }

        let token_ids: &[TokenId] = bytemuck::cast_slice(tokens);
        let dispatched = run(tokens, first, second, |_, m, f, s| {
            scan_merge_matches(
                token_ids,
                m,
                TokenId::from_idx(f).unwrap(),
                TokenId::from_idx(s).unwrap(),
            )
        });
        assert_eq!(dispatched, expected, "dispatch mismatch on {tokens:?}");
    }

    /// Returns a slice of `len` elements starting `offset` u32s past a 64-byte boundary,
    /// so that tests control the head/middle/tail split of the AVX512 path.
    fn aligned(buf: &mut Vec<u32>, offset: usize, len: usize) -> &mut [u32] {
        buf.clear();
        buf.resize(len + offset + 16, 0);
        let misalign = (buf.as_ptr() as usize % 64) / 4;
        let start = (16 - misalign) % 16 + offset;
        &mut buf[start..start + len]
    }

    #[test]
    fn repeating_pattern() {
        let tokens: Vec<u32> = (0..1000).flat_map(|_| [0, 1, 2, 3]).collect();
        let token_ids: &[TokenId] = bytemuck::cast_slice(&tokens);

        let mut matches = vec![0u32; tokens.len() - 1];
        let count = scan_merge_matches(
            token_ids,
            &mut matches,
            TokenId::from_idx(3).unwrap(),
            TokenId::from_idx(0).unwrap(),
        );
        assert_eq!(count, 999);

        for (i, idx) in matches[..count].iter().copied().enumerate() {
            assert_eq!(idx, (i * 4 + 3) as u32);
        }

        check(&tokens, 3, 0);
        check(&tokens, 0, 1);
        check(&tokens, 1, 0);
    }

    #[test]
    fn tiny_inputs() {
        check(&[], 0, 1);
        check(&[0], 0, 1);
        check(&[0, 1], 0, 1);
        check(&[1, 0], 0, 1);
        check(&[0, 0], 0, 0);
        check(&[0, 1, 0], 0, 1);
    }

    #[test]
    fn empty_match_buffer_for_empty_input() {
        let count = scan_merge_matches(
            &[],
            &mut [],
            TokenId::from_idx(0).unwrap(),
            TokenId::from_idx(1).unwrap(),
        );
        assert_eq!(count, 0);
    }

    #[test]
    #[should_panic]
    fn undersized_match_buffer_panics() {
        let tokens: Vec<TokenId> = (0..64).map(|i| TokenId::from_idx(i).unwrap()).collect();
        let mut matches = vec![0u32; 62];
        scan_merge_matches(&tokens, &mut matches, tokens[0], tokens[1]);
    }

    #[test]
    fn no_matches() {
        let mut buf = Vec::new();
        for offset in 0..16 {
            let tokens = aligned(&mut buf, offset, 100);
            tokens
                .iter_mut()
                .enumerate()
                .for_each(|(i, t)| *t = i as u32);
            check(tokens, 5, 7);
            check(tokens, 1000, 1001);
        }
    }

    #[test]
    fn every_position_matches() {
        let mut buf = Vec::new();
        for offset in 0..16 {
            for len in [16, 17, 31, 32, 33, 48, 100] {
                let tokens = aligned(&mut buf, offset, len);
                tokens.fill(7);
                check(tokens, 7, 7);
            }
        }
    }

    #[test]
    fn overlapping_runs() {
        let mut tokens = vec![1u32; 70];
        tokens[10..20].fill(2);
        tokens[40..45].fill(2);
        check(&tokens, 2, 2);
        check(&tokens, 1, 2);
        check(&tokens, 2, 1);
    }

    /// Places a single match at every position for every alignment, covering
    /// matches inside the head, middle, and tail as well as pairs that straddle
    /// the head/middle, vector/vector, and middle/tail boundaries.
    #[test]
    fn single_match_at_each_position() {
        let mut buf = Vec::new();
        for offset in 0..16 {
            for len in [2, 15, 16, 17, 32, 33, 47, 64, 65] {
                for pos in 0..len - 1 {
                    let tokens = aligned(&mut buf, offset, len);
                    tokens.fill(0);
                    tokens[pos] = 1;
                    tokens[pos + 1] = 2;
                    check(tokens, 1, 2);
                }
            }
        }
    }

    /// The final lane of the last vector has no following token when there is
    /// no tail, so it must never be reported as a match.
    #[test]
    fn last_lane_without_tail() {
        let mut buf = Vec::new();
        for len in [16, 32, 64] {
            let tokens = aligned(&mut buf, 0, len);
            tokens.fill(0);
            tokens[len - 1] = 1;
            check(tokens, 1, 2);
            check(tokens, 1, 0);
        }
    }

    #[test]
    fn large_token_values() {
        let big = u32::MAX - 1;
        let mut buf = Vec::new();
        for offset in 0..16 {
            let tokens = aligned(&mut buf, offset, 80);
            for (i, t) in tokens.iter_mut().enumerate() {
                *t = if i % 3 == 0 { big } else { 1 << 31 };
            }
            check(tokens, big, 1 << 31);
            check(tokens, 1 << 31, big);
            check(tokens, 1 << 31, 1 << 31);
        }
    }

    #[test]
    fn random_against_reference() {
        let mut rng = StdRng::seed_from_u64(0x9e37_79b9_7f4a_7c15);
        let mut buf = Vec::new();
        for _ in 0..2000 {
            let offset = rng.random_range(0..16);
            let len = rng.random_range(0..300);
            let alphabet = 1 + rng.random_range(0..4);
            let tokens = aligned(&mut buf, offset, len);
            tokens
                .iter_mut()
                .for_each(|t| *t = rng.random_range(0..alphabet));
            let first = rng.random_range(0..alphabet);
            let second = rng.random_range(0..alphabet);
            check(tokens, first, second);
        }
    }

    #[test]
    fn large_random_buffer() {
        let mut rng = StdRng::seed_from_u64(42);
        let tokens: Vec<u32> = (0..64 * 1024 + 7).map(|_| rng.random_range(0..3)).collect();
        for first in 0..3 {
            for second in 0..3 {
                check(&tokens, first, second);
                check(&tokens[5..], first, second);
            }
        }
    }
}
