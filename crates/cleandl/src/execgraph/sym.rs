//! Symbolic shape and layout algebra for tensors whose dimensions may be
//! known only at execution time.

use anyhow::{anyhow, bail, ensure};
use skytens_core::{
    data_type::ElementType,
    shape::{MAX_DIMENSIONS, Shape, Strides},
};
use slotmap::{SecondaryMap, SlotMap, new_key_type};
use smallvec::SmallVec;
use std::{
    fmt,
    ops::{Add, Index, IndexMut, Mul, Sub},
};

new_key_type! {
    /// Symbolic dimension: exact value not necessarily known at graph build time.
    pub struct SymDimId;
}

/// Stores guarantees about a dimension known ahead-of-time.
#[derive(Debug, Clone)]
pub struct SymDim {
    pub name: String,
    /// Representative size used for autotuning.
    pub hint: usize,
    /// Size is guaranteed to be a multiple of exp2(this value).
    pub alignment_log2: u32,
}

#[derive(Debug, Clone, Default)]
pub struct SymDims(SlotMap<SymDimId, SymDim>);

impl SymDims {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, dim: SymDim) -> SymDimId {
        assert!(
            satisfies_alignment(dim.hint, dim.alignment_log2),
            "hint {} of dimension {} violates its alignment guarantee",
            dim.hint,
            dim.name
        );
        self.0.insert(dim)
    }

    pub fn get(&self, id: SymDimId) -> &SymDim {
        &self.0[id]
    }

    pub fn name(&self, id: SymDimId) -> &str {
        &self.0[id].name
    }

    pub fn hint(&self, id: SymDimId) -> usize {
        self.0[id].hint
    }

    pub fn alignment_log2(&self, id: SymDimId) -> u32 {
        self.0[id].alignment_log2
    }

    pub fn iter(&self) -> impl Iterator<Item = (SymDimId, &SymDim)> {
        self.0.iter()
    }
}

fn satisfies_alignment(value: usize, alignment_log2: u32) -> bool {
    value == 0 || value.trailing_zeros() >= alignment_log2
}

/// Runtime values bound to symbolic dimensions.
#[derive(Debug, Clone, Default)]
pub struct DimValues(SecondaryMap<SymDimId, usize>);

impl DimValues {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the previously bound value, if any.
    pub fn bind(&mut self, id: SymDimId, value: usize) -> Option<usize> {
        self.0.insert(id, value)
    }

    pub fn get(&self, id: SymDimId) -> Option<usize> {
        self.0.get(id).copied()
    }

    pub fn contains(&self, id: SymDimId) -> bool {
        self.0.contains_key(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (SymDimId, usize)> {
        self.0.iter().map(|(id, v)| (id, *v))
    }

    fn lookup(&self, id: SymDimId) -> anyhow::Result<usize> {
        self.get(id)
            .ok_or_else(|| anyhow!("symbolic dimension {id:?} has no bound value"))
    }
}

/// `coef * product(atoms)`. Normalized so that structural equality is
/// semantic equality: atoms are sorted, and a zero coefficient has no atoms.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Monomial {
    coef: usize,
    atoms: SmallVec<[SymDimId; 2]>,
}

impl Monomial {
    pub fn new(coef: usize, atoms: impl IntoIterator<Item = SymDimId>) -> Self {
        let mut atoms: SmallVec<[SymDimId; 2]> = atoms.into_iter().collect();
        if coef == 0 {
            atoms.clear();
        }
        atoms.sort_unstable();
        Self { coef, atoms }
    }

    pub fn fixed(n: usize) -> Self {
        Self::new(n, [])
    }

    pub fn one() -> Self {
        Self::fixed(1)
    }

    pub fn zero() -> Self {
        Self::fixed(0)
    }

    pub fn atom(id: SymDimId) -> Self {
        Self::new(1, [id])
    }

    pub fn coef(&self) -> usize {
        self.coef
    }

    /// Sorted multiset of atoms.
    pub fn atoms(&self) -> &[SymDimId] {
        &self.atoms
    }

    pub fn is_fixed(&self) -> bool {
        self.atoms.is_empty()
    }

    pub fn as_fixed(&self) -> Option<usize> {
        self.is_fixed().then_some(self.coef)
    }

    pub fn is_one(&self) -> bool {
        self.as_fixed() == Some(1)
    }

    pub fn is_zero(&self) -> bool {
        self.coef == 0
    }

    pub fn checked_mul(&self, other: &Monomial) -> Option<Monomial> {
        let coef = self.coef.checked_mul(other.coef)?;
        Some(Self::new(
            coef,
            self.atoms.iter().chain(&other.atoms).copied(),
        ))
    }

    /// Returns `self / divisor` if the division is provably exact.
    pub fn checked_div(&self, divisor: &Monomial) -> Option<Monomial> {
        if divisor.coef == 0 {
            return None;
        }
        if self.is_zero() {
            return Some(Self::zero());
        }
        if !self.coef.is_multiple_of(divisor.coef) {
            return None;
        }
        let mut atoms = self.atoms.clone();
        for atom in &divisor.atoms {
            let pos = atoms.iter().position(|a| a == atom)?;
            atoms.remove(pos);
        }
        Some(Self::new(self.coef / divisor.coef, atoms))
    }

    pub fn is_divisible_by(&self, divisor: &Monomial) -> bool {
        self.checked_div(divisor).is_some()
    }

    /// Largest `k` such that the value is provably a multiple of `2^k`.
    pub fn pow2_divisibility(&self, dims: &SymDims) -> u32 {
        if self.coef == 0 {
            return u32::MAX;
        }
        self.atoms
            .iter()
            .fold(self.coef.trailing_zeros(), |acc, &a| {
                acc.saturating_add(dims.alignment_log2(a))
            })
    }

    pub fn is_divisible_by_pow2(&self, log2: u32, dims: &SymDims) -> bool {
        self.pow2_divisibility(dims) >= log2
    }

    fn try_eval(
        &self,
        mut f: impl FnMut(SymDimId) -> anyhow::Result<usize>,
    ) -> anyhow::Result<usize> {
        let mut result = self.coef;
        for &atom in &self.atoms {
            result = result
                .checked_mul(f(atom)?)
                .ok_or_else(|| anyhow!("overflow evaluating symbolic size"))?;
        }
        Ok(result)
    }

    pub fn eval(&self, values: &DimValues) -> anyhow::Result<usize> {
        self.try_eval(|id| values.lookup(id))
    }

    pub fn eval_hint(&self, dims: &SymDims) -> usize {
        self.try_eval(|id| Ok(dims.hint(id)))
            .expect("overflow evaluating size hint")
    }

    pub fn display<'a>(&'a self, dims: &'a SymDims) -> impl fmt::Display + 'a {
        WithDims(self, dims)
    }

    fn fmt_with(&self, f: &mut fmt::Formatter<'_>, dims: &SymDims) -> fmt::Result {
        if self.atoms.is_empty() {
            return write!(f, "{}", self.coef);
        }
        if self.coef != 1 {
            write!(f, "{}*", self.coef)?;
        }
        for (i, &atom) in self.atoms.iter().enumerate() {
            if i > 0 {
                write!(f, "*")?;
            }
            write!(f, "{}", dims.name(atom))?;
        }
        Ok(())
    }
}

impl fmt::Debug for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.coef)?;
        for atom in &self.atoms {
            write!(f, "*{atom:?}")?;
        }
        Ok(())
    }
}

impl From<usize> for Monomial {
    fn from(n: usize) -> Self {
        Self::fixed(n)
    }
}

impl From<SymDimId> for Monomial {
    fn from(id: SymDimId) -> Self {
        Self::atom(id)
    }
}

impl Mul<&Monomial> for &Monomial {
    type Output = Monomial;

    fn mul(self, rhs: &Monomial) -> Monomial {
        self.checked_mul(rhs)
            .expect("monomial coefficient overflow")
    }
}

impl Mul for Monomial {
    type Output = Monomial;

    fn mul(self, rhs: Monomial) -> Monomial {
        &self * &rhs
    }
}

impl Mul<usize> for Monomial {
    type Output = Monomial;

    fn mul(self, rhs: usize) -> Monomial {
        &self * &Monomial::fixed(rhs)
    }
}

impl Mul<usize> for &Monomial {
    type Output = Monomial;

    fn mul(self, rhs: usize) -> Monomial {
        self * &Monomial::fixed(rhs)
    }
}

struct WithDims<'a, T>(&'a T, &'a SymDims);

impl fmt::Display for WithDims<'_, Monomial> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_with(f, self.1)
    }
}

impl fmt::Display for WithDims<'_, Polynomial> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_zero() {
            return write!(f, "0");
        }
        for (i, term) in self.0.0.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            term.fmt_with(f, self.1)?;
        }
        Ok(())
    }
}

impl fmt::Display for WithDims<'_, SymShape> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.0.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            dim.fmt_with(f, self.1)?;
        }
        write!(f, "]")
    }
}

/// Sum of monomials with distinct atom sets, sorted by atoms, no zero terms.
#[derive(Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct Polynomial(Vec<Monomial>);

impl Polynomial {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn terms(&self) -> &[Monomial] {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_fixed(&self) -> Option<usize> {
        match self.0.as_slice() {
            [] => Some(0),
            [term] => term.as_fixed(),
            _ => None,
        }
    }

    pub fn add_monomial(&mut self, term: &Monomial) {
        if term.is_zero() {
            return;
        }
        match self.0.binary_search_by(|t| t.atoms.cmp(&term.atoms)) {
            Ok(i) => {
                self.0[i].coef = self.0[i]
                    .coef
                    .checked_add(term.coef)
                    .expect("polynomial coefficient overflow");
            }
            Err(i) => self.0.insert(i, term.clone()),
        }
    }

    fn atoms(&self) -> impl Iterator<Item = SymDimId> + '_ {
        self.0.iter().flat_map(|t| t.atoms.iter().copied())
    }

    pub fn eval(&self, values: &DimValues) -> anyhow::Result<usize> {
        self.0.iter().try_fold(0usize, |acc, term| {
            acc.checked_add(term.eval(values)?)
                .ok_or_else(|| anyhow!("overflow evaluating symbolic size"))
        })
    }

    pub fn eval_hint(&self, dims: &SymDims) -> usize {
        self.0.iter().fold(0usize, |acc, term| {
            acc.checked_add(term.eval_hint(dims))
                .expect("overflow evaluating size hint")
        })
    }

    pub fn pow2_divisibility(&self, dims: &SymDims) -> u32 {
        self.0
            .iter()
            .map(|t| t.pow2_divisibility(dims))
            .min()
            .unwrap_or(u32::MAX)
    }

    pub fn is_divisible_by_pow2(&self, log2: u32, dims: &SymDims) -> bool {
        self.pow2_divisibility(dims) >= log2
    }

    pub fn display<'a>(&'a self, dims: &'a SymDims) -> impl fmt::Display + 'a {
        WithDims(self, dims)
    }
}

impl From<Monomial> for Polynomial {
    fn from(term: Monomial) -> Self {
        let mut p = Self::zero();
        p.add_monomial(&term);
        p
    }
}

impl From<usize> for Polynomial {
    fn from(n: usize) -> Self {
        Monomial::fixed(n).into()
    }
}

impl Add<&Monomial> for Polynomial {
    type Output = Polynomial;

    fn add(mut self, rhs: &Monomial) -> Polynomial {
        self.add_monomial(rhs);
        self
    }
}

impl Add<Monomial> for Polynomial {
    type Output = Polynomial;

    fn add(self, rhs: Monomial) -> Polynomial {
        self + &rhs
    }
}

impl Add for Polynomial {
    type Output = Polynomial;

    fn add(mut self, rhs: Polynomial) -> Polynomial {
        for term in &rhs.0 {
            self.add_monomial(term);
        }
        self
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct SymShape(SmallVec<[Monomial; MAX_DIMENSIONS]>);

impl SymShape {
    pub fn new(dims: impl IntoIterator<Item = Monomial>) -> Self {
        let dims: SmallVec<_> = dims.into_iter().collect();
        assert!(
            dims.len() <= MAX_DIMENSIONS,
            "rank {} exceeds the maximum of {MAX_DIMENSIONS}",
            dims.len()
        );
        Self(dims)
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[Monomial] {
        &self.0
    }

    pub fn numel(&self) -> Monomial {
        self.0
            .iter()
            .fold(Monomial::one(), |acc, d| acc * d.clone())
    }

    pub fn eval(&self, values: &DimValues) -> anyhow::Result<Shape> {
        Ok(Shape::new(
            self.0
                .iter()
                .map(|d| d.eval(values))
                .collect::<anyhow::Result<Vec<_>>>()?,
        ))
    }

    pub fn eval_hint(&self, dims: &SymDims) -> Shape {
        Shape::new(self.0.iter().map(|d| d.eval_hint(dims)))
    }

    pub fn display<'a>(&'a self, dims: &'a SymDims) -> impl fmt::Display + 'a {
        WithDims(self, dims)
    }
}

impl Index<usize> for SymShape {
    type Output = Monomial;

    fn index(&self, index: usize) -> &Monomial {
        &self.0[index]
    }
}

impl IndexMut<usize> for SymShape {
    fn index_mut(&mut self, index: usize) -> &mut Monomial {
        &mut self.0[index]
    }
}

impl From<Shape> for SymShape {
    fn from(shape: Shape) -> Self {
        Self::new(shape.dims().map(Monomial::fixed))
    }
}

impl FromIterator<Monomial> for SymShape {
    fn from_iter<T: IntoIterator<Item = Monomial>>(iter: T) -> Self {
        Self::new(iter)
    }
}

fn row_major_strides(shape: &SymShape) -> Vec<Monomial> {
    let mut strides = vec![Monomial::one(); shape.rank()];
    let mut stride = Monomial::one();
    for dim in (0..shape.rank()).rev() {
        strides[dim] = stride.clone();
        stride = stride * shape[dim].clone();
    }
    strides
}

/// Number of elements whose total bit size is a multiple of `align_bytes`,
/// as a power of two.
fn element_alignment_log2(align_bytes: usize, element_type: ElementType) -> u32 {
    assert!(align_bytes.is_power_of_two());
    let bits = element_type.bits_per_element();
    assert!(bits.is_power_of_two());
    (align_bytes * 8).ilog2().saturating_sub(bits.ilog2())
}

/// Symbolic view over a buffer. Strides and offset are in elements.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SymLayout {
    pub shape: SymShape,
    pub strides: Vec<Monomial>,
    pub offset: Polynomial,
}

impl SymLayout {
    pub fn new(shape: SymShape, strides: Vec<Monomial>, offset: Polynomial) -> Self {
        assert_eq!(shape.rank(), strides.len());
        Self {
            shape,
            strides,
            offset,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Packed row-major layout. For sub-byte element types, every row stride
    /// must provably be a whole number of bytes, so no padding is required.
    pub fn contiguous_row_major(
        shape: SymShape,
        element_type: ElementType,
        dims: &SymDims,
    ) -> anyhow::Result<Self> {
        let strides = row_major_strides(&shape);
        let bits = element_type.bits_per_element();
        assert!(bits.is_power_of_two());
        if bits < 8 {
            let align_log2 = (8 / bits).ilog2();
            for stride in strides.iter().take(shape.rank().saturating_sub(1)) {
                ensure!(
                    stride.is_divisible_by_pow2(align_log2, dims),
                    "row stride {} of {element_type:?} tensor with shape {} is not provably a multiple of {} elements",
                    stride.display(dims),
                    shape.display(dims),
                    8 / bits
                );
            }
        }
        Ok(Self::new(shape, strides, Polynomial::zero()))
    }

    #[must_use]
    pub fn transposed(&self, dim_a: usize, dim_b: usize) -> Self {
        assert!(dim_a < self.rank());
        assert!(dim_b < self.rank());
        let mut result = self.clone();
        result.shape.0.swap(dim_a, dim_b);
        result.strides.swap(dim_a, dim_b);
        result
    }

    /// Attempts to reshape the logical contents without changing their order.
    /// Returns `None` if the reshape cannot be proven valid for all values of
    /// the symbolic dimensions.
    #[must_use]
    pub fn try_reshaped(&self, target_shape: &SymShape) -> Option<Self> {
        let element_count = self.shape.numel();
        if element_count != target_shape.numel() {
            return None;
        }
        if self.shape == *target_shape {
            return Some(self.clone());
        }

        let mut target_strides = vec![Monomial::zero(); target_shape.rank()];
        if element_count.is_zero() || self.rank() == 0 {
            return Some(Self::new(
                target_shape.clone(),
                row_major_strides(target_shape),
                self.offset.clone(),
            ));
        }

        let mut target_dim = target_shape.rank();
        let mut chunk_base_stride = self.strides[self.rank() - 1].clone();
        let mut source_chunk_elements = Monomial::one();
        let mut target_chunk_elements = Monomial::one();

        for source_dim in (0..self.rank()).rev() {
            source_chunk_elements = source_chunk_elements * self.shape[source_dim].clone();
            let at_chunk_boundary = source_dim == 0
                || (!self.shape[source_dim - 1].is_one()
                    && &source_chunk_elements * &chunk_base_stride != self.strides[source_dim - 1]);

            if at_chunk_boundary {
                while target_dim > 0
                    && (target_chunk_elements != source_chunk_elements
                        || target_shape[target_dim - 1].is_one())
                {
                    target_dim -= 1;
                    target_strides[target_dim] = &target_chunk_elements * &chunk_base_stride;
                    target_chunk_elements = &target_chunk_elements * &target_shape[target_dim];
                    if !source_chunk_elements.is_divisible_by(&target_chunk_elements) {
                        return None;
                    }
                }
                if target_chunk_elements != source_chunk_elements {
                    return None;
                }
                if source_dim > 0 {
                    chunk_base_stride = self.strides[source_dim - 1].clone();
                    source_chunk_elements = Monomial::one();
                    target_chunk_elements = Monomial::one();
                }
            }
        }

        if target_dim != 0 {
            return None;
        }
        Some(Self::new(
            target_shape.clone(),
            target_strides,
            self.offset.clone(),
        ))
    }

    /// Slices `len` elements starting at `start` along a dimension of fixed size.
    pub fn sliced(&self, dim: usize, start: usize, len: usize) -> anyhow::Result<Self> {
        ensure!(
            dim < self.rank(),
            "slice dimension {dim} out of range for rank {}",
            self.rank()
        );
        let Some(size) = self.shape[dim].as_fixed() else {
            bail!(
                "cannot slice dimension {dim} of dynamic size {:?}",
                self.shape[dim]
            );
        };
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow!("invalid slice bounds"))?;
        ensure!(
            end <= size,
            "slice {start}..{end} out of range for dimension {dim} of size {size}"
        );
        let mut result = self.clone();
        result.offset.add_monomial(&(&self.strides[dim] * start));
        result.shape[dim] = Monomial::fixed(len);
        Ok(result)
    }

    /// Whether strides are the packed row-major strides and the offset is zero.
    pub fn is_contiguous_row_major(&self) -> bool {
        self.offset.is_zero() && self.strides == row_major_strides(&self.shape)
    }

    /// Whether the view provably covers a dense range of elements in some
    /// dimension order. Ignores the offset.
    pub fn is_contiguous(&self) -> bool {
        let mut used = vec![false; self.rank()];
        let mut expected = Monomial::one();
        for _ in 0..self.rank() {
            // Size-1 dims don't advance the expected stride, so take them first.
            let next = (0..self.rank())
                .filter(|&i| !used[i] && self.strides[i] == expected)
                .min_by_key(|&i| !self.shape[i].is_one());
            let Some(i) = next else {
                return false;
            };
            used[i] = true;
            expected = expected * self.shape[i].clone();
        }
        true
    }

    /// Returns (shape, strides, offset in elements).
    pub fn eval(&self, values: &DimValues) -> anyhow::Result<(Shape, Strides, usize)> {
        let shape = self.shape.eval(values)?;
        let strides = Strides::new(
            self.strides
                .iter()
                .map(|s| s.eval(values))
                .collect::<anyhow::Result<Vec<_>>>()?,
        );
        Ok((shape, strides, self.offset.eval(values)?))
    }

    pub fn eval_hint(&self, dims: &SymDims) -> (Shape, Strides, usize) {
        (
            self.shape.eval_hint(dims),
            Strides::new(self.strides.iter().map(|s| s.eval_hint(dims))),
            self.offset.eval_hint(dims),
        )
    }

    /// Whether the byte offset of the view is provably a multiple of `align_bytes`.
    pub fn offset_is_aligned(
        &self,
        align_bytes: usize,
        element_type: ElementType,
        dims: &SymDims,
    ) -> bool {
        self.offset
            .is_divisible_by_pow2(element_alignment_log2(align_bytes, element_type), dims)
    }

    /// Whether the byte stride of `dim` is provably a multiple of `align_bytes`.
    pub fn stride_is_aligned(
        &self,
        dim: usize,
        align_bytes: usize,
        element_type: ElementType,
        dims: &SymDims,
    ) -> bool {
        self.strides[dim]
            .is_divisible_by_pow2(element_alignment_log2(align_bytes, element_type), dims)
    }

    /// Number of bytes of storage this view addresses, starting at the
    /// beginning of the underlying buffer (i.e. including the offset).
    pub fn required_bytes(&self, element_type: ElementType) -> SizeExpr {
        let mut elements = SizeExpr::Poly(self.offset.clone()) + SizeExpr::Const(1);
        for (size, stride) in self.shape.dims().iter().zip(&self.strides) {
            elements = elements
                + (SizeExpr::Mono(size.clone()) - SizeExpr::Const(1))
                    * SizeExpr::Mono(stride.clone());
        }
        (elements * SizeExpr::Const(element_type.bits_per_element())).div_ceil(8)
    }
}

/// Evaluation-only expression for buffer byte sizes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum SizeExpr {
    Const(usize),
    Mono(Monomial),
    Poly(Polynomial),
    Add(Box<SizeExpr>, Box<SizeExpr>),
    Mul(Box<SizeExpr>, Box<SizeExpr>),
    /// Saturates at zero.
    Sub(Box<SizeExpr>, Box<SizeExpr>),
    RoundUpTo(Box<SizeExpr>, usize),
    DivCeil(Box<SizeExpr>, usize),
}

impl SizeExpr {
    #[must_use]
    pub fn round_up_to(self, multiple: usize) -> Self {
        Self::RoundUpTo(Box::new(self), multiple)
    }

    #[must_use]
    pub fn div_ceil(self, divisor: usize) -> Self {
        Self::DivCeil(Box::new(self), divisor)
    }

    fn try_eval(
        &self,
        leaf: &impl Fn(&Polynomial) -> anyhow::Result<usize>,
    ) -> anyhow::Result<usize> {
        let overflow = || anyhow!("overflow evaluating buffer size");
        Ok(match self {
            SizeExpr::Const(n) => *n,
            SizeExpr::Mono(m) => leaf(&m.clone().into())?,
            SizeExpr::Poly(p) => leaf(p)?,
            SizeExpr::Add(a, b) => a
                .try_eval(leaf)?
                .checked_add(b.try_eval(leaf)?)
                .ok_or_else(overflow)?,
            SizeExpr::Mul(a, b) => a
                .try_eval(leaf)?
                .checked_mul(b.try_eval(leaf)?)
                .ok_or_else(overflow)?,
            SizeExpr::Sub(a, b) => a.try_eval(leaf)?.saturating_sub(b.try_eval(leaf)?),
            SizeExpr::RoundUpTo(a, n) => {
                ensure!(*n != 0, "round up to multiple of zero");
                a.try_eval(leaf)?
                    .checked_next_multiple_of(*n)
                    .ok_or_else(overflow)?
            }
            SizeExpr::DivCeil(a, n) => {
                ensure!(*n != 0, "division by zero");
                a.try_eval(leaf)?.div_ceil(*n)
            }
        })
    }

    pub fn eval(&self, values: &DimValues) -> anyhow::Result<usize> {
        self.try_eval(&|p| p.eval(values))
    }

    pub fn eval_hint(&self, dims: &SymDims) -> usize {
        self.try_eval(&|p| Ok(p.eval_hint(dims)))
            .expect("failed to evaluate size hint")
    }
}

impl From<usize> for SizeExpr {
    fn from(n: usize) -> Self {
        Self::Const(n)
    }
}

impl From<Monomial> for SizeExpr {
    fn from(m: Monomial) -> Self {
        Self::Mono(m)
    }
}

impl From<Polynomial> for SizeExpr {
    fn from(p: Polynomial) -> Self {
        Self::Poly(p)
    }
}

impl Add for SizeExpr {
    type Output = SizeExpr;

    fn add(self, rhs: SizeExpr) -> SizeExpr {
        SizeExpr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Mul for SizeExpr {
    type Output = SizeExpr;

    fn mul(self, rhs: SizeExpr) -> SizeExpr {
        SizeExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Sub for SizeExpr {
    type Output = SizeExpr;

    fn sub(self, rhs: SizeExpr) -> SizeExpr {
        SizeExpr::Sub(Box::new(self), Box::new(rhs))
    }
}

#[derive(Debug, Clone)]
struct Equation {
    what: String,
    expr: Polynomial,
    value: usize,
}

/// Infers values of symbolic dimensions from a set of equations
/// `expr = value`, typically collected from concrete tensors matched against
/// symbolic layouts.
#[derive(Debug, Clone, Default)]
pub struct Unifier {
    equations: Vec<Equation>,
}

impl Unifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_equation(
        &mut self,
        what: impl Into<String>,
        expr: impl Into<Polynomial>,
        value: usize,
    ) {
        self.equations.push(Equation {
            what: what.into(),
            expr: expr.into(),
            value,
        });
    }

    /// Adds equations matching each shape dim, stride, and the offset of
    /// `layout` against a concrete view. `label` names the tensor in errors.
    pub fn add_layout(
        &mut self,
        label: &str,
        layout: &SymLayout,
        shape: &Shape,
        strides: &Strides,
        offset_elems: usize,
    ) -> anyhow::Result<()> {
        ensure!(
            shape.num_dims() == layout.rank() && strides.num_dims() == layout.rank(),
            "{label}: expected rank {}, got shape {shape:?} with strides {strides:?}",
            layout.rank()
        );
        for (i, dim) in layout.shape.dims().iter().enumerate() {
            self.add_equation(format!("{label} shape[{i}]"), dim.clone(), shape[i]);
        }
        for (i, stride) in layout.strides.iter().enumerate() {
            self.add_equation(format!("{label} stride[{i}]"), stride.clone(), strides[i]);
        }
        self.add_equation(
            format!("{label} offset"),
            layout.offset.clone(),
            offset_elems,
        );
        Ok(())
    }

    /// Binds every atom appearing in the equations, then verifies that all
    /// equations hold and all bound values satisfy their alignment guarantees.
    /// Existing bindings in `values` are respected.
    pub fn solve(&self, values: &mut DimValues, dims: &SymDims) -> anyhow::Result<()> {
        loop {
            let mut progress = false;
            for eq in &self.equations {
                if self.try_solve_one(eq, values, dims)? {
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }

        let mut atoms: Vec<SymDimId> = self
            .equations
            .iter()
            .flat_map(|eq| eq.expr.atoms())
            .collect();
        atoms.sort_unstable();
        atoms.dedup();

        for eq in &self.equations {
            if let Some(atom) = eq.expr.atoms().find(|&a| !values.contains(a)) {
                bail!(
                    "cannot infer dimension {} (appears in {}: {})",
                    dims.name(atom),
                    eq.what,
                    eq.expr.display(dims)
                );
            }
        }

        for eq in &self.equations {
            let actual = eq.expr.eval(values)?;
            if actual != eq.value {
                if eq.expr.as_fixed().is_some() {
                    bail!("{}: expected {actual}, got {}", eq.what, eq.value);
                }
                bail!(
                    "{}: expected {} = {actual}{}, got {}",
                    eq.what,
                    eq.expr.display(dims),
                    format_bindings(&eq.expr, values, dims),
                    eq.value
                );
            }
        }

        for atom in atoms {
            let value = values.get(atom).unwrap();
            let alignment_log2 = dims.alignment_log2(atom);
            ensure!(
                satisfies_alignment(value, alignment_log2),
                "dimension {} = {value} violates its alignment guarantee (must be a multiple of {})",
                dims.name(atom),
                1u128 << alignment_log2.min(127)
            );
        }

        Ok(())
    }

    /// Solves `eq` for its single unbound atom, if it has exactly one.
    fn try_solve_one(
        &self,
        eq: &Equation,
        values: &mut DimValues,
        dims: &SymDims,
    ) -> anyhow::Result<bool> {
        let mut unknown_term = None;
        let mut known_sum = 0usize;
        for term in eq.expr.terms() {
            let unbound = term
                .atoms()
                .iter()
                .filter(|&&a| !values.contains(a))
                .count();
            if unbound == 0 {
                known_sum = known_sum
                    .checked_add(term.eval(values)?)
                    .ok_or_else(|| anyhow!("{}: overflow evaluating symbolic size", eq.what))?;
            } else if unbound == 1 && unknown_term.is_none() {
                unknown_term = Some(term);
            } else {
                return Ok(false);
            }
        }
        let Some(term) = unknown_term else {
            return Ok(false);
        };

        let atom = *term.atoms().iter().find(|&&a| !values.contains(a)).unwrap();
        let mut divisor = term.coef();
        for &a in term.atoms() {
            if a != atom {
                divisor = divisor
                    .checked_mul(values.get(a).unwrap())
                    .ok_or_else(|| anyhow!("{}: overflow evaluating symbolic size", eq.what))?;
            }
        }
        let Some(remainder) = eq.value.checked_sub(known_sum) else {
            bail!(
                "{}: expected {}, got {}, which is less than the known part {known_sum}",
                eq.what,
                eq.expr.display(dims),
                eq.value
            );
        };
        if divisor == 0 {
            // The atom is multiplied by zero, so this equation does not determine it.
            return Ok(false);
        }
        ensure!(
            remainder.is_multiple_of(divisor),
            "{}: expected {}, got {}; cannot solve for dimension {} since {remainder} is not divisible by {divisor}",
            eq.what,
            eq.expr.display(dims),
            eq.value,
            dims.name(atom)
        );
        values.bind(atom, remainder / divisor);
        Ok(true)
    }
}

fn format_bindings(expr: &Polynomial, values: &DimValues, dims: &SymDims) -> String {
    let mut atoms: Vec<SymDimId> = expr.atoms().collect();
    atoms.sort_unstable();
    atoms.dedup();
    if atoms.is_empty() {
        return String::new();
    }
    let bindings: Vec<String> = atoms
        .iter()
        .map(|&a| format!("{} = {}", dims.name(a), values.get(a).unwrap()))
        .collect();
    format!(" (with {})", bindings.join(", "))
}

/// Unifies a single symbolic layout with a concrete view. Compose multiple
/// layouts with [`Unifier`] or by calling this repeatedly with the same `values`.
pub fn unify(
    layout: &SymLayout,
    concrete_shape: &Shape,
    concrete_strides: &Strides,
    concrete_offset_elems: usize,
    values: &mut DimValues,
    dims: &SymDims,
) -> anyhow::Result<()> {
    let mut unifier = Unifier::new();
    unifier.add_layout(
        "tensor",
        layout,
        concrete_shape,
        concrete_strides,
        concrete_offset_elems,
    )?;
    unifier.solve(values, dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skytens_core::data_type::FloatType;

    const BF16: ElementType = ElementType::Float(FloatType::Bf16);
    const F4: ElementType = ElementType::Float(FloatType::F4E2M1);

    struct Env {
        dims: SymDims,
        b: SymDimId,
        s: SymDimId,
        t: SymDimId,
    }

    fn env() -> Env {
        let mut dims = SymDims::new();
        let b = dims.insert(SymDim {
            name: "batch".into(),
            hint: 4,
            alignment_log2: 0,
        });
        let s = dims.insert(SymDim {
            name: "seq".into(),
            hint: 1024,
            alignment_log2: 3,
        });
        let t = dims.insert(SymDim {
            name: "tokens".into(),
            hint: 7,
            alignment_log2: 0,
        });
        Env { dims, b, s, t }
    }

    fn m(x: impl Into<Monomial>) -> Monomial {
        x.into()
    }

    fn shape(dims: impl IntoIterator<Item = Monomial>) -> SymShape {
        SymShape::new(dims)
    }

    fn values(pairs: &[(SymDimId, usize)]) -> DimValues {
        let mut v = DimValues::new();
        for &(id, x) in pairs {
            v.bind(id, x);
        }
        v
    }

    #[test]
    fn monomial_algebra() {
        let e = env();
        let a = m(e.s) * m(e.b) * 2;
        let b = Monomial::new(2, [e.b, e.s]);
        assert_eq!(a, b);
        assert_eq!(a.coef(), 2);
        assert!(!a.is_fixed());
        assert_eq!(m(6).as_fixed(), Some(6));
        assert!(m(1).is_one());
        assert!(!m(e.s).is_one());
        assert_eq!(m(e.s) * Monomial::zero(), Monomial::zero());
        assert_eq!(Monomial::new(0, [e.s]), Monomial::zero());
        assert_eq!(format!("{}", a.display(&e.dims)), "2*batch*seq");
        assert_eq!(format!("{}", m(e.s).display(&e.dims)), "seq");
        assert_eq!(format!("{}", m(5).display(&e.dims)), "5");

        assert_eq!(a.checked_div(&m(e.s)), Some(m(e.b) * 2));
        assert_eq!(a.checked_div(&(m(e.s) * m(e.s))), None);
        assert_eq!(a.checked_div(&m(4)), None);
        assert!(a.is_divisible_by(&m(e.b)));

        let v = values(&[(e.b, 3), (e.s, 5)]);
        assert_eq!(a.eval(&v).unwrap(), 30);
        assert!(m(e.t).eval(&v).is_err());
        assert!(
            (m(usize::MAX) * m(e.s))
                .eval(&v)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        assert_eq!(a.eval_hint(&e.dims), 2 * 4 * 1024);

        // 12 = 4*3 contributes 2, seq contributes 3.
        let c = m(e.s) * 12;
        assert!(c.is_divisible_by_pow2(5, &e.dims));
        assert!(!c.is_divisible_by_pow2(6, &e.dims));
        assert!(!m(e.b).is_divisible_by_pow2(1, &e.dims));
        assert!(Monomial::zero().is_divisible_by_pow2(60, &e.dims));
    }

    #[test]
    fn polynomial_algebra() {
        let e = env();
        let p = Polynomial::from(m(e.s) * 2) + m(3) + m(e.s) * 4 + Monomial::zero();
        assert_eq!(p.terms().len(), 2);
        assert_eq!(p, Polynomial::from(3) + m(e.s) * 6);
        assert_eq!(format!("{}", p.display(&e.dims)), "3 + 6*seq");
        assert!(Polynomial::zero().is_zero());
        assert!(Polynomial::from(0).is_zero());
        assert_eq!(format!("{}", Polynomial::zero().display(&e.dims)), "0");
        assert_eq!(p.eval(&values(&[(e.s, 2)])).unwrap(), 15);
        assert_eq!(p.as_fixed(), None);
        assert_eq!(Polynomial::from(7).as_fixed(), Some(7));

        let q = Polynomial::from(m(e.s) * 2) + m(16);
        assert!(q.is_divisible_by_pow2(4, &e.dims));
        assert!(!q.is_divisible_by_pow2(5, &e.dims));
    }

    #[test]
    fn contiguous_strides() {
        let e = env();
        let l =
            SymLayout::contiguous_row_major(shape([m(e.b), m(e.s), m(64)]), BF16, &e.dims).unwrap();
        assert_eq!(l.strides, vec![m(e.s) * 64, m(64), m(1)]);
        assert!(l.offset.is_zero());
        assert!(l.is_contiguous_row_major());
        assert!(l.is_contiguous());
        assert_eq!(l.shape.numel(), m(e.b) * m(e.s) * 64);

        let (sh, st, off) = l.eval(&values(&[(e.b, 2), (e.s, 3)])).unwrap();
        assert_eq!(sh, Shape::new([2, 3, 64]));
        assert_eq!(st, Strides::new([192, 64, 1]));
        assert_eq!(off, 0);

        let scalar = SymLayout::contiguous_row_major(SymShape::default(), BF16, &e.dims).unwrap();
        assert!(scalar.strides.is_empty());
    }

    #[test]
    fn contiguous_sub_byte() {
        let e = env();
        assert!(SymLayout::contiguous_row_major(shape([m(e.b), m(64)]), F4, &e.dims).is_ok());
        // seq is a multiple of 8, so rows of seq F4 elements are whole bytes.
        assert!(SymLayout::contiguous_row_major(shape([m(e.b), m(e.s)]), F4, &e.dims).is_ok());
        assert!(SymLayout::contiguous_row_major(shape([m(e.b), m(e.t)]), F4, &e.dims).is_err());
        assert!(SymLayout::contiguous_row_major(shape([m(e.b), m(3)]), F4, &e.dims).is_err());
        assert!(SymLayout::contiguous_row_major(shape([m(e.t)]), F4, &e.dims).is_ok());
        // Inner dims combine to a whole number of bytes.
        assert!(
            SymLayout::contiguous_row_major(shape([m(e.b), m(e.t), m(2)]), F4, &e.dims).is_ok()
        );
    }

    #[test]
    fn transpose() {
        let e = env();
        let l = SymLayout::contiguous_row_major(shape([m(e.s), m(64)]), BF16, &e.dims).unwrap();
        let t = l.transposed(0, 1);
        assert_eq!(t.shape, shape([m(64), m(e.s)]));
        assert_eq!(t.strides, vec![m(1), m(64)]);
        assert!(!t.is_contiguous_row_major());
        assert!(t.is_contiguous());
        assert_eq!(t.transposed(1, 0), l);
    }

    #[test]
    fn reshape_merge_split_dynamic() {
        let e = env();
        let (b, s) = (m(e.b), m(e.s));
        let l3 =
            SymLayout::contiguous_row_major(shape([b.clone(), s.clone(), m(64)]), BF16, &e.dims)
                .unwrap();
        let l2 = l3.try_reshaped(&shape([&b * &s, m(64)])).unwrap();
        assert_eq!(l2.strides, vec![m(64), m(1)]);
        assert!(l2.is_contiguous_row_major());

        let back = l2
            .try_reshaped(&shape([b.clone(), s.clone(), m(64)]))
            .unwrap();
        assert_eq!(back, l3);

        let flat = l3.try_reshaped(&shape([&b * &s * 64])).unwrap();
        assert_eq!(flat.strides, vec![m(1)]);

        let split = SymLayout::contiguous_row_major(shape([s.clone(), m(64)]), BF16, &e.dims)
            .unwrap()
            .try_reshaped(&shape([s.clone(), m(32), m(2)]))
            .unwrap();
        assert_eq!(split.strides, vec![m(64), m(2), m(1)]);

        // Splitting a dynamic dim into itself times fixed factors.
        let l = SymLayout::contiguous_row_major(shape([s.clone() * 2]), BF16, &e.dims).unwrap();
        let r = l.try_reshaped(&shape([s.clone(), m(2)])).unwrap();
        assert_eq!(r.strides, vec![m(2), m(1)]);

        // Element count mismatch.
        assert!(
            l3.try_reshaped(&shape([b.clone(), s.clone(), m(32)]))
                .is_none()
        );
        assert!(
            l3.try_reshaped(&shape([b.clone(), m(e.t), m(64)]))
                .is_none()
        );
    }

    #[test]
    fn reshape_unit_dims() {
        let e = env();
        let s = m(e.s);
        let l = SymLayout::contiguous_row_major(shape([s.clone(), m(64)]), BF16, &e.dims).unwrap();
        let r = l
            .try_reshaped(&shape([m(1), s.clone(), m(1), m(64)]))
            .unwrap();
        assert_eq!(r.strides, vec![s.clone() * 64, m(64), m(64), m(1)]);
        let back = r.try_reshaped(&shape([s.clone(), m(64)])).unwrap();
        assert_eq!(back, l);

        // Unit dims with arbitrary strides are ignored.
        let odd = SymLayout::new(
            shape([s.clone(), m(1), m(64)]),
            vec![m(64), m(12345), m(1)],
            Polynomial::zero(),
        );
        assert_eq!(odd.try_reshaped(&shape([s.clone(), m(64)])).unwrap(), l);
    }

    #[test]
    fn reshape_transposed() {
        let e = env();
        let s = m(e.s);
        let t = SymLayout::contiguous_row_major(shape([s.clone(), m(64)]), BF16, &e.dims)
            .unwrap()
            .transposed(0, 1);
        assert!(t.try_reshaped(&shape([s.clone() * 64])).is_none());
        assert!(t.try_reshaped(&shape([s.clone(), m(64)])).is_none());

        // Splitting within a single chunk still works.
        let r = t.try_reshaped(&shape([m(32), m(2), s.clone()])).unwrap();
        assert_eq!(r.strides, vec![m(2), m(1), m(64)]);

        let b = m(e.b);
        let t3 =
            SymLayout::contiguous_row_major(shape([b.clone(), s.clone(), m(64)]), BF16, &e.dims)
                .unwrap()
                .transposed(0, 1);
        assert!(t3.try_reshaped(&shape([&s * &b, m(64)])).is_none());
        let r = t3
            .try_reshaped(&shape([s.clone(), b.clone(), m(8), m(8)]))
            .unwrap();
        assert_eq!(r.strides, vec![m(64), s.clone() * 64, m(8), m(1)]);
    }

    #[test]
    fn reshape_sliced() {
        let e = env();
        let s = m(e.s);
        let l = SymLayout::contiguous_row_major(shape([s.clone(), m(64)]), BF16, &e.dims)
            .unwrap()
            .sliced(1, 16, 32)
            .unwrap();
        assert!(l.try_reshaped(&shape([s.clone() * 32])).is_none());
        let r = l.try_reshaped(&shape([s.clone(), m(16), m(2)])).unwrap();
        assert_eq!(r.strides, vec![m(64), m(2), m(1)]);
        assert_eq!(r.offset, Polynomial::from(16));
        assert!(!l.is_contiguous());
    }

    #[test]
    fn slicing() {
        let e = env();
        let s = m(e.s);
        let l = SymLayout::contiguous_row_major(shape([m(e.b), m(4), s.clone()]), BF16, &e.dims)
            .unwrap();
        let a = l.sliced(1, 2, 2).unwrap();
        assert_eq!(a.shape, shape([m(e.b), m(2), s.clone()]));
        assert_eq!(a.offset, Polynomial::from(s.clone() * 2));
        assert!(!a.is_contiguous_row_major());
        let a2 = a.sliced(1, 1, 1).unwrap();
        assert_eq!(a2.offset, Polynomial::from(s.clone() * 3));
        assert_eq!(a2.shape[1], m(1));

        assert!(l.sliced(0, 0, 1).is_err());
        assert!(l.sliced(2, 0, 1).is_err());
        assert!(l.sliced(1, 3, 2).is_err());
        assert!(l.sliced(1, usize::MAX, 2).is_err());
        assert!(l.sliced(3, 0, 1).is_err());
        assert!(l.sliced(1, 4, 0).is_ok());

        let (sh, st, off) = a2.eval(&values(&[(e.b, 2), (e.s, 8)])).unwrap();
        assert_eq!(sh, Shape::new([2, 1, 8]));
        assert_eq!(st, Strides::new([32, 8, 1]));
        assert_eq!(off, 24);
    }

    #[test]
    fn alignment_proofs() {
        let e = env();
        let s = m(e.s);
        let l = SymLayout::contiguous_row_major(shape([m(e.t), m(4), s.clone()]), BF16, &e.dims)
            .unwrap();
        // stride[1] = seq elements = seq*2 bytes, seq multiple of 8 -> 16 bytes.
        assert!(l.stride_is_aligned(1, 16, BF16, &e.dims));
        assert!(!l.stride_is_aligned(1, 32, BF16, &e.dims));
        assert!(l.stride_is_aligned(0, 64, BF16, &e.dims));
        assert!(!l.stride_is_aligned(2, 4, BF16, &e.dims));
        assert!(l.stride_is_aligned(2, 2, BF16, &e.dims));
        assert!(l.stride_is_aligned(2, 1, BF16, &e.dims));
        assert!(l.offset_is_aligned(1024, BF16, &e.dims));

        let sl = l.sliced(1, 1, 2).unwrap();
        assert!(sl.offset_is_aligned(16, BF16, &e.dims));
        assert!(!sl.offset_is_aligned(32, BF16, &e.dims));

        let r = SymLayout::contiguous_row_major(shape([m(e.t), m(96)]), BF16, &e.dims).unwrap();
        let sl = r.sliced(1, 8, 8).unwrap();
        assert!(sl.offset_is_aligned(16, BF16, &e.dims));
        assert!(!sl.offset_is_aligned(32, BF16, &e.dims));
        assert!(!r.stride_is_aligned(0, 128, BF16, &e.dims));
        assert!(r.stride_is_aligned(0, 64, BF16, &e.dims));

        // Sub-byte: 32 F4 elements = 16 bytes.
        let f = SymLayout::contiguous_row_major(shape([m(e.t), m(32)]), F4, &e.dims).unwrap();
        assert!(f.stride_is_aligned(0, 16, F4, &e.dims));
        assert!(!f.stride_is_aligned(0, 32, F4, &e.dims));
        assert!(!f.stride_is_aligned(1, 1, F4, &e.dims));
        let f = f.sliced(1, 1, 2).unwrap();
        assert!(!f.offset_is_aligned(1, F4, &e.dims));
        let f2 = SymLayout::contiguous_row_major(shape([m(e.t), m(32)]), F4, &e.dims)
            .unwrap()
            .sliced(1, 2, 2)
            .unwrap();
        assert!(f2.offset_is_aligned(1, F4, &e.dims));
        assert!(!f2.offset_is_aligned(2, F4, &e.dims));
    }

    #[test]
    fn required_bytes_and_size_expr() {
        let e = env();
        let s = m(e.s);
        let v = values(&[(e.s, 10), (e.t, 3)]);
        let l = SymLayout::contiguous_row_major(shape([s.clone(), m(64)]), BF16, &e.dims).unwrap();
        assert_eq!(l.required_bytes(BF16).eval(&v).unwrap(), 1280);
        assert_eq!(l.required_bytes(BF16).eval_hint(&e.dims), 1024 * 64 * 2);

        let sl = l.sliced(1, 16, 8).unwrap();
        // Last element: 9*64 + 16 + 7.
        assert_eq!(
            sl.required_bytes(BF16).eval(&v).unwrap(),
            (9 * 64 + 16 + 7 + 1) * 2
        );

        let f = SymLayout::contiguous_row_major(shape([m(e.t)]), F4, &e.dims).unwrap();
        assert_eq!(f.required_bytes(F4).eval(&v).unwrap(), 2);

        let t = l.transposed(0, 1);
        assert_eq!(t.required_bytes(BF16).eval(&v).unwrap(), 1280);

        let expr = (SizeExpr::from(m(e.t) * 5) + SizeExpr::Const(2)).round_up_to(128);
        assert_eq!(expr.eval(&v).unwrap(), 128);
        assert_eq!(expr.eval_hint(&e.dims), 128);
        let expr = (SizeExpr::from(m(e.s) * 100) * SizeExpr::Const(3)).div_ceil(128);
        assert_eq!(expr.eval(&v).unwrap(), 24);
        let expr = SizeExpr::from(Polynomial::from(m(e.s)) + m(4)) - SizeExpr::Const(100);
        assert_eq!(expr.eval(&v).unwrap(), 0);
        assert!(SizeExpr::from(m(e.b)).eval(&v).is_err());
        assert!(SizeExpr::Const(1).div_ceil(0).eval(&v).is_err());
        assert!(
            (SizeExpr::Const(usize::MAX) + SizeExpr::Const(1))
                .eval(&v)
                .is_err()
        );
    }

    #[test]
    fn unify_success() {
        let e = env();
        let l =
            SymLayout::contiguous_row_major(shape([m(e.b), m(e.s), m(64)]), BF16, &e.dims).unwrap();
        let mut v = DimValues::new();
        unify(
            &l,
            &Shape::new([2, 16, 64]),
            &Strides::new([1024, 64, 1]),
            0,
            &mut v,
            &e.dims,
        )
        .unwrap();
        assert_eq!(v.get(e.b), Some(2));
        assert_eq!(v.get(e.s), Some(16));

        // Atoms only appearing in strides and offsets.
        let sl = SymLayout::contiguous_row_major(shape([m(4), m(e.t) * 2]), BF16, &e.dims)
            .unwrap()
            .sliced(0, 1, 2)
            .unwrap();
        let mut v = DimValues::new();
        unify(
            &sl,
            &Shape::new([2, 10]),
            &Strides::new([10, 1]),
            10,
            &mut v,
            &e.dims,
        )
        .unwrap();
        assert_eq!(v.get(e.t), Some(5));
    }

    #[test]
    fn unify_multi_atom_fixpoint() {
        let e = env();
        let (b, s) = (m(e.b), m(e.s));
        let flat = SymLayout::contiguous_row_major(shape([&b * &s, m(64)]), BF16, &e.dims).unwrap();
        let tokens = SymLayout::contiguous_row_major(shape([b.clone()]), BF16, &e.dims).unwrap();

        let err = unify(
            &flat,
            &Shape::new([48, 64]),
            &Strides::new([64, 1]),
            0,
            &mut DimValues::new(),
            &e.dims,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("cannot infer dimension"), "{err}");

        // `flat` comes first so the fixpoint loop must revisit it.
        let mut u = Unifier::new();
        u.add_layout("x", &flat, &Shape::new([48, 64]), &Strides::new([64, 1]), 0)
            .unwrap();
        u.add_layout("lens", &tokens, &Shape::new([3]), &Strides::new([1]), 0)
            .unwrap();
        let mut v = DimValues::new();
        u.solve(&mut v, &e.dims).unwrap();
        assert_eq!(v.get(e.b), Some(3));
        assert_eq!(v.get(e.s), Some(16));

        // Composing via shared values.
        let mut v = values(&[(e.b, 3)]);
        unify(
            &flat,
            &Shape::new([48, 64]),
            &Strides::new([64, 1]),
            0,
            &mut v,
            &e.dims,
        )
        .unwrap();
        assert_eq!(v.get(e.s), Some(16));
    }

    #[test]
    fn unify_failures() {
        let e = env();
        let s = m(e.s);
        let ss =
            SymLayout::contiguous_row_major(shape([s.clone(), s.clone()]), BF16, &e.dims).unwrap();
        let err = unify(
            &ss,
            &Shape::new([8, 16]),
            &Strides::new([16, 1]),
            0,
            &mut DimValues::new(),
            &e.dims,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("seq"), "{err}");
        assert!(err.contains("16"), "{err}");

        // Contradicts an existing binding.
        let l = SymLayout::contiguous_row_major(shape([s.clone(), m(4)]), BF16, &e.dims).unwrap();
        let err = unify(
            &l,
            &Shape::new([16, 4]),
            &Strides::new([4, 1]),
            0,
            &mut values(&[(e.s, 8)]),
            &e.dims,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("tensor shape[0]: expected seq = 8 (with seq = 8), got 16"),
            "{err}"
        );

        // Wrong stride.
        let err = unify(
            &l,
            &Shape::new([16, 4]),
            &Strides::new([5, 1]),
            0,
            &mut DimValues::new(),
            &e.dims,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("stride[0]"), "{err}");

        // Non-divisible coefficient.
        let l2 = SymLayout::contiguous_row_major(shape([m(e.t) * 3]), BF16, &e.dims).unwrap();
        let err = unify(
            &l2,
            &Shape::new([7]),
            &Strides::new([1]),
            0,
            &mut DimValues::new(),
            &e.dims,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("tokens") && err.contains("divisible"), "{err}");

        // Alignment guarantee: seq must be a multiple of 8.
        let err = unify(
            &l,
            &Shape::new([12, 4]),
            &Strides::new([4, 1]),
            0,
            &mut DimValues::new(),
            &e.dims,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("seq = 12") && err.contains("multiple of 8"),
            "{err}"
        );

        // Rank mismatch.
        assert!(
            unify(
                &l,
                &Shape::new([16]),
                &Strides::new([1]),
                0,
                &mut DimValues::new(),
                &e.dims,
            )
            .is_err()
        );

        // Fixed dims and offsets are checked too.
        assert!(
            unify(
                &l,
                &Shape::new([16, 5]),
                &Strides::new([4, 1]),
                0,
                &mut DimValues::new(),
                &e.dims,
            )
            .is_err()
        );
        assert!(
            unify(
                &l,
                &Shape::new([16, 4]),
                &Strides::new([4, 1]),
                3,
                &mut DimValues::new(),
                &e.dims,
            )
            .is_err()
        );

        // Offset smaller than its known part.
        let sl = SymLayout::contiguous_row_major(shape([m(4), m(e.t)]), BF16, &e.dims)
            .unwrap()
            .sliced(0, 1, 1)
            .unwrap();
        let mut u = Unifier::new();
        u.add_equation("off", sl.offset.clone() + m(100), 5);
        assert!(u.solve(&mut values(&[]), &e.dims).is_err());
    }

    #[test]
    fn reshape_matches_concrete() {
        use crate::tensor::VirtTensorView;

        fn shapes(rank: usize, out: &mut Vec<Vec<usize>>, cur: &mut Vec<usize>) {
            if cur.len() == rank {
                out.push(cur.clone());
                return;
            }
            for d in [0, 1, 2, 3, 4, 6] {
                cur.push(d);
                shapes(rank, out, cur);
                cur.pop();
            }
        }
        let mut all = Vec::new();
        for rank in 0..=3 {
            shapes(rank, &mut all, &mut Vec::new());
        }

        let mut checked = 0;
        for src in &all {
            let base = VirtTensorView::new(Shape::new(src.iter().copied()), BF16);
            let mut views = vec![base.clone()];
            for a in 0..src.len() {
                for b in a + 1..src.len() {
                    views.push(base.transposed(a, b));
                }
            }
            for view in views {
                let sym = SymLayout::new(
                    SymShape::from(view.shape()),
                    view.strides().dims().map(Monomial::fixed).collect(),
                    Polynomial::zero(),
                );
                for target in &all {
                    let target_shape = Shape::new(target.iter().copied());
                    if target_shape.element_count() != view.shape().element_count() {
                        continue;
                    }
                    let concrete = view.try_reshaped(target_shape);
                    let symbolic = sym.try_reshaped(&SymShape::from(target_shape));
                    assert_eq!(
                        concrete.is_some(),
                        symbolic.is_some(),
                        "{view:?} -> {target_shape:?}"
                    );
                    if let (Some(c), Some(s)) = (concrete, symbolic) {
                        let strides: Vec<usize> =
                            s.strides.iter().map(|x| x.as_fixed().unwrap()).collect();
                        assert_eq!(
                            strides,
                            c.strides().dims_slice(),
                            "{view:?} -> {target_shape:?}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000);
    }
}
