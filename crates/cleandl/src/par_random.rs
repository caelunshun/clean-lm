use rand::{Rng, RngExt, SeedableRng};
use rand_distr::Distribution;
use rand_xoshiro::Xoshiro128StarStar;
use rayon::prelude::*;

pub fn par_fill_from_distr<T: Copy + Send + Sync>(
    distr: impl Distribution<T> + Send + Sync,
    data: &mut [T],
    rng: &mut impl Rng,
) {
    const CHUNK_SIZE: usize = 65536;
    let num_chunks = data.len().div_ceil(CHUNK_SIZE);
    let mut chunk_rngs = Vec::new();
    let mut current_rng = Xoshiro128StarStar::from_rng(rng);
    for _ in 0..num_chunks {
        chunk_rngs.push(current_rng.clone());
        current_rng.jump();
    }

    data.par_chunks_mut(CHUNK_SIZE)
        .zip(&mut chunk_rngs)
        .for_each(|(chunk, rng)| {
            chunk.iter_mut().for_each(|x| {
                *x = rng.sample(&distr);
            });
        });
}

pub fn par_vec_from_distr<T: Copy + Default + Send + Sync>(
    distr: impl Distribution<T> + Send + Sync,
    len: usize,
    rng: &mut impl Rng,
) -> Vec<T> {
    let mut data = vec![T::default(); len];
    par_fill_from_distr(distr, &mut data, rng);
    data
}
