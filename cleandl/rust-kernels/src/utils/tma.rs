use ptx_std::{
    memory::CachePolicy,
    tma::{TmaDescriptor, TmaReduceOp, tma_reduce_tile, tma_store_tile},
};

/// TMA store with non-const-generic reduction type
#[inline(always)]
pub unsafe fn tma_store_tile_with_reduction<const N: usize>(
    src: *const u8,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    cache_policy: Option<CachePolicy>,
    reduction: Option<TmaReduceOp>,
) {
    unsafe {
        match reduction {
            Some(reduction) => match reduction {
                TmaReduceOp::Add => {
                    tma_reduce_tile::<_, { TmaReduceOp::Add }>(src, tmap, coords, cache_policy)
                }
                TmaReduceOp::Min => {
                    tma_reduce_tile::<_, { TmaReduceOp::Min }>(src, tmap, coords, cache_policy)
                }
                TmaReduceOp::Max => {
                    tma_reduce_tile::<_, { TmaReduceOp::Max }>(src, tmap, coords, cache_policy)
                }
                TmaReduceOp::WrapIncrement => tma_reduce_tile::<_, { TmaReduceOp::WrapIncrement }>(
                    src,
                    tmap,
                    coords,
                    cache_policy,
                ),
                TmaReduceOp::WrapDecrement => tma_reduce_tile::<_, { TmaReduceOp::WrapDecrement }>(
                    src,
                    tmap,
                    coords,
                    cache_policy,
                ),
                TmaReduceOp::And => {
                    tma_reduce_tile::<_, { TmaReduceOp::And }>(src, tmap, coords, cache_policy)
                }
                TmaReduceOp::Or => {
                    tma_reduce_tile::<_, { TmaReduceOp::Or }>(src, tmap, coords, cache_policy)
                }
                TmaReduceOp::Xor => {
                    tma_reduce_tile::<_, { TmaReduceOp::Xor }>(src, tmap, coords, cache_policy)
                }
            },

            None => tma_store_tile(src, tmap, coords, cache_policy),
        }
    }
}
