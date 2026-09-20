use cuda_device::blockIdx_x;
use cuda_device_ext::bf16;
use cuda_macros::{kernel, launch_bounds};

use crate::{
    WARP_SIZE, make_register_slice,
    partition::{
        AllChunksPartitioner, DistributeCoalescedWithinWarpPartitioner, for_each_chunk,
        for_each_chunk_zip2, partition, partition_mut,
    },
    view::{AlignedSlice, AlignedSliceMut},
};

#[kernel]
#[launch_bounds(32, 1)]
unsafe fn copy(x: *const bf16, y: *mut bf16) {
    unsafe {
        let i = blockIdx_x() as usize * 256;
        let x = AlignedSlice::from_ptr(x.add(i), 1024);

        make_register_slice!(x_reg, bf16, 1024 / WARP_SIZE);
        for_each_chunk_zip2(
            partition(x, DistributeCoalescedWithinWarpPartitioner),
            partition_mut(x_reg, AllChunksPartitioner),
            |x, mut dst| dst.copy_from(x),
        );

        for val in x_reg.iter_mut() {
            *val = bf16::from_f32(val.to_f32() * 0.5);
        }

        let y = AlignedSliceMut::from_ptr(y.add(i), 1024);

        for_each_chunk_zip2(
            partition(x_reg.as_ref(), AllChunksPartitioner),
            partition_mut(y, DistributeCoalescedWithinWarpPartitioner),
            |x_reg, mut y| {
                y.copy_from(x_reg);
            },
        );
    }
}
