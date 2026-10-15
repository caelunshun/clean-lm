//! Helpers for loading from and storing to the shared memory layouts
//! used by the tile_elementwise template in various situations, to/from
//! the corresponding optimized thread partitioning layout.
//!
//! Currently, the kernel statically selects one of two layout classes based on the value of TM:
//!
//! # Vectorized layout
//! Selected when `TN.div_ceil(NUM_WORKER_THREADS)` is a multiple of 8. This allows elements
//! to be loaded/stored using 128-bit instructions on `bf16x8` vectors, where `bf16` type is used.
//! For other types, there are extra considerations (see below).
//!
//! Elements are partitioned across threads along the N dimension according to this pattern:
//!
//! ```
//! N      | 0..8 | 8..16 | 16..24 | ... | 1016..1024 | 1024..1032 | 1032..1040 | etc.
//! Thread | 0    | 1     | 2      | ... | 127        | 0          | 1          | etc.
//! ```
//! For bf16x8 loads and stores in SMEM, the SMEM layout is unswizzled and row-major;
//! this access pattern has zero bank conflicts. By contrast,
//! the naive partitioning where thread `i` gets a contiguous range of elements `i*(N/128)..(i+1)*(N/128)`
//! would produce bank conflicts when N>1024.
//! 
//! For 8-bit elements, we use the same SMEM layout as bf16, accessing it with 64-bit loads/stores.
//! 
//! For f32, we cannot use f32x8 as the load/store type, because 256-bit SMEM operations do not exist.
//! We must use two back-to-back f32x4 operations instead. 
//! However, doing such f32x4 loads/stores on the unswizzled SMEM layout would
//! cause a two-way bank conflict on every access, due to the stride across lanes.
//! The solution is to use the 32B TMA swizzling pattern in SMEM:
//! 
//! ```
//! Logical N-index   | 0..4 | 4..8 | 8..12  | 12..16 | 16..20 | 20..24 | 24..28 | 28..32 | 32..36 | 36..40 | 40..44 | 44..48 | 48..52 | 52..56 | 56..60 | 60..64 | etc.
//! SMEM N-index      | 0..4 | 4..8 | 8..12  | 12..16 | 16..20 | 20..24 | 24..28 | 28..32 | 36..40 | 32..36 | 44..48 | 40..44 | 52..56 | 48..52 | 60..64 | 56..60 | etc.
//! f32x4-in-pair     | 0    | 1    | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | 0      | 1      | etc.
//! Accessing lane    | 0    | 0    | 1      | 1      | 2      | 2      | 3      | 3      | 4      | 4      | 5      | 5      | 6      | 6      | 7      | 7      | etc.
//! ``` 
//! 
//! Each warp issues two operations: one for each lane's f32x4-in-pair-0, then one for each lane's f32x4-in-pair-1.
//! When the hardware splits our 32x128-bit memory operation for each of the two f32x4s
//! into 4 8x128-bit SMEM wavefronts,
//! the swizzle pattern ensures that within each group of 8 lanes, we access all 32 banks.
//!
//! # Fallback layout
//! For when the vectorized layout cannot be used.
//! 
//! Partitioning pattern:
//! ```
//! N      | 0  | 1 | 2 | ... | 127 | 128 | 129 | etc.
//! Thread | 0  | 1 | 2 | ... | 127 | 0   | 1   | etc.
//! ```
//! 
//! For SMEM layouts, we use the same layouts as the vectorized variants, with each row padded to 16 bytes,
//! so that host-side TMA encoding code is unaffected. bf16 operations are done
//! using 16-bit accesses, f32 with 32-bit accesses, and f8 with 8-bit accesses,
//! i.e. all operations are unvectorized. For consistency, f32 keeps 32B swizzling as in the vectorized
//! layout, although the swizzle is not technically necessary.


