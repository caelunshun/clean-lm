#![no_std]
#![feature(
    abi_ptx,
    nvptx_ext,
    loop_hints,
    stmt_expr_attributes,
    min_adt_const_params,
    gca_min_const_items,
    gca_const_items,
    gca_adts,
    adt_const_params,
    portable_simd,
    inherent_associated_types,
    gca_macroless_args,
    asm_experimental_arch,
    generic_const_items,
    const_cmp,
    const_trait_impl
)]
#![allow(
    incomplete_features,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    // These unidiomatic expressions are needed
    // to get proper loop unrolling in same cases.
    clippy::needless_range_loop,
    clippy::chunks_exact_to_as_chunks,
)]

#[cfg(not(target_feature = "sm_100"))]
compile_error!("rust-kernels requires sm_100 or newer");

// pub mod sky_matmul;
pub mod tile_elementwise;
mod utils;
