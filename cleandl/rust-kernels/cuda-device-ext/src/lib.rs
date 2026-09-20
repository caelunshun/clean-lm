// --- LLM-generated --- //

#![feature(f16)]
#![no_std]
#![allow(non_camel_case_types)]

#[macro_use]
mod macros;

mod types;

pub use types::bf16::bf16;
pub use types::bf16x2::bf16x2;
pub use types::f16::f16;
pub use types::f16x2::f16x2;

mod primitive {
    pub type Half = f16;
}
