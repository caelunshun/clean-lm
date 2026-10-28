use nvvm_builder::nvvm_sys::Arch;

pub fn max_smem(arch: &Arch) -> usize {
    match arch {
        Arch::Compute75
        | Arch::Compute80
        | Arch::Compute87
        | Arch::Compute89
        | Arch::Compute90
        | Arch::Compute90a
        | Arch::Other(_) => panic!("unsupported arch"),
        Arch::Compute100
        | Arch::Compute100a
        | Arch::Compute100f
        | Arch::Compute103
        | Arch::Compute103a
        | Arch::Compute103f
        | Arch::Compute110
        | Arch::Compute110a
        | Arch::Compute110f => 227 * 1024,
        Arch::Compute120
        | Arch::Compute120a
        | Arch::Compute120f
        | Arch::Compute121
        | Arch::Compute121a
        | Arch::Compute121f => 99 * 1024,
    }
}

pub fn reg_bytes_per_sm(_arch: &Arch) -> usize {
    // Currently all supported architectures have 256 KiB of registers.
    256 * 1024
}

pub fn max_cluster_size(arch: &Arch) -> usize {
    match arch {
        Arch::Compute100
        | Arch::Compute100a
        | Arch::Compute100f
        | Arch::Compute103
        | Arch::Compute103a
        | Arch::Compute103f
        | Arch::Compute110
        | Arch::Compute110a
        | Arch::Compute110f => 16,
        Arch::Compute120
        | Arch::Compute120a
        | Arch::Compute120f
        | Arch::Compute121
        | Arch::Compute121a
        | Arch::Compute121f => 8,
        _ => panic!("unsupported arch"),
    }
}
