fn main() {
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=dylib=cudnn");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
