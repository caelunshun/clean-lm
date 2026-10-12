set positional-arguments

runner := justfile_directory() / "scripts/nebius-runner.sh"

# compute-sanitizer tool for nebius-sanitize, e.g. `just sanitizer=racecheck nebius-sanitize`.
sanitizer := "memcheck"

# Run `cargo test --release` on the first running instance of the active Nebius profile.
nebius-test *ARGS:
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER={{runner}} \
        cargo test --release "$@"

# Like nebius-test, but under compute-sanitizer. Extra sanitizer flags go in NEBIUS_RUNNER_SANITIZER_ARGS.
nebius-sanitize *ARGS:
    NEBIUS_RUNNER_SANITIZER={{sanitizer}} CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER={{runner}} \
        cargo test --release "$@"
