TILE_CU_FILES := $(wildcard cleandl/cutile-kernels/*.cu)
CUDA_CU_FILES := $(wildcard cleandl/cuda-kernels/*.cu)
CU_FILES := $(TILE_CU_FILES) $(CUDA_CU_FILES)
ARCHS := sm89 sm90a sm100a sm103a sm120a sm121a
# rust-kernels require TMA (sm_90+).
RUST_ARCHS := $(filter-out sm89,$(ARCHS))

RUST_KERNEL_DIR := cleandl/rust-kernels
RUST_KERNEL_OUT := $(RUST_KERNEL_DIR)/target/kernels
PTX_STD_DIR := ../ptx-std
RUST_KERNEL_DEPS := $(shell find $(RUST_KERNEL_DIR)/src $(PTX_STD_DIR)/src -name '*.rs') \
	$(RUST_KERNEL_DIR)/Cargo.toml $(RUST_KERNEL_DIR)/build.rs $(RUST_KERNEL_DIR)/.cargo/config.toml \
	$(PTX_STD_DIR)/Cargo.toml

TARGETS := $(foreach arch,$(ARCHS),$(CU_FILES:.cu=.$(arch).sass))
TEMPS := $(foreach arch,$(ARCHS),$(CU_FILES:.cu=.$(arch).cubin))

RUST_TARGETS := $(foreach arch,$(RUST_ARCHS),$(RUST_KERNEL_OUT).$(arch).sass)
RUST_TEMPS := $(foreach arch,$(RUST_ARCHS),$(RUST_KERNEL_OUT).$(arch).ptx $(RUST_KERNEL_OUT).$(arch).cubin)

NVCC := nvcc
PTXAS := ptxas
PTXASFLAGS := -O3
NVCCFLAGS := -O3 -std=c++20 --expt-relaxed-constexpr
TILEFLAGS := $(NVCCFLAGS) --enable-tile --tileiras-options "--remark-format=command-line --remarks=all"

all: $(TARGETS) $(RUST_TARGETS)

# $(1): arch name as used in artifact names (e.g. sm89)
# $(2): arch name as passed to nvcc -arch (e.g. sm_89)
define ARCH_RULE
cleandl/cutile-kernels/%.$(1).cubin: cleandl/cutile-kernels/%.cu
	$$(NVCC) $$(TILEFLAGS) -arch=$(2) --tilecubin -o $$@ $$<

cleandl/cuda-kernels/%.$(1).cubin: cleandl/cuda-kernels/%.cu
	$$(NVCC) $$(NVCCFLAGS) -arch=$(2) --cubin -o $$@ $$<

%.$(1).sass: %.$(1).cubin
	cuobjdump --dump-sass $$< > $$@
endef

$(foreach arch,$(ARCHS),$(eval $(call ARCH_RULE,$(arch),sm_$(patsubst sm%,%,$(arch)))))

define RUST_ARCH_RULE
$$(RUST_KERNEL_OUT).$(1).ptx: $$(RUST_KERNEL_DEPS)
	cd $$(RUST_KERNEL_DIR) && cargo +nvptx-ext build --release \
		--target-dir target/$(1) \
		--config 'target.nvptx64-nvidia-cuda.rustflags = ["-C", "target-cpu=$(2)"]'
	cp $$(RUST_KERNEL_DIR)/target/$(1)/nvptx64-nvidia-cuda/release/cleandl_kernels.ptx $$@

$$(RUST_KERNEL_OUT).$(1).cubin: $$(RUST_KERNEL_OUT).$(1).ptx
	$$(PTXAS) $$(PTXASFLAGS) -arch=$(2) -o $$@ $$<

$$(RUST_KERNEL_OUT).$(1).sass: $$(RUST_KERNEL_OUT).$(1).cubin
	cuobjdump --dump-sass $$< > $$@
endef

$(foreach arch,$(RUST_ARCHS),$(eval $(call RUST_ARCH_RULE,$(arch),sm_$(patsubst sm%,%,$(arch)))))

.PRECIOUS: $(TEMPS) $(RUST_TEMPS)

clean:
	rm -f cleandl/cuda-kernels/*.sass cleandl/cuda-kernels/*.cubin
	rm -f cleandl/cutile-kernels/*.sass cleandl/cutile-kernels/*.cubin
	rm -rf cleandl/rust-kernels/target

.PHONY: all clean
