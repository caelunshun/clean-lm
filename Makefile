CU_FILES := $(wildcard cleandl/cutile-kernels/*.cu)
ARCHS := sm89 sm90a sm100a sm103a sm107a sm120a sm121a

TARGETS := $(foreach arch,$(ARCHS),$(CU_FILES:.cu=.$(arch).sass))
TEMPS := $(foreach arch,$(ARCHS),$(CU_FILES:.cu=.$(arch).cubin))

NVCC := nvcc
NVCCFLAGS := -O3 -std=c++20 --enable-tile --expt-relaxed-constexpr --tileiras-options "--remark-format=command-line --remarks=all"

all: $(TARGETS)

# $(1): arch name as used in artifact names (e.g. sm89)
# $(2): arch name as passed to nvcc -arch (e.g. sm_89)
define ARCH_RULE
%.$(1).cubin: %.cu
	$$(NVCC) $$(NVCCFLAGS) -arch=$(2) --tilecubin -o $$@ $$<

%.$(1).sass: %.$(1).cubin
	cuobjdump --dump-sass $$< > $$@
endef

$(foreach arch,$(ARCHS),$(eval $(call ARCH_RULE,$(arch),sm_$(patsubst sm%,%,$(arch)))))

.PRECIOUS: $(TEMPS)

clean:
	rm -f $(TARGETS) $(TEMPS)

.PHONY: all clean
