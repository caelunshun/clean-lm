CU_FILES := $(wildcard cleandl/cutile-kernels/*.cu)
TARGETS := $(CU_FILES:.cu=.sass)
TEMPS := $(CU_FILES:.cu=.cubin)

NVCC := nvcc
NVCCFLAGS := -O3 -std=c++20 --enable-tile --expt-relaxed-constexpr -arch=sm_100f --tileiras-options "--remark-format=command-line --remarks=all"

all: $(TARGETS)

%.sass: %.cu
	$(NVCC) $(NVCCFLAGS) --tilecubin  -o $*.cubin $*.cu
	cuobjdump --dump-sass $*.cubin > $*.sass

clean:
	rm -f $(TARGETS) $(TEMPS)