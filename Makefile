CU_FILES := $(wildcard cleandl/cutile-kernels/*.cu)
TARGETS := $(CU_FILES:.cu=.sass)

NVCC := nvcc
NVCCFLAGS := -O3 -std=c++20 --enable-tile --expt-relaxed-constexpr -arch=sm_120f

all: $(TARGETS)

%.sass: %.cu
	$(NVCC) -o $*.cubin $(NVCCFLAGS) --tilecubin $*.cu
	cuobjdump --dump-sass $*.cubin > $*.sass

clean:
	rm -f $(TARGETS)