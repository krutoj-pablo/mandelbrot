CXX := nvc++
CXX_ACCFLAGS_GPU := -acc=gpu -gpu=cc86,managed
CXX_ACCFLAGS_CPU := -acc=multicore

CXXFLAGS := -Minfo=accel -fast
CXXLIBS := -lsfml-graphics -lsfml-window -lsfml-system

all: mandelbrot_cpu mandelbrot_gpu

mandelbrot_cpu:
	$(CXX) $(CXXFLAGS) mandelbrot.cpp -o mandelbrot_cpu $(CXXLIBS)

mandelbrot_gpu:
	$(CXX) $(CXXFLAGS) $(CXX_ACCFLAGS_GPU) mandelbrot.cpp -o mandelbrot_gpu $(CXXLIBS)

clean:
	rm -f mandelbrot_*