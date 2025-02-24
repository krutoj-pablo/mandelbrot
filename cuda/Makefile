CXX := nvc++
CXXFLAGS := -Minfo=accel -fast
CXXLIBS := -lsfml-graphics -lsfml-window -lsfml-system

all: mandelbrot_cuda

mandelbrot_cuda:
	$(CXX) $(CXXFLAGS) mandelbrot.cu -o mandelbrot_cuda $(CXXLIBS)

clean:
	rm -f mandelbrot_cuda