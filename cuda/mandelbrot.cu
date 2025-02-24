#include <SFML/Graphics.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <cuda_runtime.h>

#include "common.h"

#include <iostream>

__device__ int mandelbrot(float real, float imag, int iterations)
{
    int n;
    float r = 0.0;
    float i = 0.0;

    for(n = 0; n < iterations; n++)
    {
        float r2 = r * r;
        float i2 = i * i;
        if (r2 + i2 > 4.0)
        {
            return n;
        }
        i = 2 * r * i + imag;
        r = r2 - i2 + real;
    }
    return n;
}

__global__ void mandelbrotKernel(int width, int height, int iterations, float scale, float cx, float cy, int *data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float real = cx + (x - width / 2.0f) * scale;
        float imag = cy + (y - height / 2.0f) * scale;

        data[y * width + x] = mandelbrot(real, imag, iterations);
    }
}

sf::Color HSVtoRGB(int H, float S, float V) {
    H = H % 360; // Wrap the hue to be within [0, 360)
    if (H < 0) H += 360;

    float C = V * S; // Chroma
    float X = C * (1 - fabs(fmod(H / 60.0, 2) - 1));
    float m = V - C;

    float rPrime, gPrime, bPrime;

    if (0 <= H && H < 60) {
        rPrime = C;
        gPrime = X;
        bPrime = 0;
    } else if (60 <= H && H < 120) {
        rPrime = X;
        gPrime = C;
        bPrime = 0;
    } else if (120 <= H && H < 180) {
        rPrime = 0;
        gPrime = C;
        bPrime = X;
    } else if (180 <= H && H < 240) {
        rPrime = 0;
        gPrime = X;
        bPrime = C;
    } else if (240 <= H && H < 300) {
        rPrime = X;
        gPrime = 0;
        bPrime = C;
    } else if (300 <= H && H < 360) {
        rPrime = C;
        gPrime = 0;
        bPrime = X;
    } else {
        rPrime = 0;
        gPrime = 0;
        bPrime = 0;
    }

    int R = static_cast<int>((rPrime + m) * 255);
    int G = static_cast<int>((gPrime + m) * 255);
    int B = static_cast<int>((bPrime + m) * 255);

    return sf::Color(R, G, B);
}

void convertPoints(sf::VertexArray &points, int width, int height, int max_iter, int *data)
{
    for(uint16_t y = 0; y < height; y++)
    {
        for(uint16_t x = 0; x < width; x++)
        {
            int value = data[y * width + x];
            points[y * width + x].position = sf::Vector2f(x, y);
            if (value != max_iter)
                points[y * width + x].color = HSVtoRGB(value, 1, 4);
            else
                points[y * width + x].color = sf::Color(0, 0, 0);
        }
    }
}

int main()
{
    constexpr int MAX_ITER = 10000;
    const int width = 1024;
    const int height = 512;

    float initialScale = 0.005f;
    float centerX = -0.5f;
    float centerY = 0.0f;

    sf::RenderWindow window(sf::VideoMode(width, height), "Mandelbrot Set");
    sf::VertexArray points(sf::Points, width * height);

    int *data_d = nullptr;
    size_t data_size = width * height * sizeof(int);
    /* We're assuming here, that we have only one CUDA capable device, thus quering properties only for the first one.*/
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);


    dim3 blockDim(devProp.warpSize, devProp.warpSize); // 32x32 = 1024 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    std::unique_ptr<int[]> data_h(new int[width * height]);

    CUDA_CALL(cudaMalloc((void **)&data_d, data_size));
    mandelbrotKernel<<<gridDim, blockDim>>>(width, height, MAX_ITER, initialScale, centerX, centerY, data_d);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(data_h.get(), data_d, data_size, cudaMemcpyDeviceToHost));


    convertPoints(points, width, height, MAX_ITER, data_h.get());

    while (window.isOpen())
    {
        sf::Event event;

        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            // Zoom in should be done with mouse click
            if (event.type == sf::Event::MouseButtonPressed)
            {
                // On left click zooming in
                if(event.mouseButton.button == sf::Mouse::Left)
                    initialScale *= 0.9;

                // On right click zooming out
                if (event.mouseButton.button == sf::Mouse::Right)
                    initialScale /= 0.9;

                int width = window.getSize().x;
                int height = window.getSize().y;

                int mouseX = event.mouseButton.x;
                int mouseY = event.mouseButton.y;

                centerX = centerX + (mouseX - width / 2.0f) * initialScale;
                centerY = centerY + (mouseY - height / 2.0f) * initialScale;
                mandelbrotKernel<<<gridDim, blockDim>>>(width, height, MAX_ITER, initialScale, centerX, centerY, data_d);
                CUDA_CALL(cudaMemcpy(data_h.get(), data_d,  data_size, cudaMemcpyDeviceToHost));
                convertPoints(points, width, height, MAX_ITER, data_h.get());
            }
            else if (event.type == sf::Event::Resized)
            {
                CUDA_CALL(cudaFree(data_d));

                sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
                window.setView(sf::View(visibleArea));
                int width = event.size.width;
                int height = event.size.height;

                data_size = width * height * sizeof(int);
                data_h.reset(new int[width * height]);

                gridDim.x = (width + blockDim.x - 1) / blockDim.x;
                gridDim.y = (height + blockDim.y - 1) / blockDim.y;

                points.resize(event.size.width * event.size.height);
                CUDA_CALL(cudaMalloc((void **)&data_d, data_size));
                mandelbrotKernel<<<gridDim, blockDim>>>(width, height, MAX_ITER, initialScale, centerX, centerY, data_d);
                CUDA_CALL(cudaMemcpy(data_h.get(), data_d, data_size, cudaMemcpyDeviceToHost));
                convertPoints(points, width, height, MAX_ITER, data_h.get());
            }

        }
        window.clear();
        window.draw(points);
        window.display();
    }
    return 0;
}