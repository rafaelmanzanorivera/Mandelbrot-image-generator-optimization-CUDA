//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

#include <png.h>

#include <cstring>


#include <chrono>
#include <ctime>
#include <ratio>
#include <cmath>

#define N 1000000
#define RADIOUS 5

using namespace std::chrono;

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
cudaError_t e=cudaGetLastError();                                 \
if(e!=cudaSuccess) {                                              \
printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
exit(0); \
}                                                                 \
}





void copyToGPU(float* h_data, float** d_data, int size)
{
    cudaMalloc((void**)d_data, size * sizeof(float));
    cudaCheckError(); //recomendado chek error despues de cada llamada
    cudaMemset(*d_data, 0, sizeof(float)*size);
    cudaCheckError();
    cudaMemcpy(*d_data, h_data, sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaCheckError();
}


void copyFromGPU(float** h_data, float* d_data, int size)
{
    *h_data = new float[size];
    memset(*h_data, 0, sizeof(float)*size);
    cudaMemcpy(*h_data, d_data, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCheckError();
}




void copyToGPUAsync(float* h_data, float** d_data, int size, cudaStream_t stream)
{
    cudaMalloc((void**)d_data, size * sizeof(float));
    cudaCheckError();
    cudaMemsetAsync(*d_data, 0, sizeof(float)*size,stream);
    cudaCheckError();
    cudaMemcpyAsync(*d_data, h_data, sizeof(float)*size, cudaMemcpyHostToDevice, stream);
    cudaCheckError();
}


void copyFromGPUAsync(float** h_data, float* d_data, int size, cudaStream_t stream)
{
    *h_data = new float[size];
    memset(*h_data, 0, sizeof(float)*size);
    cudaMemcpyAsync(*h_data, d_data, sizeof(float)*size, cudaMemcpyDeviceToHost, stream);
    cudaCheckError();
}



__global__ void mandelbrotKernel(float* d_buffer, int width, int height, float xS, float yS, float rad, int maxIteration)
{
    //float *buffer = (float *) malloc(width * height * sizeof(float));
    if (d_buffer == NULL) {
        //printf("buffer not setted\n");
        return;
    }
    
    int maxIter = 110;
    int linearIndex = threadIdx.x + blockIdx.x * blockDim.x;
    
    int yPos = linearIndex/width;
    int xPos = linearIndex%width;
    
    if (linearIndex >= width*height)
    {
        return;
    }

    
    float yP = (yS-rad) + (2.0f*rad/height)*yPos;
    float xP = (xS-rad) + (2.0f*rad/width)*xPos;
        
    int iteration = 0;
    float x = 0;
    float y = 0;
    float mu = 0;
    
    while (x*x + y*y <= 4 && iteration < maxIter)
    {
        float tmp = x*x - y*y + xP;
        y = 2*x*y + yP;
        x = tmp;
        iteration++;
    }
    
    if (iteration < maxIter)
    {
        float modZ = sqrt(x*x + y*y);
        mu = iteration - (logf(logf(modZ))) / logf(2);
        d_buffer[yPos * width + xPos] = mu;
    }
    else
    {
        d_buffer[yPos * width + xPos] = 0;
    }
    
    return;
}





float* createMandelbrotImageGPUAsync(int width, int height, float xS, float yS, float rad, int maxIteration)
{
    float *d_buffer = NULL;
    float *h_buffer;
    float *result_host_buffer = NULL;
    int numThreadPorBloque = 1024;
    float blocksNeeded = (height*width)/numThreadPorBloque;
    int offset = (height*width)%numThreadPorBloque;
    
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    
    float minMu = maxIteration;
    float maxMu = 0;
    
    if(offset>0) blocksNeeded++;

    //Alloc
    h_buffer = (float *) malloc(width * height * sizeof(float));
    memset(h_buffer,0,sizeof(float)*width*height);
    
    
    copyToGPUAsync(h_buffer, &d_buffer, width*height, stream1);
    cudaCheckError();
    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    
    //Kernel call
    mandelbrotKernel <<< blocksNeeded, numThreadPorBloque,0,stream1 >>> (d_buffer, width,height, xS, yS, rad, maxIteration);
    
    duration<double> time_span = duration_cast<duration<double>>(high_resolution_clock::now() - t1);
    std::cout << "tiempo kernel: " << time_span.count() << "\n";
    cudaCheckError();
    
    //Bring data from GPU
    copyFromGPUAsync(&result_host_buffer, d_buffer, width*height, stream1);

    cudaStreamSynchronize(stream1);
    cudaCheckError();
    cudaFree(d_buffer);
    cudaCheckError();
    
    
    //TODO do on GPU
    for (int i = 0; i< (width * height); i++)
    {
        if (result_host_buffer[i] != 0.0)
        {
            if (result_host_buffer[i] > maxMu) maxMu = result_host_buffer[i];
            if (result_host_buffer[i] < minMu) minMu = result_host_buffer[i];
        }
    }
    
    
    for (int i = 0; i< (width * height); i++)
    {
        result_host_buffer[i] = (result_host_buffer[i] - minMu) / (maxMu - minMu);
    }
    
    return result_host_buffer;
}



float* createMandelbrotImageGPU(int width, int height, float xS, float yS, float rad, int maxIteration)
{
    float *d_buffer = NULL;
    float *h_buffer;
    float *result_host_buffer = NULL;
    int numThreadPorBloque = 1024;
    float blocksNeeded = (height*width)/numThreadPorBloque;
    int offset = (height*width)%numThreadPorBloque;
    
    
    float minMu = maxIteration;
    float maxMu = 0;
    
    if(offset>0) blocksNeeded++;

    
    h_buffer = (float *) malloc(width * height * sizeof(float));
    memset(h_buffer,0,sizeof(float)*width*height);
    
    copyToGPU(h_buffer, &d_buffer,width*height);
    //d_buffer = (float*)CopyArrayToGPU(h_buffer,width*height);
    cudaCheckError();
    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    mandelbrotKernel <<< blocksNeeded, numThreadPorBloque >>> (d_buffer, width,height, xS, yS, rad, maxIteration);
    cudaDeviceSynchronize();
    duration<double> time_span = duration_cast<duration<double>>(high_resolution_clock::now() - t1);
    std::cout << "tiempo kernel: " << time_span.count() << "\n";
    
    cudaCheckError();
    
    copyFromGPU(&result_host_buffer, d_buffer, width*height);
    

    cudaFree(d_buffer);
    cudaCheckError();

    for (int i = 0; i< (width * height); i++)
    {
        if (result_host_buffer[i] != 0.0)
        {
            if (result_host_buffer[i] > maxMu) maxMu = result_host_buffer[i];
            if (result_host_buffer[i] < minMu) minMu = result_host_buffer[i];
        }
    }


    for (int i = 0; i< (width * height); i++)
    {
        result_host_buffer[i] = (result_host_buffer[i] - minMu) / (maxMu - minMu);
    }

    return result_host_buffer;
}




float *createMandelbrotImageSerial(int width, int height, float xS, float yS, float rad, int maxIteration)
{
    float *buffer = (float *) malloc(width * height * sizeof(float));
    if (buffer == NULL) {
        fprintf(stderr, "Could not create image buffer\n");
        return NULL;
    }
    
    int xPos, yPos;
    float minMu = maxIteration;
    float maxMu = 0;

    float mu;

    for (yPos=0 ; yPos<height ; yPos++)
    {
        float yP = (yS-rad) + (2.0f*rad/height)*yPos;

        for (xPos=0 ; xPos<width ; xPos++)
        {
            float xP = (xS-rad) + (2.0f*rad/width)*xPos;

            int iteration = 0;
            float x = 0;
            float y = 0;
            mu = 0;
            
            while (x*x + y*y <= 4 && iteration < maxIteration)
            {
                float tmp = x*x - y*y + xP;
                y = 2*x*y + yP;
                x = tmp;
                iteration++;
                
            }
            
            if (iteration < maxIteration)
            {
                float modZ = sqrt(x*x + y*y);
                mu = iteration - (log(log(modZ))) / log(2);
                if (mu > maxMu) maxMu = mu;
                if (mu < minMu) minMu = mu;
                buffer[yPos * width + xPos] = mu;
            }
            else {
                buffer[yPos * width + xPos] = 0;
            }
        }
    }

    
    // Scale buffer values between 0 and 1
    int count = width * height;
    while (count) {
        count --;
        buffer[count] = (buffer[count] - minMu) / (maxMu - minMu);
    }

    return buffer;
}






inline void setRGB(png_byte *ptr, float val)
{
    int v = (int)(val * 767);
    if (v < 0) v = 0;
    if (v > 767) v = 767;
    int offset = v % 256;
    
    // int r = 0;
    // int g = 1;
    // int b = 2;
    
    int r = 2;
    int g = 1;
    int b = 0;
    
    
    if (v<256) {
        ptr[r] = 0; ptr[g] = 0; ptr[b] = offset;
    }
    else if (v<512) {
        ptr[r] = 0; ptr[g] = offset; ptr[b] = 255-offset;
    }
    else {
        ptr[r] = offset; ptr[g] = 255-offset; ptr[b] = 0;
    }
}



int writeImageSerial(char* filename, int width, int height, float *buffer, char* title)
{
    int code = 0;
    FILE *fp = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep row = NULL;
    
    //filename[strlen(filename)-1] = '\0';
    
    // Open file for writing (binary mode)
    fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file %s for writing\n", filename);
        code = 1;
        goto finalise;
    }
    
    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fprintf(stderr, "Could not allocate write struct\n");
        code = 1;
        goto finalise;
    }
    
    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fprintf(stderr, "Could not allocate info struct\n");
        code = 1;
        goto finalise;
    }
    
    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during png creation\n");
        code = 1;
        goto finalise;
    }
    
    png_init_io(png_ptr, fp);
    
    // Write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    
    // Set title
    if (title != NULL) {
        png_text title_text;
        title_text.compression = PNG_TEXT_COMPRESSION_NONE;
        title_text.key = "Title";
        title_text.text = title;
        png_set_text(png_ptr, info_ptr, &title_text, 1);
    }
    
    png_write_info(png_ptr, info_ptr);
    
    // Allocate memory for one row (3 bytes per pixel - RGB)
    row = (png_bytep) malloc(3 * width * sizeof(png_byte));
    
    // Write image data
    int x, y;
    for (y=0 ; y<height ; y++) {
        for (x=0 ; x<width ; x++) {
            setRGB(&(row[x*3]), buffer[y*width + x]);
        }
        png_write_row(png_ptr, row);
    }
    
    // End write
    png_write_end(png_ptr, NULL);
    
finalise:
    if (fp != NULL) fclose(fp);
    if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    if (row != NULL) free(row);
    
    return code;
}





int main(int argc, char** argv) {
    float *dataOutCPU, *dataOutGPU;
    
    int imageTotalPixels = 1000000;
    
    int width = 20024;//1000;
    int height = 20024;//(1024*2)-10;
    float xS = -0.802;//-0.802;
    float yS = -0.177;//-0.177;
    float rad = 0.011;
    int iter = 110;
    

    printf("Imagen 20024x20024\n");
    
    t1 = high_resolution_clock::now();
    dataOutCPU = createMandelbrotImageSerial(width, height, xS, yS, rad, iter);
    time_span = duration_cast<duration<double>>(high_resolution_clock::now() - t1);
    std::cout << "tiempo CPU: " << time_span.count() << "\n";

    t1 = high_resolution_clock::now();
    dataOutGPU = createMandelbrotImageGPU(width, height, xS, yS, rad, iter);
    time_span = duration_cast<duration<double>>(high_resolution_clock::now() - t1);
    std::cout << "tiempo GPU: " << time_span.count() << "\n";
    
    t1 = high_resolution_clock::now();
    dataOutGPU = createMandelbrotImageGPUAsync(width, height, xS, yS, rad, iter);
    time_span = duration_cast<duration<double>>(high_resolution_clock::now() - t1);
    std::cout << "tiempo GPU Async: " << time_span.count() << "\n";
    
    
    
    char* gpuname = "GPU.png\0";
    char* cpuname = "CPU.png\0";
    
    writeImageSerial(gpuname, width, height, dataOutGPU, "h");
    writeImageSerial(cpuname, width, height, dataOutCPU, "h");
    
}




