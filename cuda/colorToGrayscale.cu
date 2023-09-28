#include <iostream>
#include <cuda_runtime.h>

#define CHANNELS 3

__global__ void colortoGrayscaleConvertion(
    unsigned char * Pout,
    unsigned char * Pin,
    int width,
    int height)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {

        // Get ID offset for the grayscale image
        int grayOffset = row * width + col;

        // One can think of the RGB image having CHANNEL
        // times more columns than the gray scale image

        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];     // Red value
        unsigned char g = Pin[rgbOffset + 1]; // Green value
        unsigned char b = Pin[rgbOffset + 2]; // Blue value

        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] = static_cast<unsigned char>(round(0.21f * r + 0.71f * g + 0.07f * b));
    }
}
