import numpy as np
import imageio
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def colorToGrayscaleConvertion( Pout,
                                Pin,
                                width, 
                                height
                                ):
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    CHANNELS = 3

    if col < width and row < height:
        # Get ID offset for the grayscale image
        grayOffset = row * width + col
        # One can think of the RGB image having CHANNELS times more columns than the gray scale image
        rgbOffset = grayOffset * CHANNELS
        r = Pin[rgbOffset]      # Red value
        g = Pin[rgbOffset + 1]  # Green value
        b = Pin[rgbOffset + 2]  # Blue value

        # Perform the rescaling and store it
        grayscale_value = 0.21 * r + 0.71 * g + 0.07 * b
        Pout[grayOffset] = np.uint8( grayscale_value )
