import numpy as np
import imageio
import matplotlib.pyplot as plt
from numba import cuda


@cuda.jit
def imageBlur(Pout, Pin, width, height):
    # Really basic image blur using a 3x3 neighborhood
    # TODO: #10 Add more sophisticated blur, e.g., Gaussian blur, etc.
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    CHANNELS = 3

    if col < width and row < height:
        for channel in range(CHANNELS):
            # Initialize the sum for the average
            sum_val = 0
            count = 0
            
            # Loop over the 3x3 neighborhood
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Check image boundaries
                    if 0 <= row + i < height and 0 <= col + j < width:
                        offset = ((row + i) * width + (col + j)) * CHANNELS + channel
                        sum_val += Pin[offset]
                        count += 1
            
            # Set the output pixel value
            Pout[row * width * CHANNELS + col * CHANNELS + channel] = sum_val // count
