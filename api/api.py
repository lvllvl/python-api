from ..cuda.colorToGrayscale import colorToGrayscaleConvertion
from ..cuda.imageBlur import imageBlur
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import os
import base64
from io import BytesIO
import numpy as np
import math

UPLOAD_FOLDER = 'uploads'
if not os.path.exists( UPLOAD_FOLDER ):
    os.makedirs( UPLOAD_FOLDER )

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app) # Handling CORS for dev purposes only

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        imageData = request.form.get('imageData')
        processing_type = request.form.get('type')

        # Convert the Base64 encoded data to a PIL Image
        image_data = base64.b64decode(imageData.split(",")[1])
        image = Image.open(BytesIO(image_data))
        
        # Save the image temporarily and process it
        image_path = os.path.join(UPLOAD_FOLDER, "temp_image.png")
        image.save(image_path)
        
        processed_image_path = process_with_cuda(image_path, processing_type)
        
        return send_from_directory(UPLOAD_FOLDER, processed_image_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

def process_with_cuda(image_path, processing_type):
    # Load image and prepare data
    image_data = image_to_rgb_array(image_path)
    height, width = image_data.shape[ 0 ] // 3, image_data.shape[ 1 ] # Assuming a flattened 1d array

    # Allocate device memory and copy data to device
    pin_device = cuda.to_device(image_data)
    pout_device = cuda.device_array((height * width,), dtype=np.uint8)
    
    # Define block and grid dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = int( width / threads_per_block[0])
    blocks_per_grid_y = int( height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch the CUDA kernel
    if processing_type == 'color-to-grayscale':
        colorToGrayscaleConvertion[blocks_per_grid, threads_per_block](pout_device, pin_device, width, height)
        # Copy the processed data back to the host
        processed_image_data = pout_device.copy_to_host().reshape(height, width)
        # Convert the processed data back to an image
        processed_image = Image.fromarray(processed_image_data, 'L')  # 'L' mode is for grayscale
    elif processing_type == 'image-blur':
        imageBlur[blocks_per_grid, threads_per_block](pout_device, pin_device, width, height)
        # Copy the processed data back to the host
        processed_image_data = pout_device.copy_to_host().reshape(height, width, 3)  # 3 channels for RGB
        # Convert the processed data back to an image
        processed_image = Image.fromarray(processed_image_data, 'RGB')  # 'RGB' mode for colored image
    
    processed_image_path = os.path.join(UPLOAD_FOLDER, "processed_image.png")
    processed_image.save(processed_image_path)
    
    return processed_image_path


# def process_with_cuda( image_path, processing_type ):
#     if processing_type == 'color-to-grayscale':
#         # call the CUDA python function for grayscale conversion
#         processed_image_path = colorToGrayscaleConvertion( image_path )
#     elif processing_type == 'image-blur':
#         # call the CUDA python function for image blur
#         processed_image_path = imageBlur( image_path )
#     else:
#         # unexpected processing type
#         raise Exception("Unexpected processing type: {}".format(processing_type))

#     return image_path

def image_to_rgb_array(image_path):
    # Open the image and convert it to RGB mode
    image = Image.open(image_path).convert('RGB')
    # Convert image data to a numpy array
    image_np = np.array(image)
    
    # check for alpha channel and remove it, if present
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # Flatten the array to 1D and return it
    return image_np.flatten()

# Define a route for a basic GET request
@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({"message": "Hello, World!"})

# Define a route for a basic POST request
@app.route('/echo', methods=['POST'])
def echo():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
