from flask_cors import CORS
import sys
sys.path.append('.')
from cudas.colorToGrayscale import colorToGrayscaleConvertion
from cudas.imageBlur import imageBlur
from flask import Flask, request, jsonify, send_from_directory
from werkzeug import urls
from werkzeug.utils import secure_filename
from PIL import Image
import os
import base64
from io import BytesIO
import numpy as np
import math
from numba import cuda
import logging

print( "SYSTEM.PATH == ", sys.path )  
UPLOAD_FOLDER = 'uploads'
if not os.path.exists( UPLOAD_FOLDER ):
    os.makedirs( UPLOAD_FOLDER )

logging.basicConfig(filename='app.log', level=logging.DEBUG)
app = Flask(__name__, static_folder='../dist')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE"]}})
CORS( app ) # Allow CORS for all domains on all routes


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

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
        app.logger.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 400

def process_with_cuda(image_path, processing_type):
    # Load image and prepare data
    image_data = image_to_rgb_array(image_path)
    height, width, channels = image_data.shape # get height, width and channels for the image directly

    # Flatten the image dat for GPU processing
    flattened_image_data = image_data.flatten()

    # Allocate device memory and copy data to device
    pin_device = cuda.to_device(flattened_image_data)
    pout_device = cuda.device_array((height * width * channels,), dtype=np.uint8) # allocate memory for the output image
    
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


def image_to_rgb_array(image_path):
    # Open the image and convert it to RGB mode
    image = Image.open(image_path).convert('RGB')
    # Convert image data to a numpy array
    image_np = np.array(image)
    
    # check for alpha channel and remove it, if present
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # rleturn it
    return image_np

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
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Hello, World from Flask!"})

@app.route('/test', methods=['OPTIONS'])
def options_for_test():
    return jsonify({}), 200


if __name__ == '__main__':
    port = int( os.environ.get( "PORT", 5000 )) # Use PORT if it's there
    app.run( debug=True, host='0.0.0.0', port=port )
