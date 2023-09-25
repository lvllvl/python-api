from flask import Flask, request, jsonify

app = Flask(__name__)

# ... (previous code for image upload)

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
    app.run(debug=True)
