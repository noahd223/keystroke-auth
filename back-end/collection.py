import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Initialization ---
app = Flask(__name__)
# Enable CORS to allow requests from the React front-end
CORS(app) 

# --- Configuration ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'keystroke_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'keystroke_data.csv')

@app.route('/api/save_data', methods=['POST'])
def save_data():
    """
    API endpoint to receive a CSV file from the front-end
    and append its contents to the main data file.
    """
    # 1. Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # 2. Check if the file has a name
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 3. Process the file
    if file:
        try:
            # Read the content of the uploaded file
            content = file.read().decode('utf-8')
            
            # Determine if the main output file already exists and has content
            file_exists = os.path.isfile(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
            
            # Use 'a' for append mode, ensuring newline='' to prevent extra blank rows
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                if file_exists:
                    # If the main file exists, skip the header of the new file
                    # by splitting lines and joining all but the first one.
                    # Add a newline first to separate from existing data.
                    lines = content.strip().split('\n')
                    data_to_append = '\n'.join(lines[1:])
                    f.write('\n' + data_to_append)
                else:
                    # If the file is new, write the whole content including the header
                    f.write(content)

            return jsonify({"message": f"File '{file.filename}' processed and data appended successfully."}), 201

        except Exception as e:
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    return jsonify({"error": "An unknown error occurred"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
