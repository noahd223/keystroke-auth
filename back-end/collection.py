import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Initialization ---
app = Flask(__name__)
# Enable CORS to allow requests from the React front-end
CORS(app) 

# --- Configuration ---
@app.route('/api/save_data', methods=['POST'])
def save_data():
    """
    API endpoint to receive a CSV file from the front-end
    and append its contents to the main data file.
    Now saves to keystroke_data/{username}_keystroke_data.csv
    """
    # 1. Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # 2. Check if the file has a name
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 3. Extract username from the filename (e.g., noah_keystrokes.csv)
    filename = file.filename
    if not filename or '_keystrokes.csv' not in filename:
        return jsonify({"error": "Invalid filename. Expected format: {username}_keystrokes.csv"}), 400
    
    username = filename.replace('_keystrokes.csv', '')
    if not username:
        return jsonify({"error": "Username could not be determined from filename."}), 400

    # 4. Build output path
    output_dir = os.path.join(os.path.dirname(__file__), 'keystroke_data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{username}_keystroke_data.csv")

    # 5. Process the file
    if file:
        try:
            # Read the content of the uploaded file
            content = file.read().decode('utf-8')
            
            # Determine if the main output file already exists and has content
            file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
            
            # Use 'a' for append mode, ensuring newline='' to prevent extra blank rows
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                if file_exists:
                    # If the main file exists, skip the header of the new file
                    # by splitting lines and joining all but the first one.
                    # Add a newline first to separate from existing data.
                    lines = content.strip().split('\n')
                    if len(lines) > 1:
                        data_to_append = '\n'.join(lines[1:])
                        f.write('\n' + data_to_append)
                else:
                    # If the file is new, write the whole content including the header
                    f.write(content)

            return jsonify({"message": f"File '{file.filename}' processed and data appended successfully for user {username}."}), 201

        except Exception as e:
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    return jsonify({"error": "An unknown error occurred"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
