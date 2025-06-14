import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from ml_pipeline.evaluate_model import KeystrokeAuthenticator

# --- Initialization ---
app = Flask(__name__)
# Enable CORS to allow requests from the React front-end
CORS(app) 

# Initialize the keystroke authenticator
authenticator = KeystrokeAuthenticator()
authenticator.load_model()

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

@app.route('/api/authenticate', methods=['POST'])
def authenticate_user():
    """
    API endpoint to authenticate a user based on their keystroke patterns.
    Expects JSON data with username and keystroke data.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        username = data.get('username')
        keystroke_data = data.get('keystroke_data')
        threshold = data.get('threshold', 0.5)  # Default threshold
        
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        if not keystroke_data or not isinstance(keystroke_data, list):
            return jsonify({"error": "Valid keystroke_data array is required"}), 400
        
        # Convert keystroke data to DataFrame
        df = pd.DataFrame(keystroke_data)
        
        # Check if user needs to be enrolled first
        if username not in authenticator.enrolled_users:
            # Try to load user data from existing files
            user_file = os.path.join('keystroke_data', f"{username}_keystroke_data.csv")
            if os.path.exists(user_file):
                try:
                    user_df = pd.read_csv(user_file)
                    # Group by prompt to create multiple reference sequences
                    reference_sequences = []
                    for prompt, group in user_df.groupby('prompt'):
                        if len(group) >= 10:  # Minimum sequence length
                            reference_sequences.append(group)
                    
                    if reference_sequences:
                        authenticator.enroll_user(username, reference_sequences)
                        print(f"Auto-enrolled user {username} with {len(reference_sequences)} reference sequences")
                    else:
                        return jsonify({
                            "error": f"User {username} has insufficient reference data. Please collect more typing samples first.",
                            "authenticated": False,
                            "confidence": 0.0
                        }), 400
                except Exception as e:
                    return jsonify({
                        "error": f"Error loading reference data for user {username}: {str(e)}",
                        "authenticated": False,
                        "confidence": 0.0
                    }), 400
            else:
                return jsonify({
                    "error": f"User {username} not found. Please collect reference data first.",
                    "authenticated": False,
                    "confidence": 0.0
                }), 400
        
        # Perform authentication
        result = authenticator.authenticate_user(username, df, threshold)
        
        if 'error' in result:
            return jsonify({
                "error": result['error'],
                "authenticated": False,
                "confidence": 0.0
            }), 400
        
        # Convert confidence to percentage
        confidence_percentage = result['max_probability'] * 100
        
        return jsonify({
            "authenticated": result['authenticated'],
            "confidence": confidence_percentage,
            "max_probability": result['max_probability'],
            "avg_probability": result['avg_probability'],
            "threshold": result['threshold'],
            "username": username,
            "message": f"Authentication {'successful' if result['authenticated'] else 'failed'} with {confidence_percentage:.1f}% confidence"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Authentication error: {str(e)}",
            "authenticated": False,
            "confidence": 0.0
        }), 500

@app.route('/api/enroll', methods=['POST'])
def enroll_user():
    """
    API endpoint to enroll a user with reference keystroke patterns.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        username = data.get('username')
        
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        # Load user data from file
        user_file = os.path.join('keystroke_data', f"{username}_keystroke_data.csv")
        if not os.path.exists(user_file):
            return jsonify({"error": f"No data file found for user {username}"}), 400
        
        try:
            user_df = pd.read_csv(user_file)
            # Group by prompt to create multiple reference sequences
            reference_sequences = []
            for prompt, group in user_df.groupby('prompt'):
                if len(group) >= 10:  # Minimum sequence length
                    reference_sequences.append(group)
            
            if not reference_sequences:
                return jsonify({"error": f"Insufficient reference data for user {username}"}), 400
            
            success = authenticator.enroll_user(username, reference_sequences)
            
            if success:
                return jsonify({
                    "message": f"User {username} enrolled successfully with {len(reference_sequences)} reference sequences",
                    "enrolled": True,
                    "reference_sequences": len(reference_sequences)
                }), 200
            else:
                return jsonify({"error": "Failed to enroll user"}), 500
                
        except Exception as e:
            return jsonify({"error": f"Error processing user data: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Enrollment error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
