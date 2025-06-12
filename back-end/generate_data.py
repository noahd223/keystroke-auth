import pandas as pd
import numpy as np
import os
import random
import string

# --- Configuration ---
# The original CSV file with one user's data that will be augmented.
ORIGINAL_DATA_FILE = './keystroke_data/noah_keystroke_data.csv' 
# A directory to save the individual user CSV files.
OUTPUT_DIR = 'keystroke_data'
# --- Personality Configuration ---
# How much variation can each bot have from the original user?
VARIATION_INTENSITY = 0.15 
# How much random noise to add to each individual keystroke?
KEYSTROKE_NOISE = 0.05
# The base chance that a user will make a typo after a correct key press.
BASE_ERROR_RATE = 0.015 # 4% chance of error
# How much can a user's error rate vary from the base?
ERROR_RATE_VARIATION = 0.03 # Can range from 1% to 7%
# Extra pause when a user notices they made a mistake (in ms)
RECOGNITION_PAUSE = 100 

# The features we will modify for the synthetic users.
TIMING_FEATURES = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']

def generate_augmented_data_separated(num_users=50):
    """
    Loads a single user's keystroke data, generates multiple synthetic users
    by applying variations and simulating errors, and saves each as a separate CSV file.
    """
    print("Starting augmented data generation with error simulation...")
    
    # 1. Load the original user's data
    try:
        df_original = pd.read_csv(ORIGINAL_DATA_FILE)
        original_user_name = df_original['user'].unique()[0]
        print(f"Loaded data for original user: '{original_user_name}'")
    except FileNotFoundError:
        print(f"Error: The original data file '{ORIGINAL_DATA_FILE}' was not found.")
        return
    except IndexError:
        print("Error: Could not find a user in the original data file.")
        return

    # 2. Create the output directory if it doesn't already exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: '{OUTPUT_DIR}'")

    # 3. Loop through each new user name to create a separate file
    for i in range(num_users):
        bot_name = f"user_{i+1}"
        print(f"  - Generating data for '{bot_name}'...")
        
        # --- Create a unique "personality" for this bot ---
        bot_profile = {}
        for feature in TIMING_FEATURES:
            multiplier = 1 + (np.random.rand() - 0.5) * 2 * VARIATION_INTENSITY
            bot_profile[feature] = multiplier
        
        # NEW: Assign a unique error rate to this bot
        bot_error_rate = BASE_ERROR_RATE + (np.random.rand() - 0.5) * 2 * ERROR_RATE_VARIATION

        # This list will hold the new, error-prone keystroke data for this bot
        new_bot_keystrokes = []

        # 4. Iterate through the original data to create the new augmented data
        for index, row in df_original.iterrows():
            # Append the original, correct keystroke first
            new_row = row.to_dict()
            new_row['user'] = bot_name

            # Apply the bot's personality and noise
            for feature in TIMING_FEATURES:
                new_row[feature] *= bot_profile[feature] * (1 + np.random.normal(0, KEYSTROKE_NOISE))
            
            new_bot_keystrokes.append(new_row)

            # NEW: Check if this bot makes an error after this keystroke
            if np.random.rand() < bot_error_rate:
                last_event = new_bot_keystrokes[-1]
                
                # --- Simulate the typo keystroke ---
                typo_char = random.choice(string.ascii_lowercase)
                typo_press_time = last_event['release_time'] + (last_event['r2p_time'] * 0.5) # Fast typo
                typo_dwell_time = last_event['dwell_time'] * 0.8 # Errors are often hit quickly
                typo_release_time = typo_press_time + typo_dwell_time
                
                typo_event = last_event.copy() # Start with a copy to inherit timings
                typo_event.update({
                    'key': typo_char,
                    'press_time': typo_press_time,
                    'release_time': typo_release_time,
                    'dwell_time': typo_dwell_time,
                    'p2p_time': typo_press_time - last_event['press_time'],
                    'r2p_time': typo_press_time - last_event['release_time'],
                    'r2r_time': typo_release_time - last_event['release_time'],
                })
                new_bot_keystrokes.append(typo_event)

                # --- Simulate the backspace keystroke ---
                backspace_press_time = typo_release_time + (last_event['r2p_time'] * 0.5) + RECOGNITION_PAUSE
                backspace_dwell_time = last_event['dwell_time'] * 1.2 # Backspace is often held longer
                backspace_release_time = backspace_press_time + backspace_dwell_time
                
                backspace_event = typo_event.copy()
                backspace_event.update({
                    'key': 'Backspace',
                    'press_time': backspace_press_time,
                    'release_time': backspace_release_time,
                    'dwell_time': backspace_dwell_time,
                    'p2p_time': backspace_press_time - typo_event['press_time'],
                    'r2p_time': backspace_press_time - typo_event['release_time'],
                    'r2r_time': backspace_release_time - typo_event['release_time'],
                })
                new_bot_keystrokes.append(backspace_event)

        # 5. Create a DataFrame and save this user's data to their own CSV file
        df_bot = pd.DataFrame(new_bot_keystrokes)
        df_bot[TIMING_FEATURES] = df_bot[TIMING_FEATURES].clip(lower=0)
        
        output_path = os.path.join(OUTPUT_DIR, f"{bot_name}_keystrokes.csv")
        
        try:
            df_bot.to_csv(output_path, index=False)
            print(f"    -> Saved {len(df_bot)} events (including errors) to '{output_path}'")
        except IOError as e:
            print(f"    -> Error saving file for {bot_name}: {e}")

    print("\nAugmented data generation complete.")

if __name__ == '__main__':
    # Make sure your original data file (e.g., 'noah_keystroke_data.csv') 
    # is in the same directory as this script.
    generate_augmented_data_separated()

