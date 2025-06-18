import pandas as pd
import numpy as np
import os
import random
import string

# --- Configuration ---
ORIGINAL_DATA_FILE = './keystroke_data/noah_keystroke_data.csv' 
OUTPUT_DIR = 'keystroke_data'

# --- IMPROVED Personality Configuration ---
# More realistic variations between different users
VARIATION_INTENSITY = 0.4  # Increased from 0.15 to 0.4 for more diversity
KEYSTROKE_NOISE = 0.15     # Increased from 0.05 to 0.15 for more variation
BASE_ERROR_RATE = 0.02     # Base error rate
ERROR_RATE_VARIATION = 0.08 # Much larger variation in error rates

# Additional typing personality parameters
TYPING_SPEED_VARIATION = 0.6  # How much typing speed can vary between users
RHYTHM_VARIATION = 0.3        # Variation in typing rhythm patterns
PAUSE_TENDENCY_VARIATION = 0.5 # Some users pause more between words

TIMING_FEATURES = ['dwell_time', 'p2p_time', 'r2p_time', 'r2r_time']

def create_user_personality():
    """Create a more diverse user personality profile."""
    return {
        # Basic timing multipliers
        'dwell_time': 1 + (np.random.rand() - 0.5) * 2 * VARIATION_INTENSITY,
        'p2p_time': 1 + (np.random.rand() - 0.5) * 2 * VARIATION_INTENSITY,
        'r2p_time': 1 + (np.random.rand() - 0.5) * 2 * VARIATION_INTENSITY,
        'r2r_time': 1 + (np.random.rand() - 0.5) * 2 * VARIATION_INTENSITY,
        
        # Advanced personality traits
        'overall_speed': 1 + (np.random.rand() - 0.5) * 2 * TYPING_SPEED_VARIATION,
        'rhythm_consistency': 0.5 + np.random.rand() * 0.5,  # 0.5 to 1.0
        'pause_tendency': np.random.rand(),  # 0 to 1
        'error_rate': BASE_ERROR_RATE + (np.random.rand() - 0.5) * 2 * ERROR_RATE_VARIATION,
        
        # Different typing patterns
        'fast_key_bias': np.random.choice(['vowels', 'consonants', 'none']),
        'hand_dominance': np.random.choice(['left', 'right', 'balanced']),
    }

def apply_personality_to_keystroke(row, personality, position_in_sequence):
    """Apply personality traits to a single keystroke."""
    new_row = row.copy()
    
    # Apply basic timing multipliers
    for feature in TIMING_FEATURES:
        base_multiplier = personality[feature]
        
        # Add rhythm variation based on position
        rhythm_factor = 1 + np.sin(position_in_sequence * personality['rhythm_consistency']) * 0.1
        
        # Apply overall speed
        speed_factor = personality['overall_speed']
        
        # Add noise
        noise_factor = 1 + np.random.normal(0, KEYSTROKE_NOISE)
        
        new_row[feature] *= base_multiplier * rhythm_factor * speed_factor * noise_factor
    
    # Add pauses for some users
    if personality['pause_tendency'] > 0.7 and np.random.rand() < 0.1:
        new_row['r2p_time'] *= (1 + personality['pause_tendency'])
    
    # Apply hand dominance effects
    key = row['key'].lower()
    if personality['hand_dominance'] == 'left' and key in 'qwertasdfgzxcv':
        new_row['dwell_time'] *= 0.9  # Slightly faster for dominant hand
    elif personality['hand_dominance'] == 'right' and key in 'yuiophjklbnm':
        new_row['dwell_time'] *= 0.9
    
    return new_row

def generate_synthetic_data(num_users=100, original_data_file=ORIGINAL_DATA_FILE):
    """
    Generate more diverse synthetic users with realistic typing patterns.
    """
    print("Starting improved synthetic data generation...")
    
    try:
        df_original = pd.read_csv(original_data_file)
        original_user_name = df_original['user'].unique()[0]
        print(f"Loaded data for original user: '{original_user_name}'")
    except FileNotFoundError:
        print(f"Error: The original data file '{original_data_file}' was not found.")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Keep the original user's data
    print(f"Keeping original user data...")
    
    for i in range(num_users):
        bot_name = f"{original_user_name}_copy_{i+1}"
        print(f"  - Generating data for '{bot_name}' with enhanced personality...")
        
        # Create a unique personality
        personality = create_user_personality()
        
        new_bot_keystrokes = []
        position = 0
        
        for index, row in df_original.iterrows():
            # Apply personality to keystroke
            new_row = apply_personality_to_keystroke(row.to_dict(), personality, position)
            new_row['user'] = bot_name
            new_bot_keystrokes.append(new_row)
            position += 1
            
            # Simulate errors with this user's error rate
            if np.random.rand() < personality['error_rate']:
                # Create typo
                typo_char = random.choice(string.ascii_lowercase)
                typo_row = new_row.copy()
                typo_row.update({
                    'key': typo_char,
                    'press_time': new_row['release_time'] + new_row['r2p_time'] * 0.3,
                    'dwell_time': new_row['dwell_time'] * 0.7,
                })
                typo_row['release_time'] = typo_row['press_time'] + typo_row['dwell_time']
                new_bot_keystrokes.append(typo_row)
                
                # Create backspace
                backspace_row = typo_row.copy()
                backspace_row.update({
                    'key': 'Backspace',
                    'press_time': typo_row['release_time'] + 50 + np.random.exponential(100),
                    'dwell_time': new_row['dwell_time'] * 1.3,
                })
                backspace_row['release_time'] = backspace_row['press_time'] + backspace_row['dwell_time']
                new_bot_keystrokes.append(backspace_row)
                
                position += 2
        
        # Save user data
        df_bot = pd.DataFrame(new_bot_keystrokes)
        df_bot[TIMING_FEATURES] = df_bot[TIMING_FEATURES].clip(lower=1.0)  # Minimum 1ms
        
        output_path = os.path.join(OUTPUT_DIR, f"{bot_name}_keystrokes.csv")
        df_bot.to_csv(output_path, index=False)
        print(f"    -> Saved {len(df_bot)} events to '{output_path}'")
        print(f"    -> User personality: speed={personality['overall_speed']:.2f}, error_rate={personality['error_rate']:.3f}")

    print("\nImproved synthetic data generation complete.")

if __name__ == '__main__':
    num_users = int(input("Enter the number of users to generate: "))
    original_data_file = input("Enter the path to the original data file: ")
    generate_synthetic_data(num_users, original_data_file) 