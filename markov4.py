import pandas as pd
import numpy as np
from collections import defaultdict

# Load data from Excel file
file_path = "Data (8).xlsx"  # Update with the correct file path
keystroke_data = pd.read_excel(file_path, sheet_name="Keystrokes")

# Add Dwell Time
def calculate_dwell_time(df):
    df['DwellTime'] = df['KeyHoldTime']
    return df

# Compute Flight Time
def compute_flight_time(df):
    df['FlightTime'] = df['StartTime'].shift(-1) - df['EndTime']
    return df

# Compute Bigram Frequencies
def compute_bigram_frequency(df):
    bigram_counts = defaultdict(int)
    for i in range(len(df) - 1):
        bigram = (df.loc[i, 'KeyPressed'], df.loc[i + 1, 'KeyPressed'])
        bigram_counts[bigram] += 1
    df['BigramFrequency'] = df.apply(lambda row: bigram_counts.get((row['KeyPressed'], row['KeyPressed']), 0), axis=1)
    return df

# Compute Entropy
def compute_entropy(df):
    key_counts = df['KeyPressed'].value_counts(normalize=True)
    df['Entropy'] = -sum(p * np.log2(p) for p in key_counts)
    return df

# Compute KeyPress Speed
def compute_keypress_speed(df):
    df['KeyPressSpeed'] = 1 / (df['FlightTime'] + 1e-6)  # Prevent division by zero
    return df

# Apply Markov Model to compute transition probabilities
def markov_model(df):
    transition_matrix = defaultdict(lambda: defaultdict(int))
    total_transitions = defaultdict(int)
    
    for i in range(len(df) - 1):
        current_key = df.loc[i, 'KeyPressed']
        next_key = df.loc[i + 1, 'KeyPressed']
        transition_matrix[current_key][next_key] += 1
        total_transitions[current_key] += 1
    
    for key in transition_matrix:
        for next_key in transition_matrix[key]:
            transition_matrix[key][next_key] /= total_transitions[key]
    
    df['MarkovTransitionProb'] = df.apply(lambda row: transition_matrix[row['KeyPressed']].get(row['KeyPressed'], 0), axis=1)
    return df

# Apply all transformations
keystroke_data = calculate_dwell_time(keystroke_data)
keystroke_data = compute_flight_time(keystroke_data)
keystroke_data = compute_bigram_frequency(keystroke_data)
keystroke_data = compute_entropy(keystroke_data)
keystroke_data = compute_keypress_speed(keystroke_data)
keystroke_data = markov_model(keystroke_data)

# Save transformed dataset to CSV
keystroke_data.to_csv("keystroke_features.csv", index=False)

# Display transformed DataFrame
print("Dataset saved as 'keystroke_features.csv'")
print(keystroke_data.head())
