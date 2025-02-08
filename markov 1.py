import numpy as np
import pandas as pd
from hmmlearn import hmm

# Load the data
df = pd.read_excel("Data (7).xlsx", sheet_name="Keystrokes")

# Select relevant features (drop NaN values if any)
df_filtered = df[['FlightTime', 'KeyHoldTime']].dropna()

# Convert data to numpy array
X = df_filtered.to_numpy()

# Define and train the HMM (assuming 3 hidden states)
num_hidden_states = 3
model = hmm.GaussianHMM(n_components=num_hidden_states, covariance_type="full", n_iter=100, random_state=42)
model.fit(X)

# Predict hidden states for the keystroke data
hidden_states = model.predict(X)

# Add predicted states to the dataframe
df_filtered["HiddenState"] = hidden_states

# Display a sample of the data with hidden states
print(df_filtered.head(100))
