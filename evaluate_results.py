import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re
import time

# Load .npy file
training_summary = np.load("results/training_summary.npy", allow_pickle=True)

# Filter entries that include 'agent' and extract relevant fields
records = []
for entry in training_summary:
    if 'agent' in entry:
        # Extract goal count from the grid filename
        match = re.match(r"(\d+)_goals", entry['grid'])
        goals = int(match.group(1)) if match else None
        records.append({
            'grid': entry['grid'],
            'agent': entry['agent'],
            'steps': entry['steps_to_goal'],
            'rewards': entry.get('rewards', None),  # In case 'rewards' field is missing
            'goals': goals
        })

# Create DataFrame
df = pd.DataFrame(records)

# Check for missing rewards
if df['rewards'].isnull().any():
    print("Warning: Some entries are missing 'rewards'. These will be excluded.")
    df = df.dropna(subset=['rewards'])

# Compute average rewards per goal count and agent
avg_rewards = df.groupby(['goals', 'agent'])['rewards'].mean().reset_index()

# Pivot table for plotting
pivot_df = avg_rewards.pivot(index='goals', columns='agent', values='rewards')

# Plotting
plt.figure(figsize=(10, 6))
for agent in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[agent], marker='o', label=agent)

plt.title('Average Rewards per Agent by Number of Goals')
plt.xlabel('Number of Goals')
plt.ylabel('Average Reward')
plt.xticks(pivot_df.index)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
