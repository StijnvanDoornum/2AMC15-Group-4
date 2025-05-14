import subprocess
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

sigma_values = [0.0, 0.1, 0.2, 0.5]
performance_scores = []
results_dir = Path("results")  # assuming this is where .txt files are saved

for sigma in sigma_values:
    print(f"Running for sigma={sigma}")
    command = ["python", "train.py", "grid_configs/A1_grid.npy", "--sigma", str(sigma)]
    subprocess.run(command)

    # Find the most recent .txt file
    txt_files = sorted(results_dir.glob("*.txt"), key=os.path.getmtime, reverse=True)
    if not txt_files:
        print(f"No result file found for sigma={sigma}")
        performance_scores.append(0)
        continue

    latest_txt = txt_files[0]

    # Parse cumulative reward from the .txt file
    with open(latest_txt, "r") as f:
        content = f.read()
        match = re.search(r"cumulative_reward:\s*(-?\d+)", content)
        if match:
            score = int(match.group(1))
        else:
            print(f"No cumulative reward found in {latest_txt}")
            score = 0

    performance_scores.append(score)

# Plotting
plt.plot(sigma_values, performance_scores, marker='o', color='green')
plt.xlabel("Sigma")
plt.ylabel("Cumulative Reward")
plt.title("Effect of Sigma on Agent Performance")
plt.grid(True)
plt.show()