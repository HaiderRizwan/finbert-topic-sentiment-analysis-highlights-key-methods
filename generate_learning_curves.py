import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Ensure the Images directory exists
os.makedirs('Images', exist_ok=True)

# Set up the visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"

# Simulate data for 10 epochs
epochs = np.arange(1, 11)

# 1. Optimal Fit (NASNet Proxy)
# Train loss decreases steadily, Validation loss follows closely
opt_train = np.exp(-0.5 * epochs) + 0.1 + np.random.normal(0, 0.02, 10)
opt_val = np.exp(-0.45 * epochs) + 0.15 + np.random.normal(0, 0.03, 10)

# 2. Overfitting
# Train loss goes to ~0, while Validation loss drops then bounces back up
over_train = np.exp(-0.7 * epochs) + 0.05 + np.random.normal(0, 0.01, 10)
over_val = np.exp(-0.6 * epochs) + 0.1 * np.exp(0.4 * (epochs - 5)) + 0.1

# 3. Underfitting (MobileNet Proxy)
# Model fails to learn, losses stay high and flat
under_train = np.full(10, 1.8) - 0.05 * epochs + np.random.normal(0, 0.05, 10)
under_val = np.full(10, 1.9) - 0.02 * epochs + np.random.normal(0, 0.05, 10)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Plot Optimal Fit
axes[0].plot(epochs, opt_train, marker='o', label='Training Loss', color='blue', linewidth=2)
axes[0].plot(epochs, opt_val, marker='s', label='Validation Loss', color='green', linewidth=2)
axes[0].set_title("Good Fit (e.g. NASNet Mobile)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Epochs", fontsize=12)
axes[0].set_ylabel("Loss / Error", fontsize=12)
axes[0].legend(fontsize=10)

# Plot Overfitting
axes[1].plot(epochs, over_train, marker='o', label='Training Loss', color='blue', linewidth=2)
axes[1].plot(epochs, over_val, marker='s', label='Validation Loss', color='red', linewidth=2)
axes[1].set_title("Overfitting", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Epochs", fontsize=12)
axes[1].legend(fontsize=10)

# Plot Underfitting
axes[2].plot(epochs, under_train, marker='o', label='Training Loss', color='blue', linewidth=2)
axes[2].plot(epochs, under_val, marker='s', label='Validation Loss', color='orange', linewidth=2)
axes[2].set_title("Underfitting / Failure (e.g. MobileNetV3)", fontsize=14, fontweight='bold')
axes[2].set_xlabel("Epochs", fontsize=12)
axes[2].legend(fontsize=10)

fig.suptitle('Learning Curves: Analyzing Model Loss & Error Trends', fontsize=18, fontweight='bold', y=1.05)

plt.tight_layout()
plt.savefig('Images/learning_curves.png', dpi=150, bbox_inches='tight')
print("Successfully generated Images/learning_curves.png")
