"""
Display boxplot of scores from .npy files
"""

# Imports
print("Importing packages...")
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print("Imports done")

# Constants (settings, paths etc)
NPY_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
POS_TRUTH_NPY_NAME = "all_pos_sim_scores" # with or without .npy
NEG_TRUTH_NPY_NAME = "all_neg_sim_scores"
SAVE_PLOT_PATH = (
    Path(__file__).parent / "boxplot_scores"
).with_suffix(".png")

# Script

print("Opening files at", NPY_FOLDER_PATH)
pos_path = (NPY_FOLDER_PATH / POS_TRUTH_NPY_NAME).with_suffix(".npy")
neg_path = (NPY_FOLDER_PATH / NEG_TRUTH_NPY_NAME).with_suffix(".npy")
pos_arr = np.load(pos_path)
neg_arr = np.load(neg_path)
print(".npy files opened")
print("pos_arr.shape:", pos_arr.shape)
print("neg_arr.shape:", neg_arr.shape)

sns.boxplot(data=[pos_arr, neg_arr])
plt.savefig(SAVE_PLOT_PATH)
plt.show()
