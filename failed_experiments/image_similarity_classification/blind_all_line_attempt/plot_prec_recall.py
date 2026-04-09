"""
Plot Precision-Recall from .npy files
"""

# Imports
print("Importing packages...")
from pathlib import Path
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
print("Imports done")

# Constants (settings, paths etc)
NPY_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
POS_TRUTH_NPY_NAME = "all_pos_sim_scores" # with or without .npy
NEG_TRUTH_NPY_NAME = "all_neg_sim_scores"
SAVE_PLOT_PATH = (
    Path(__file__).parent / "prec_recall_test"
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

pos_n = len(pos_arr)
neg_n = len(neg_arr)

all_arr = np.hstack((pos_arr, neg_arr))
truth_arr = np.hstack((np.ones(pos_n), np.zeros(neg_n)))
print("all_arr.shape:", all_arr.shape)
print("truth_arr.shape:", truth_arr.shape)

display = PrecisionRecallDisplay.from_predictions(
    truth_arr, all_arr,
    name="vit_b_32 embeddings", plot_chance_level=True, despine=True
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.savefig(SAVE_PLOT_PATH)
plt.show()
