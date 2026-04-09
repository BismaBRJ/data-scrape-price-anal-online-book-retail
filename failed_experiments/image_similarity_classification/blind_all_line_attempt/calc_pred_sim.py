"""
This script computes similarity scores for all possible
pairs of rows of all given CSVs, and stores the scores in
two CSVs, one for each ground truth
"""

# Imports
print("Importing packages...")
from pathlib import Path
import polars as pl
from tqdm import tqdm
import time
import torch
from vit_cosine_model import my_pred_sim
import numpy as np
print("Imports complete")

# Constants (settings, paths etc)
CSV_PATHS = [ # with or without .csv
    Path(__file__).parent / "embeddings_etc" /
        "selling_detail_embeddings_1nf"
    ,
    Path(__file__).parent / "embeddings_etc" /
        "reviews_sep_detail_embeddings_1nf"
]
KEY_COLS = ["title"]
EMBEDDING_COL = "img_base64"
RESULTS_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
POSITIVE_TRUTH_FILE_NAME = "all_pos_sim_scores" # with or without .npy
NEGATIVE_TRUTH_FILE_NAME = "all_neg_sim_scores"

# Script

print("Reading CSVs...")
list_df = [
    pl.read_csv(path.with_suffix(".csv"), glob=False)
    for path in CSV_PATHS
]
big_df = pl.concat(list_df)
big_n = len(big_df)
print("Total number of rows:", big_n)

pos_scores = []
neg_scores = []
print("Computing similarity scores...")
start_time = time.time()
for idx1 in tqdm(range(big_n)):
    row1 = big_df[idx1]
    embedding1 = torch.tensor(eval(row1[EMBEDDING_COL].item()))
    for idx2 in tqdm(range(idx1 + 1, big_n)):
        row2 = big_df[idx2]
        embedding2 = torch.tensor(eval(row2[EMBEDDING_COL].item()))

        sim_score = my_pred_sim(embedding1, embedding2).item()

        ground_truth = True
        for key in KEY_COLS:
            if (row1[key].item() != row2[key].item()):
                ground_truth = False
                break
        
        if (ground_truth == True):
            # I know "if ground_truth" is shorter, but, readability...
            pos_scores.append(sim_score)
        else:
            neg_scores.append(sim_score)
end_time = time.time()
time_taken = end_time - start_time
print("Total time taken (in seconds):", time_taken)

pos_scores_arr = np.array(pos_scores)
neg_scores_arr = np.array(neg_scores)

print("Saving files at", RESULTS_FOLDER_PATH)
np.save(
    (RESULTS_FOLDER_PATH / POSITIVE_TRUTH_FILE_NAME).with_suffix(".npy"),
    pos_scores_arr
)
np.save(
    (RESULTS_FOLDER_PATH / NEGATIVE_TRUTH_FILE_NAME).with_suffix(".npy"),
    neg_scores_arr
)
print("Saved")
