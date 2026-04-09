"""
Simple code to see the similarity score of
any two embeddings from any given CSV
"""

# Imports
from pathlib import Path
import polars as pl
import torch
from vit_cosine_model import my_pred_sim

# Constants (settings, paths etc)
CSV_NAME = "reviews_sep_detail_embeddings_1nf" # with or without .csv
CSV_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
EMBEDDING_COL = "img_base64"
ROW_IDX_1 = 0
ROW_IDX_2 = 5

# Script

final_path = (CSV_FOLDER_PATH / CSV_NAME).with_suffix(".csv")
print("Opening:", final_path)
print("File exists:", final_path.is_file())
peek_df = pl.read_csv(final_path, glob=False)
print("CSV opened.")

row1 = peek_df[ROW_IDX_1]
row2 = peek_df[ROW_IDX_2]

print("Row at index", ROW_IDX_1)
print(row1)
print("Row at index", ROW_IDX_2)
print(row2)

embedding1 = torch.tensor(eval(row1[EMBEDDING_COL].item()))
embedding2 = torch.tensor(eval(row2[EMBEDDING_COL].item()))

sim_score = my_pred_sim(embedding1, embedding2)
print("Similarity score:")
print(sim_score)
