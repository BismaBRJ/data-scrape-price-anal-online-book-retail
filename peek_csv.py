# Simple code to peek at the csv

# Imports
from pathlib import Path
import polars as pl

# Constants (settings, paths etc)
CSV_NAME = "reviews_sep_detail" # with or without .csv
CSV_FOLDER_PATH = (
    Path(__file__).parent /
    "dataset_from_html" / "results"
)
PRINT_COLS = ["title", "author"]

# Script

final_path = (CSV_FOLDER_PATH / CSV_NAME).with_suffix(".csv")
print("Opening:", final_path)
print("File exists:", final_path.is_file())
peek_df = pl.read_csv(final_path, glob=False)
print("CSV opened.")
print(peek_df)
print("CSV columns:")
print(peek_df.columns)

for col_name in PRINT_COLS:
    print("Column:", col_name)
    print(pl.Series(peek_df.select(col_name)).to_list())
