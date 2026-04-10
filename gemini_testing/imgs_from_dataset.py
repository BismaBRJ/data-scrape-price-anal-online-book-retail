# Simple code to "export" base64 image data from the csv

# Imports
from pathlib import Path
import polars as pl
import base64
import ast # for literal eval

# Constants (settings, paths etc)
BOOK_TITLE = "Stochastic Process"
CSV_NAME = "selling_detail" # with or without .csv
TOPDIR = Path(__file__).parent.parent
CSV_FOLDER_PATH = (
    TOPDIR / "dataset_from_html" / "results"
)
IMG_SAVE_PATH = Path(__file__).parent / "images"

# Script

final_path = (CSV_FOLDER_PATH / CSV_NAME).with_suffix(".csv")
print("Opening:", final_path)
print("File exists:", final_path.is_file())
df = pl.read_csv(final_path, glob=False)

rows = df.filter(pl.col("title") == BOOK_TITLE)
book_row = rows[0]
thumb_base64 = book_row["thumb_base64"].to_list()[0]
str_sliders_base64 = book_row["sliders_base64"].to_list()[0]
sliders_base64 = ast.literal_eval(str_sliders_base64)

all_base64 = [thumb_base64, *sliders_base64]

intro_len = len("data:image/jpeg;base64,")
trimmed_base64 = [ img[intro_len:] for img in all_base64 ]
encoded_base64 = [ x.encode("utf-8") for x in trimmed_base64 ]

for idx, img_base64 in enumerate(encoded_base64):
    with open(IMG_SAVE_PATH / f"img{idx}.jpeg", "wb") as file:
        file.write(base64.decodebytes(img_base64))

print("End of script.")

