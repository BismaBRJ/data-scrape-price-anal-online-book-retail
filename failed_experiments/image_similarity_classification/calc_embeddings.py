"""
A script to calculate embeddings all the way from a CSV of base64 images
along a column (i.e. 1st normal form), later saved to CSV of similar format.
"""

# Imports
print("Importing packages...")
from pathlib import Path
import polars as pl
import base64, io
from PIL import Image
from vit_cosine_model import trim_base64_prefix, img_to_encodable, MyEncoder
import time
print("Imports done")

# Constants (settings, paths etc)
CSV_SOURCE_NAME = "reviews_sep_detail_base64_1nf" # with or without .csv
CSV_SOURCE_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
CSV_RESULT_NAME = "reviews_sep_detail_embeddings_1nf"
CSV_RESULT_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
BASE64_COL = "img_base64"

# Script

source_path = (
    CSV_SOURCE_FOLDER_PATH / CSV_SOURCE_NAME
).with_suffix(".csv")
print("Reading source CSV...")
source_df = pl.read_csv(source_path, glob=False)
print("Source CSV read")

encode_img = MyEncoder()

print("Processing CSV...")
start_time = time.time()
result_df = source_df.with_columns(
    pl.col(BASE64_COL).map_elements(
        lambda x:
        str(
            encode_img(
                img_to_encodable(
                    Image.open(io.BytesIO(base64.b64decode(
                        trim_base64_prefix(x)
                    )))
                ).unsqueeze(0)
            )[0].detach().numpy().tolist()
        ),
        return_dtype=str
    )
)
time_taken = time.time() - start_time
print("Done. Time taken (seconds):", time_taken)

result_path = (
    CSV_RESULT_FOLDER_PATH / CSV_RESULT_NAME
).with_suffix(".csv")
print("Saving to:")
print(result_path)
result_df.write_csv(result_path)
print("Saved")
