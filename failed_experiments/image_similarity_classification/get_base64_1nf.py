"""
A somewhat specific converter to 1st normal form for CSV files,
in this case with base64 strings and lists of those.
I'm sure there's a more modular/reusable way to write this code...
"""

# Imports
from pathlib import Path
import polars as pl

# Constants (settings, paths etc)
CSV_SOURCE_NAME = "reviews_sep_detail" # with or without .csv
CSV_SOURCE_FOLDER_PATH = (
    Path(__file__).parent.parent.parent /
    "dataset_from_html" / "results"
)
CSV_RESULT_NAME = "reviews_sep_detail_base64_1nf"
CSV_RESULT_FOLDER_PATH = (
    Path(__file__).parent / "embeddings_etc"
)
KEY_COLS = ["title", "author"]
SINGLE_IMG_COLS = [] #["thumb_base64"]
MANY_IMG_COLS = ["sliders_base64", "review_imgs_base64"]
WARN_NONEXISTENT_COLUMNS = True

# Script

source_path = (
    CSV_SOURCE_FOLDER_PATH / CSV_SOURCE_NAME
).with_suffix(".csv")
source_df = pl.read_csv(source_path, glob=False)
source_cols = source_df.columns

# I know two list comprehensions would be far more compact, but, efficiency...
key_cols = []
non_key_cols = []
for col in KEY_COLS:
    if col in source_cols:
        key_cols.append(col)
    else:
        non_key_cols.append(col)

single_img_cols = []
non_single_img_cols = []
for col in SINGLE_IMG_COLS:
    if col in source_cols:
        single_img_cols.append(col)
    else:
        non_single_img_cols.append(col)

many_img_cols = []
non_many_img_cols = []
for col in MANY_IMG_COLS:
    if col in source_cols:
        many_img_cols.append(col)
    else:
        non_many_img_cols.append(col)

if WARN_NONEXISTENT_COLUMNS and (
    (len(non_key_cols) != 0) or
    (len(non_single_img_cols) != 0) or
    (len(non_many_img_cols) != 0)
    ):
    print("Warning: the following requested columns do not exist in the CSV.")
    if (len(non_key_cols) != 0):
        print("KEY_COLS:", non_key_cols)
    if (len(non_single_img_cols) != 0):
        print("SINGLE_IMG_COLS:", non_single_img_cols)
    if (len(non_many_img_cols) != 0):
        print("MANY_IMG_COLS:", non_many_img_cols)
    print("They will be ignored.")

result_cols = key_cols + ["img_base64"]
result_dict = {k:[] for k in result_cols}
for old_row_dict in source_df.iter_rows(named=True):
    cur_new_rows = {k:[] for k in result_cols}

    #cur_keys_dict = {k:old_row_dict[k] for k in key_cols}
    cur_keys_dict = {k:v for k,v in old_row_dict.items() if k in key_cols}
    # cur_keys_dict["img_base64"] = old_row_dict["img_base64"]

    cur_single_imgs = [old_row_dict[col] for col in single_img_cols]
    cur_many_imgs = [
        x
        for sublist in
        [eval(old_row_dict[col]) for col in many_img_cols]
        for x in sublist
    ]
    cur_imgs = cur_single_imgs + cur_many_imgs

    cur_imgs_n = len(cur_imgs)
    for k in key_cols:
        cur_new_rows[k] = [old_row_dict[k] for i in range(cur_imgs_n)]
    cur_new_rows["img_base64"] = cur_imgs

    for k,v in cur_new_rows.items():
        result_dict[k] += v

result_df = pl.DataFrame(result_dict)
result_path = (
    CSV_RESULT_FOLDER_PATH / CSV_RESULT_NAME
).with_suffix(".csv")
print("Saving to:")
print(result_path)
result_df.write_csv(result_path)
print("Saved")
