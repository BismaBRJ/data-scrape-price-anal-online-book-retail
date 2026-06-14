# always set to False unless about to run
USE_MODEL = False
MODEL_NAME = "gemini-3.1-flash-lite-preview"

# Imports
print("Importing libraries etc. ...")
from pathlib import Path # for path stuff
import json
import polars as pl # for dealing with CSV
from pydantic import BaseModel, Field
import numpy as np # for ceil function
import utils_enrich # for RNG, base64
import time # for time.sleep()
from google import genai
import json
import sys # for sys.exit()
print("Imports done")

# getting the API key
# see secrets.json.example if you haven't set up secrets.json

TOPDIR_PATH = Path(__file__).parent.parent
SECRETS_PATH = TOPDIR_PATH / "secrets.json"

print("Reading API key...")
with open(SECRETS_PATH, 'r', encoding="utf-8") as file:
    SECRETS_JSON = json.load(file)

GEMINI_API_KEY = SECRETS_JSON["GEMINI_API_KEY"]
print("API key obtained.")

# read CSV
CSV_FOLDER_PATH = (
        TOPDIR_PATH / "dataset_from_html" / "results"
    )
OLD_CSV_NAME = "reviews_sep_detail"
old_csv_path = (CSV_FOLDER_PATH / OLD_CSV_NAME).with_suffix(".csv") 
old_df = pl.read_csv(old_csv_path, glob=False)
old_df_cols = old_df.columns
old_df_rowlen = len(old_df)

# get reference to thumbnails column on CSV (not deep copy!) 
# this will be the start of building the new CSV
images_df = old_df.select(pl.col("thumb_base64"))

# prompt engineering and JSON output structuring

MY_PROMPT = """
You are an expert at least as good as Google Lens.
Also, you are an experienced, master librarian.
Reply with metadata and any extra information about the book you see.
Extra information includes anything useful from search results.
Cite any sources you use.
"""

class BookInfo(BaseModel):
    title: str = Field(description="Title of the book.")
    authors: list[str] = Field(description="Authors of the book.")
    year: int | None = Field(description="Year of publication.")
    publisher: str | None = Field(description="Publisher of the book.")
    edition: int | None = Field(description="Edition of the book.")
    pages: int | None = Field(description="Number of pages in the book.")
    hard: bool | None = Field(description="True if hardcover, False if paperback.")
    extra_info: dict[str, str] = Field(description="Any extra information.")
    sources: dict[str, str] = Field(description="Sources used and for what.")

# respecting rate limits with random waiting times
# code in utils_enrich.py

n_imgs = old_df_rowlen
MAX_REQS_PER_MIN = 15
init_avg_waiting_time = np.ceil(60/MAX_REQS_PER_MIN)

# seed will be yyyymmddhhmmss at execution
seed_int = utils_enrich.get_datetime_seed()
print("Random seed:", seed_int)

(rng, avg_waiting_time, waiting_times) = utils_enrich.get_floored_waiting_times(
        n_data=n_imgs,
        init_avg=init_avg_waiting_time,
        flooring_window_size=MAX_REQS_PER_MIN,
        the_floor=60,
        seed_int=seed_int
    )

print("Planned average waiting time:", avg_waiting_time)
print("Planned waiting times:")
print(waiting_times)
print("Total waiting time:", waiting_times.sum())

# finally, data enrichment with Gemini

"""
new_df = images_df.with_columns(
        # default empty values
        year = pl.lit(0),
        publisher = pl.lit(""),
        edition = pl.lit(0),
        pages = pl.lit(0),
        hard = pl.lit(False), # assume paperback
        extra_info_sources = pl.lit("") # string, to fit in CSV
    )
"""

if USE_MODEL:
    print("Starting Gemini API client...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    for idx, row in enumerate(images_df.iter_rows()):
        img_txt = row[0]
        img_obj = utils_enrich.get_PIL_from_base64_text(img_txt)

        print(f"Row {idx+1}")
        print(f"Waiting {waiting_times[idx]} seconds...")
        time.sleep(waiting_times[idx])
        success = False
        while (not success):
            print("Calling API...")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[MY_PROMPT, img_obj],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": BookInfo.model_json_schema()
                }
            )
            if (response.text != None):
                result_json = json.loads(response.text)
                success = True
            else:
                # random cooldown before reattempt
                random_waiting_time = rng.exponential(
                        scale=init_avg_waiting_time,
                        size=1
                    )[0]
                backup_waiting_time = (
                        init_avg_waiting_time + random_waiting_time
                    )
                print(f"Failed, waiting {backup_waiting_time} seconds...")
                time.sleep(backup_waiting_time)

        print("Row {idx+1} success!")

    """
    my_imgs = []
    for img_path in my_imgs_paths:
        my_imgs.append(client.files.upload(file=img_path))

    response = client.models.generate_content(
        model=MODEL_NAME
        contents=[MY_PROMPT, *my_imgs],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": BookInfo.model_json_schema()
        }
    )

    print("Response:")
    print(response.text)
    TXT_PATH = Path(__file__).parent / "img_describe.json"
    with open(TXT_PATH, 'w', encoding="utf-8") as file:
        file.write(str(response.text))

    # delete files just uploaded
    for img in my_imgs:
        client.files.delete(name=img.name)
    """
else:
    print("Set USE_MODEL to True to run this script.")
    sys.exit()

print("End of script.")

