# always set to False unless about to run
USE_MODEL = False

# Imports
from pathlib import Path # for path stuff
import json
import polars as pl # for dealing with CSV
import base64
from pydantic import BaseModel, Field
import numpy as np # for ceil function
import utils_enrich # for RNG
import time # for time.sleep()
from google import genai
import sys # for sys.exit()

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

# input images?? directly pull from CSV
# TODO

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

n_imgs = old_df_rowlen
MAX_REQS_PER_MIN = 15
init_avg_waiting_time = np.ceil(60/MAX_REQS_PER_MIN)

# seed will be yyyymmddhhmmss at execution
seed_int = utils_enrich.get_datetime_seed()
print("Random seed:", seed_int)

(avg_waiting_time, waiting_times) = utils_enrich.get_floored_waiting_times(
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
# TODO

if USE_MODEL:
    print("Calling API...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    """
    my_imgs = []
    for img_path in my_imgs_paths:
        my_imgs.append(client.files.upload(file=img_path))

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
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

