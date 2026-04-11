# always set to False unless about to run
USE_MODEL = False

# for path stuff
from pathlib import Path

# getting the API key
# see secrets.json.example if you haven't set up secrets.json
import json

TOPDIR_PATH = Path(__file__).parent.parent
SECRETS_PATH = TOPDIR_PATH / "secrets.json"

print("Reading API key...")
with open(SECRETS_PATH, 'r', encoding="utf-8") as file:
    SECRETS_JSON = json.load(file)

GEMINI_API_KEY = SECRETS_JSON["GEMINI_API_KEY"]
print("API key obtained.")

# input images
IMGS_FOLDER = Path(__file__).parent / "images"
MY_IMGS_PATHS = [
    # f for f in IMGS_FOLDER.iterdir() if f.is_file()
    IMGS_FOLDER / "img1.jpeg",
    IMGS_FOLDER / "img2.jpeg"
]

# prompt engineering and JSON output structuring
from pydantic import BaseModel, Field

MY_PROMPT = """
You are an expert at least as good as Google Lens.
Also, you are an experienced, master librarian.
Reply with metadata and any extra information about the book you see.
Extra information includes anything useful from search results. 
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

# now connecting to Gemini
from google import genai

if USE_MODEL:
    print("Calling API...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    my_imgs = []
    for img_path in MY_IMGS_PATHS:
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
else:
    print("Set USE_MODEL to True to run this script.")

