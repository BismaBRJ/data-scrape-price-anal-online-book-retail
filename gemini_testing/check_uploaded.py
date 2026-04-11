# getting the API key
# see secrets.json.example if you haven't set up secrets.json

from pathlib import Path
import json

TOPDIR_PATH = Path(__file__).parent.parent
SECRETS_PATH = TOPDIR_PATH / "secrets.json"

print("Reading API key...")
with open(SECRETS_PATH, 'r', encoding="utf-8") as file:
    SECRETS_JSON = json.load(file)

GEMINI_API_KEY = SECRETS_JSON["GEMINI_API_KEY"]
print("API key obtained.")

# now on to the AI stuff
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

print("Uploaded files:")
for f in client.files.list():
    print("File name:")
    print(f.name)
    print("File:")
    print(f)

print("End of script.")

