# getting the API key
# see secrets.py.example if you haven't set up secrets.py

from pathlib import Path
import importlib.util

TOPDIR_PATH = Path(__file__).parent.parent
SECRETS_PATH = TOPDIR_PATH / "secrets.py"

secrets_spec = importlib.util.spec_from_file_location("secrets", SECRETS_PATH)
secrets_module = importlib.util.module_from_spec(secrets_spec)
secrets_spec.loader.exec_module(secrets_module)

GEMINI_API_KEY = secrets_module.GEMINI_API_KEY

# now on to the AI stuff
"""
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)
"""

