import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API-KEY")
API_KEY_NAME = os.getenv("API-KEY-NAME")