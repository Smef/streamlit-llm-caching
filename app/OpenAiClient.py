
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()


def get_open_ai_client():
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    return client
