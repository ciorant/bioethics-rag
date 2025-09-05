from chatbot import BioethicsChatbot
import os
import re

def clean_api_key(api_key):
    """Clean the API key from any extra spaces, quotes, or invisible characters"""
    if not api_key:
        raise ValueError("No API key found")

    # Remove surrounding quotes and whitespace
    cleaned = api_key.strip().strip('"').strip("'")

    # Remove any non-printable characters
    cleaned = re.sub(r'[^\x20-\x7E]', '', cleaned)

    # Remove any extra spaces
    cleaned = cleaned.strip()

    return cleaned

raw_key = os.environ.get("OPENAI_API_KEY")
clean_key = clean_api_key(raw_key)
os.environ["OPENAI_API_KEY"] = clean_key

bot = BioethicsChatbot()
print("Bioethics Chatbot (type 'quit' to exit)\n")

while True:
    q = input("User: ")
    if q.lower() in {"quit", "exit"}:
        break
    answer = bot.ask(q)