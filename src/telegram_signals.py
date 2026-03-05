import os
import re
from dotenv import load_dotenv
from telethon import TelegramClient

# Load environment variables
load_dotenv()

TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

CHANNEL_1 = os.getenv("TELEGRAM_CHANNEL_1")
CHANNEL_2 = os.getenv("TELEGRAM_CHANNEL_2")

MESSAGE_LIMIT = int(os.getenv("TELEGRAM_MESSAGE_LIMIT", 50))


client = TelegramClient("cryptosent_session", TELEGRAM_API_ID, TELEGRAM_API_HASH)


def clean_message(text):
    """
    Clean Telegram messages for NLP processing
    """

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    text = text.strip()

    return text


def is_crypto_signal(text):
    """
    Filter messages likely related to crypto signals
    """

    keywords = [
        "btc",
        "eth",
        "long",
        "short",
        "buy",
        "sell",
        "target",
        "entry",
        "tp",
        "sl",
        "pump",
        "signal"
    ]

    for word in keywords:
        if word in text:
            return True

    return False


async def fetch_signals(channel, limit=MESSAGE_LIMIT):
    """
    Fetch messages from a Telegram channel
    """

    messages = []

    async with client:

        async for msg in client.iter_messages(channel, limit=limit):

            if msg.text:

                cleaned = clean_message(msg.text)

                if is_crypto_signal(cleaned):

                    messages.append(cleaned)

    return messages


async def collect_all_signals():
    """
    Collect messages from all configured Telegram channels
    """

    all_messages = []

    if CHANNEL_1:

        msgs = await fetch_signals(CHANNEL_1)

        all_messages.extend(msgs)

    if CHANNEL_2:

        msgs = await fetch_signals(CHANNEL_2)

        all_messages.extend(msgs)

    return all_messages
