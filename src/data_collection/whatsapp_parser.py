"""
WhatsApp Chat Parser for Franco Dataset
Parses exported WhatsApp .txt files and extracts Franco messages.
"""

import os
import re
import csv
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

FRANCO_KEYWORDS = [
    "3yzak", "3ayzak", "7elw", "kwayes", "tayeb",
    "mesh", "msh", "ana", "enta", "howa",
    "delwa2ty", "bukra", "ba3den", "3shan",
    "sa7", "sa7bi", "ya3ni", "bas", "5alas",
    "7aga", "3amel", "mabsout", "tab3an",
    "bs", "3la", "m3ak", "3ndi", "msh",
    "wala", "leh", "ezay", "fein", "emta",
    "keda", "kida", "dah", "deh", "hwa",
    "4ever", "2day", "b2olak", "b2olk",
]

# Regex patterns for different WhatsApp export formats
PATTERNS = [
    # Format: 06/11/2022, 4:37 pm - Name: message
    re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?\s?-\s?([^:]+):\s(.+)$'),
    # iOS format with brackets
    re.compile(r'^\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s?(?:am|pm|AM|PM)?\]\s([^:]+):\s(.+)$'),
]

SYSTEM_MESSAGES = [
    "messages and calls are end-to-end encrypted",
    "changed the subject",
    "added you",
    "left",
    "joined using this group",
    "changed this group",
    "you were added",
    "created group",
    "changed the group",
    "message was deleted",
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "document omitted",
    "contact card omitted",
    "missed voice call",
    "missed video call",
    "null",
]


def is_system_message(text: str) -> bool:
    text_lower = text.lower()
    return any(msg in text_lower for msg in SYSTEM_MESSAGES)


def is_franco(text: str) -> bool:
    text_lower = text.lower()
    franco_digits = ["3", "7", "2", "5", "8", "9", "6"]
    has_digit_combo = any(
        digit in text_lower and any(c.isalpha() for c in text_lower)
        for digit in franco_digits
    )
    has_keyword = any(keyword in text_lower for keyword in FRANCO_KEYWORDS)
    return has_digit_combo or has_keyword


def parse_whatsapp_file(filepath: str) -> list:
    """Parse a single WhatsApp .txt export file."""
    messages = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Cannot read {filepath}: {e}")
            return []

    current_message = None
    source_name = Path(filepath).stem.replace("WhatsApp Chat with ", "").strip()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parsed = None
        for pattern in PATTERNS:
            match = pattern.match(line)
            if match:
                parsed = match
                break

        if parsed:
            # Save previous message
            if current_message and not is_system_message(current_message["franco"]):
                if is_franco(current_message["franco"]):
                    messages.append(current_message)

            sender = parsed.group(1).strip()
            text = parsed.group(2).strip()

            current_message = {
                "franco": text,
                "arabic": "",
                "english": "",
                "source": f"whatsapp_{source_name}",
                "annotator": "",
                "verified": "false",
                "collected_at": datetime.now().isoformat()
            }
        else:
            # Continuation of previous message (multi-line)
            if current_message and line:
                current_message["franco"] += " " + line

    # Don't forget last message
    if current_message and not is_system_message(current_message["franco"]):
        if is_franco(current_message["franco"]):
            messages.append(current_message)

    return messages


def save_to_csv(comments: list, output_path: str):
    file_exists = os.path.exists(output_path)
    fieldnames = [
        "franco", "arabic", "english",
        "source", "annotator", "verified", "collected_at"
    ]
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(comments)
    logger.info(f"Saved {len(comments)} messages to {output_path}")


def run_collection():
    whatsapp_dir = "data/raw/whatsapp"
    output_path = "data/raw/whatsapp_messages.csv"
    total = 0

    txt_files = list(Path(whatsapp_dir).glob("*.txt"))
    logger.info(f"Found {len(txt_files)} WhatsApp chat files")

    for txt_file in txt_files:
        logger.info(f"Processing: {txt_file.name}")
        messages = parse_whatsapp_file(str(txt_file))

        if messages:
            save_to_csv(messages, output_path)
            total += len(messages)
            logger.info(f"{txt_file.name}: {len(messages)} Franco messages found")
        else:
            logger.info(f"{txt_file.name}: no Franco messages found")

    logger.info(f"WhatsApp collection complete. Total: {total}")
    return total


if __name__ == "__main__":
    run_collection()