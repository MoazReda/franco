"""
YouTube Comments Scraper for Franco Dataset
Collects Egyptian Arabic (Franco) comments from YouTube videos.
"""

import os
import csv
import time
import logging
from datetime import datetime
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


EGYPTIAN_CHANNELS = [
    "UCvRjpXsAZBHMWHOWcfQKqoQ",  # مش مهم يعني
    "UCp2_NRpz7CuJ0KDNGwdIVAQ",  # بودكاست مصري
]

FRANCO_KEYWORDS = [
    "3yzak", "3ayzak", "7elw", "kwayes", "tayeb",
    "mesh", "msh", "ana", "enta", "howa", "hya",
    "delwa2ty", "bukra", "ba3den", "3shan", "3alshan",
    "sa7", "sa7bi", "habibi", "ya3ni", "bas",
]


def build_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in .env file")
    return build("youtube", "v3", developerKey=api_key)


def is_franco(text: str) -> bool:
    """
    Detect if a comment contains Franco text.
    Returns True if the comment likely contains Franco.
    """
    text_lower = text.lower()

    # Check for digit-letter combinations typical in Franco
    franco_digits = ["3", "7", "2", "5", "8", "9", "6"]
    has_digit_combo = any(
        digit in text_lower and any(c.isalpha() for c in text_lower)
        for digit in franco_digits
    )

    # Check for known Franco keywords
    has_keyword = any(keyword in text_lower for keyword in FRANCO_KEYWORDS)

    return has_digit_combo or has_keyword


def get_video_comments(youtube, video_id: str, max_comments: int = 100) -> list:
    """
    Fetch comments from a single YouTube video.
    Returns a list of Franco comments.
    """
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

                if is_franco(text):
                    comments.append({
                        "franco": text.strip(),
                        "arabic": "",        # to be annotated
                        "english": "",       # to be annotated
                        "source": f"youtube_{video_id}",
                        "annotator": "",
                        "verified": "false",
                        "collected_at": datetime.now().isoformat()
                    })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.5)  # respect API rate limits

        except Exception as e:
            logger.error(f"Error fetching comments for video {video_id}: {e}")
            break

    logger.info(f"Video {video_id}: found {len(comments)} Franco comments")
    return comments


def search_franco_videos(youtube, query: str, max_videos: int = 10) -> list:
    """
    Search for Egyptian videos likely to have Franco comments.
    """
    try:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            relevanceLanguage="ar",
            regionCode="EG",
            maxResults=max_videos
        )
        response = request.execute()

        video_ids = [
            item["id"]["videoId"]
            for item in response.get("items", [])
        ]
        logger.info(f"Found {len(video_ids)} videos for query: '{query}'")
        return video_ids

    except Exception as e:
        logger.error(f"Error searching videos: {e}")
        return []


def save_to_csv(comments: list, output_path: str):
    """
    Save collected comments to CSV file.
    Appends to existing file if it exists.
    """
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

    logger.info(f"Saved {len(comments)} comments to {output_path}")


def run_collection(queries: list, max_videos_per_query: int = 5):
    """
    Main collection pipeline.
    """
    youtube = build_youtube_client()
    output_path = "data/raw/youtube_comments.csv"
    total_collected = 0

    for query in queries:
        logger.info(f"Processing query: '{query}'")
        video_ids = search_franco_videos(youtube, query, max_videos_per_query)

        for video_id in video_ids:
            comments = get_video_comments(youtube, video_id, max_comments=100)
            if comments:
                save_to_csv(comments, output_path)
                total_collected += len(comments)
            time.sleep(1)  # be respectful to the API

    logger.info(f"Collection complete. Total Franco comments: {total_collected}")
    return total_collected


if __name__ == "__main__":
    # Known videos with Franco comments
    KNOWN_FRANCO_VIDEOS = [
        "_BElKjt1NBE",
        "GJVj5WCYuFA",
    ]

    # Search queries using Franco keywords directly
    FRANCO_SEARCH_QUERIES = [
        "3yzak",
        "sa7bi",
        "7elw awy",
        "ya3ni",
        "mesh 3aref",
        "tab3an",
        "enta fein",
        "ana msh",
        "kwayes",
        "5alas",
    ]

    youtube = build_youtube_client()
    output_path = "data/raw/youtube_comments.csv"
    total = 0

    # Part 1: known videos directly
    logger.info("=== Part 1: Known Franco videos ===")
    for video_id in KNOWN_FRANCO_VIDEOS:
        comments = get_video_comments(youtube, video_id, max_comments=200)
        if comments:
            save_to_csv(comments, output_path)
            total += len(comments)
        time.sleep(1)

    # Part 2: search by Franco keywords
    logger.info("=== Part 2: Search by Franco keywords ===")
    collected_video_ids = set(KNOWN_FRANCO_VIDEOS)

    for query in FRANCO_SEARCH_QUERIES:
        video_ids = search_franco_videos(youtube, query, max_videos=3)
        for video_id in video_ids:
            if video_id in collected_video_ids:
                continue
            collected_video_ids.add(video_id)
            comments = get_video_comments(youtube, video_id, max_comments=100)
            if comments:
                save_to_csv(comments, output_path)
                total += len(comments)
            time.sleep(1)

    logger.info(f"Done. Total Franco comments collected: {total}")