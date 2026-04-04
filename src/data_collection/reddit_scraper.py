"""
Reddit Scraper for Franco Dataset
Collects Egyptian Franco from r/egypt and r/arabs using public access.
"""

import os
import csv
import time
import logging
from datetime import datetime
import requests

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
    "bs", "mn", "3la", "m3ak", "3ndi",
]

SUBREDDITS = [
    "egypt",         
    "arabs",           
    "Arabic",         
    "learnegyptian",  
    "egyptpics",       
]

HEADERS = {
    "User-Agent": "franco-dataset-collector/1.0 (academic research)"
}


def is_franco(text: str) -> bool:
    text_lower = text.lower()
    franco_digits = ["3", "7", "2", "5", "8", "9", "6"]
    has_digit_combo = any(
        digit in text_lower and any(c.isalpha() for c in text_lower)
        for digit in franco_digits
    )
    has_keyword = any(keyword in text_lower for keyword in FRANCO_KEYWORDS)
    return has_digit_combo or has_keyword


def get_subreddit_posts(subreddit: str, limit: int = 100, category: str = "hot") -> list:
    """Fetch posts from a subreddit using Reddit's public JSON API."""
    url = f"https://www.reddit.com/r/{subreddit}/{category}.json?limit={limit}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            posts = data["data"]["children"]
            logger.info(f"r/{subreddit}: fetched {len(posts)} posts")
            return posts
        else:
            logger.error(f"r/{subreddit}: status {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"r/{subreddit}: {e}")
        return []


def get_post_comments(subreddit: str, post_id: str) -> list:
    """Fetch comments from a single post."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=200"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            comments = []
            if len(data) > 1:
                extract_comments(data[1]["data"]["children"], comments)
            return comments
        else:
            return []
    except Exception as e:
        logger.error(f"Error fetching comments for {post_id}: {e}")
        return []


def extract_comments(children: list, result: list, depth: int = 0):
    """Recursively extract comments from Reddit JSON."""
    if depth > 3:
        return
    for child in children:
        if child.get("kind") == "t1":
            body = child["data"].get("body", "")
            if body and body != "[deleted]" and body != "[removed]":
                result.append(body)
            replies = child["data"].get("replies", "")
            if isinstance(replies, dict):
                extract_comments(
                    replies["data"]["children"],
                    result,
                    depth + 1
                )


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
    logger.info(f"Saved {len(comments)} comments to {output_path}")


def run_collection():
    output_path = "data/raw/reddit_comments.csv"
    total = 0

    for subreddit in SUBREDDITS:
        logger.info(f"=== Processing r/{subreddit} ===")
        franco_comments = []

        for category in ["hot", "new", "top"]:
            posts = get_subreddit_posts(subreddit, limit=100, category=category)

            for post in posts:
                post_data = post["data"]
                post_id = post_data.get("id")
                post_title = post_data.get("title", "")
                post_selftext = post_data.get("selftext", "")

                # Check post title and body
                for text in [post_title, post_selftext]:
                    if text and is_franco(text):
                        franco_comments.append({
                            "franco": text.strip(),
                            "arabic": "",
                            "english": "",
                            "source": f"reddit_r/{subreddit}_{post_id}",
                            "annotator": "",
                            "verified": "false",
                            "collected_at": datetime.now().isoformat()
                        })

                # Fetch and check comments
                comments = get_post_comments(subreddit, post_id)
                for comment in comments:
                    if is_franco(comment):
                        franco_comments.append({
                            "franco": comment.strip(),
                            "arabic": "",
                            "english": "",
                            "source": f"reddit_r/{subreddit}_{post_id}",
                            "annotator": "",
                            "verified": "false",
                            "collected_at": datetime.now().isoformat()
                        })

                time.sleep(0.5)

            time.sleep(2)

        if franco_comments:
            save_to_csv(franco_comments, output_path)
            total += len(franco_comments)
            logger.info(f"r/{subreddit}: {len(franco_comments)} Franco items collected")

    logger.info(f"Reddit collection complete. Total: {total}")
    return total


if __name__ == "__main__":
    run_collection()