from googleapiclient.discovery import build
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs

# Replace this with your actual API key
API_KEY = "AIzaSyBOT6NUcY1EvKe55rj9XmtrxkRl3MS5sHo"
YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

def extract_video_id(url):
    """Extract video ID from standard or short YouTube URLs"""
    try:
        parsed = urlparse(url)
        if parsed.hostname == 'youtu.be':
            return parsed.path[1:]
        if parsed.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed.path == '/watch':
                return parse_qs(parsed.query)['v'][0]
        return None
    except:
        return None

def scrape_comments(youtube_url, max_comments=100):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    comments = []
    try:
        response = YOUTUBE.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()

        while response and len(comments) < max_comments:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append({"comment": comment})

                if len(comments) >= max_comments:
                    break

            if "nextPageToken" in response:
                response = YOUTUBE.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response["nextPageToken"],
                    maxResults=100,
                    textFormat="plainText"
                ).execute()
            else:
                break

    except Exception as e:
        print("Error while fetching comments:", e)
        return pd.DataFrame()

    df = pd.DataFrame(comments)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/comments.csv", index=False)
    return df
