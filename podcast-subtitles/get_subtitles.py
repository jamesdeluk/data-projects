import csv
import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get('API_KEY')

max_results = 50
count_of_new = 0

def load_channels(filename="channels.csv"):
    channels = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            username = row[0].strip()
            channel_id = row[1].strip()
            category = row[2].strip()
            channels.append((username, channel_id, category))
    return channels

def save_channel(username, channel_id, filename="channels.csv"):
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, channel_id])

def get_channel_id(username):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=channel&q={username}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("items", [])
        if results:
            return results[0]["id"]["channelId"]
        else:
            print(f"No channel found for username: {username}")
            return None
    else:
        print(f"Error fetching channel ID for username {username}: {response.json()}")
        return None

def get_channel_videos(channel_id, max_results=max_results):
    url = f"https://www.googleapis.com/youtube/v3/search?key={API_KEY}&channelId={channel_id}&part=snippet&type=video&order=date&maxResults={max_results}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching videos for channel {channel_id}: {response.json()}")
        return []
    return response.json().get("items", [])

def is_video_downloaded(video_id, history_file="history.txt"):
    try:
        with open(history_file, "r") as f:
            downloaded_videos = f.read().splitlines()
        return video_id in downloaded_videos
    except FileNotFoundError:
        return False

def mark_video_as_downloaded(video_id, history_file="history.txt"):
    with open(history_file, "a") as f:
        f.write(video_id + "\n")

def download_subtitles(video_id, username, video_title, date, category):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        save_transcript(video_id, username, video_title, transcript, date, category)
        mark_video_as_downloaded(video_id)
    except Exception as e:
        print(f"No subtitles available for video {video_id}: {e}")

def save_transcript(video_id, username, video_title, transcript, date, category):
    word_count = sum(len(entry['text'].split()) for entry in transcript)
    username = "".join(c for c in username if c.isalnum() or c in " -_")
    video_title = "".join(c for c in video_title if c.isalnum() or c in " -_")
    filename = f"{date} - {category} - {username} - {video_title} - {word_count} words - [{video_id}].txt"
    if category == "Summary":
        folder = "subtitles-summary"
    elif word_count < 500:
        folder = "subtitles-short"
    elif word_count >= 500 and word_count < 2000:
        folder = "subtitles-medium"
    else:
        folder = "subtitles"

    with open(f"{folder}/{filename}", "w", encoding="utf-8") as f:
        for entry in transcript:
            f.write(f"{entry['text']}\n")
    print(f"Subtitles saved as: {filename}")

def monitor_channels():
    channels = load_channels()
    for username, channel_id, category in channels:
        if channel_id == '':
            print(f"Resolving username {username} to channel ID...")
            resolved_id = get_channel_id(username[1:])
            if resolved_id:
                save_channel(username, resolved_id)
                channel_id = resolved_id
            else:
                print(f"Could not resolve username: {username}")
                continue

        print(f"Checking channel: {channel_id}")
        videos = get_channel_videos(channel_id)
        for video in videos:
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            date = video["snippet"]["publishedAt"].split("T")[0]
            if not is_video_downloaded(video_id):
                print(f"Fetching subtitles for video: {video_title} ({video_id})")
                download_subtitles(video_id, username, video_title, date, category)
                global count_of_new
                count_of_new += 1
            else:
                print(f"Subtitles for video {video_id} already downloaded.")

    print(f"{count_of_new} new transcripts downloaded")
    print("End")

if __name__ == "__main__":
    monitor_channels()