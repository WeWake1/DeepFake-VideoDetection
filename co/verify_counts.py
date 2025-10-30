"""
Verify actual counts of celebrities and real videos
"""

import csv
import json
from collections import defaultdict

# Load frame mapping
frame_mapping_csv = r"j:\DF\frame_mapping.csv"
celebrity_mapping_json = r"j:\DF\celebrity_mapping.json"

# Count real videos
real_videos = []
fake_videos = []

with open(frame_mapping_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['type'] == 'real':
            real_videos.append(row['video_name'])
        elif row['type'] == 'fake':
            fake_videos.append(row['video_name'])

print("=" * 80)
print("VERIFICATION OF COUNTS")
print("=" * 80)
print(f"\nReal videos: {len(real_videos)}")
print(f"Fake videos: {len(fake_videos)}")
print(f"Total videos: {len(real_videos) + len(fake_videos)}")

# Count celebrities from JSON
with open(celebrity_mapping_json, 'r', encoding='utf-8') as f:
    celeb_data = json.load(f)

celebrity_ids = sorted(celeb_data.keys(), key=lambda x: int(x.replace('id', '')))

print(f"\nCelebrities in celebrity_mapping.json: {len(celebrity_ids)}")
print(f"Celebrity IDs: {', '.join(celebrity_ids)}")

# Find all unique celebrity IDs from real videos
celeb_from_real = set()
for video in real_videos:
    # Extract celebrity ID from video name (id{X}_{SEQUENCE})
    parts = video.split('_')
    if parts[0].startswith('id'):
        celeb_from_real.add(parts[0])

celeb_from_real_sorted = sorted(celeb_from_real, key=lambda x: int(x.replace('id', '')))

print(f"\nUnique celebrities from real videos: {len(celeb_from_real_sorted)}")
print(f"Celebrity IDs from real videos: {', '.join(celeb_from_real_sorted)}")

# Check for mismatches
json_celebs = set(celebrity_ids)
real_celebs = set(celeb_from_real_sorted)

missing_in_json = real_celebs - json_celebs
extra_in_json = json_celebs - real_celebs

if missing_in_json:
    print(f"\n⚠️ Celebrities in real videos but NOT in JSON: {missing_in_json}")
if extra_in_json:
    print(f"\n⚠️ Celebrities in JSON but NOT in real videos: {extra_in_json}")

# Count videos per celebrity
celeb_counts = defaultdict(int)
for video in real_videos:
    parts = video.split('_')
    if parts[0].startswith('id'):
        celeb_counts[parts[0]] += 1

print(f"\n" + "=" * 80)
print("VIDEOS PER CELEBRITY (from real videos)")
print("=" * 80)
for celeb_id in celeb_from_real_sorted:
    count = celeb_counts[celeb_id]
    print(f"{celeb_id}: {count} videos")

print(f"\n" + "=" * 80)
print(f"SUMMARY")
print("=" * 80)
print(f"✓ Real videos: {len(real_videos)} (expected: 590)")
print(f"✓ Fake videos: {len(fake_videos)}")
print(f"✓ Unique celebrities: {len(celeb_from_real_sorted)}")
print(f"✓ Average videos per celebrity: {len(real_videos) / len(celeb_from_real_sorted):.2f}")
