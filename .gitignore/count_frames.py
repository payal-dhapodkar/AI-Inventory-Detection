import os

# Paths
videos_folder = 'dataset/videos'
frames_folder = 'dataset/images'

# Get all video file names (without extension)
video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4'))]
video_names = [os.path.splitext(v)[0] for v in video_files]

print("📊 Frame count per video:\n")

# Loop through each video and count its frames in the images folder
for video in video_names:
    # Count images that start with the video name and end with .jpg or .png
    frame_count = len([
        f for f in os.listdir(frames_folder)
        if f.startswith(video) and f.endswith(('.jpg'))
    ])
    
    print(f"🎞️ {video}: {frame_count} frames")
