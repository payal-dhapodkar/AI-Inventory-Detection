
import os
import cv2

def extract_frames_from_videos(video_folder, output_folder, frame_rate=5):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4'))]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_rate == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame_{frame_id:04d}.jpg"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_id += 1
            count += 1

        cap.release()
        print(f"✅ Extracted from: {video_file}")
    print("✅ Done: All frames extracted.")

# 🔁 Use FULL ABSOLUTE PATH here (Change this to your real path)
extract_frames_from_videos(
    r"D:\Inventory Management\dataset\videos",
    r"D:\Inventory Management\dataset\images"
)