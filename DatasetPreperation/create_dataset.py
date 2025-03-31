import cv2
import os
import json
import pandas as pd
from tqdm import tqdm

DETECTIONS_CSV = 'detections.csv'
BASE_DIR = 'dataset'
OUTPUT_DIR = 'dataset_images'

def load_annotations(subdir):
    # Load annotations.json and extract frame label
    ann_path = os.path.join(BASE_DIR, subdir, 'annotations.json')
    if not os.path.exists(ann_path):
        print(f"Warning: Missing annotations.json for {subdir}")
        return []
    with open(ann_path, 'r') as f:
        return json.load(f)

def find_video_file(subdir):
    # Locate the video file for a given subdir
    data_dir = os.path.join(BASE_DIR, subdir, 'data')
    if not os.path.isdir(data_dir):
        print(f"Warning: Missing 'data' folder for {subdir}")
        return None
    for f in os.listdir(data_dir):
        if f.endswith('.mp4'):
            return os.path.join(data_dir, f)
    return None

def get_action_for_frame(annotations, frame_idx):
    #Find the action label for the given frame
    if not annotations:
        return None

    for ann in annotations:
        tracks = ann.get('tracks', [])
        for track in tracks:
            label = track.get('label', None)
            shapes = track.get('shapes', [])

            for i, shape in enumerate(shapes):
                if not shape.get('outside', False):
                    start_frame = shape.get('frame', -1)

                    # Find when the action stops
                    end_frame = None
                    for j in range(i+1, len(shapes)):
                        if shapes[j].get('outside', False):
                            end_frame = shapes[j]['frame']
                            break

                    if end_frame is None and frame_idx >= start_frame:
                        return label
                    elif start_frame <= frame_idx < end_frame:
                        return label

    return None  # No action found for this frame

def main():
    #Process detections and save cropped fighter images
    if not os.path.exists(DETECTIONS_CSV):
        print(f"Error: {DETECTIONS_CSV} does not exist.")
        return

    df = pd.read_csv(DETECTIONS_CSV)
    if df.empty:
        print("Error: detections.csv is empty! Check extract_and_detect.py")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        subdir = row['video']
        frame_idx = int(row['frame'])

        # Read bounding boxes for both fighters
        f1x1, f1y1, f1x2, f1y2 = int(row['f1x1']), int(row['f1y1']), int(row['f1x2']), int(row['f1y2'])
        f2x1, f2y1, f2x2, f2y2 = int(row['f2x1']), int(row['f2y1']), int(row['f2x2']), int(row['f2y2'])

        # Load annotations and get action label
        annotations = load_annotations(subdir)
        action = get_action_for_frame(annotations, frame_idx)
        if not action:
            print(f"Skipping frame {frame_idx} in {subdir}: No matching action.")
            continue

        # Find video file
        video_path = find_video_file(subdir)
        if not video_path:
            print(f"Warning: Video file missing for {subdir}")
            continue

        # Load video frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Failed to read frame {frame_idx} from {video_path}")
            continue

        # Crop both fighters
        fighter1_crop = frame[f1y1:f1y2, f1x1:f1x2]
        fighter2_crop = frame[f2y1:f2y2, f2x1:f2x2]

        if fighter1_crop.size == 0 or fighter2_crop.size == 0:
            print(f"Error: Empty crop for {subdir} frame {frame_idx}")
            continue

        # Save images
        action_dir = os.path.join(OUTPUT_DIR, action)
        os.makedirs(action_dir, exist_ok=True)

        out_path_f1 = os.path.join(action_dir, f"{subdir}_frame{frame_idx}_f1.jpg")
        out_path_f2 = os.path.join(action_dir, f"{subdir}_frame{frame_idx}_f2.jpg")

        cv2.imwrite(out_path_f1, fighter1_crop)
        cv2.imwrite(out_path_f2, fighter2_crop)
        print(f"Saved {out_path_f1} and {out_path_f2}")

if __name__ == '__main__':
    main()
