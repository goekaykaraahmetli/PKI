import cv2
import os
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = '../pipeline/runs/detect/train5/weights/best.pt'
BASE_DIR = 'dataset'
OUTPUT_CSV = 'detections.csv'

def load_annotation_frames(subdir):
    """
    Reads dataset/<subdir>/annotations.json, which is structured as a list
    with a single top-level object that contains 'tracks'.
    Collects all 'frame' indices from 'tracks'->'shapes' where 'outside'==False.
    Returns a sorted list of unique frame indices.
    """
    import json
    ann_path = os.path.join(BASE_DIR, subdir, 'annotations.json')
    if not os.path.isfile(ann_path):
        print(f"[WARN] Missing annotations.json in {subdir}")
        return []

    with open(ann_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list) and len(data) > 0:
        ann_data = data[0]  # get the first object in the list
    else:
        ann_data = {}

    frames_set = set()
    tracks = ann_data.get('tracks', [])
    for track in tracks:
        shapes = track.get('shapes', [])
        for shape in shapes:
            if shape.get('outside') == False:
                frame_idx = shape.get('frame')
                if frame_idx is not None:
                    frames_set.add(frame_idx)
    return sorted(frames_set)

def find_video_file(subdir):
    data_dir = os.path.join(BASE_DIR, subdir, 'data')
    if not os.path.isdir(data_dir):
        return None
    for f in os.listdir(data_dir):
        if f.endswith('.mp4'):
            return os.path.join(data_dir, f)
    return None

def main():
    # Check for GPU
    if not torch.cuda.is_available():
        print("[ERROR] No GPU detected. Exiting.")
        return
    device = 'cuda:0'

    # Initialize YOLO
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"[INFO] YOLO loaded on {device}: {torch.cuda.get_device_name(0)}")

    all_rows = []

    # Iterate subdirectories in dataset/
    for subdir in os.listdir(BASE_DIR):
        sub_path = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(sub_path):
            continue

        # Get annotated frames
        frames = load_annotation_frames(subdir)
        if not frames:
            continue

        # Find the video
        video_path = find_video_file(subdir)
        if not video_path:
            print(f"[WARN] No .mp4 found for {subdir}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] {subdir}: {len(frames)} annotated frames (video total: {total_frames} frames)")

        # For each annotated frame, run YOLO
        for fidx in tqdm(frames, desc=f"Detecting in {subdir}", unit="frame"):
            if fidx >= total_frames:
                print(f"[WARN] Frame {fidx} >= video length {total_frames} in {subdir}. Skipping.")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] Could not read frame {fidx} from {subdir}")
                continue

            # YOLO detection
            results = model(frame, device=device)
            # Filter out "referee" boxes and keep only fighters
            fighters = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name.lower() == "referee":
                    continue  # skip referee
                x1, y1, x2, y2 = box.xyxy[0]
                fighters.append((x1, y1, x2, y2))

            # Only store if exactly 2 fighter bounding boxes
            if len(fighters) == 2:
                # Convert to int
                (f1x1, f1y1, f1x2, f1y2) = [int(v) for v in fighters[0]]
                (f2x1, f2y1, f2x2, f2y2) = [int(v) for v in fighters[1]]

                # Create a single row representing this “state”
                row = {
                    'video': subdir,
                    'frame': fidx,
                    'f1x1': f1x1, 'f1y1': f1y1, 'f1x2': f1x2, 'f1y2': f1y2,
                    'f2x1': f2x1, 'f2y1': f2y1, 'f2x2': f2x2, 'f2y2': f2y2
                }
                all_rows.append(row)
            else:
                # You might choose to skip or handle the case differently
                pass

        cap.release()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Saved {len(df)} 'two-fighter' states to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
