import cv2
import sys
import numpy as np
from ultralytics import YOLO

def process_video(input_path, output_path, conf_thresh=0.25):
    # Load the trained YOLOv8 model
    model = YOLO("runs/detect/train5/weights/best.pt")
    
    # Define color mapping for specific classes (BGR format)
    # Adjust the keys if your model names are different
    color_map = {
        "blue-player": (255, 0, 0),   # Blue in BGR
        "red-player": (0, 0, 255),    # Red in BGR
        "referee": (255, 255, 255)   # White
    }
    
    # Default color if the class name is not in color_map
    default_color = (0, 255, 0)  # Green

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    
    # Initialize VideoWriter
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame with specified confidence threshold
        results = model(frame, conf=conf_thresh)
        # Get the detections from the first result
        boxes = results[0].boxes

        # Create a dictionary to hold one box per class (key: class id, value: box info)
        unique_boxes = {}
        if boxes is not None and len(boxes) > 0:
            # boxes.data is a tensor with each row:
            # [x1, y1, x2, y2, conf, cls]
            detections = boxes.data.cpu().numpy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                # Keep only the highest confidence detection per class
                if cls not in unique_boxes or conf > unique_boxes[cls][-1]:
                    unique_boxes[cls] = [x1, y1, x2, y2, conf, cls]

        # Draw the unique bounding boxes on the frame
        for box in unique_boxes.values():
            x1, y1, x2, y2, conf, cls = box
            # Get class name from model.names
            class_name = model.names[cls]
            # Choose color based on the class name
            color = color_map.get(class_name.lower(), default_color)
            label = f"{class_name} {conf:.2f}"
            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Put label text above the rectangle
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the annotated frame to the output video file
        writer.write(frame)
        
        # Optionally display the frame
        cv2.imshow("Annotated Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python video_detect_colored.py input_video.mp4 output_video.mp4 [confidence_threshold]")
    else:
        input_video = sys.argv[1]
        output_video = sys.argv[2]
        conf_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
        process_video(input_video, output_video, conf_thresh)
