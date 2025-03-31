import cv2
from ultralytics import YOLO

# Load trained model from the best weights file.
model = YOLO("runs/detect/train5/weights/best.pt")

# Open a connection to the webcam (device index 0).
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam")
        break

    # Run YOLOv8 inference on the frame with a confidence threshold (adjust conf if needed)
    results = model(frame, conf=0.25)

    # Get an annotated frame (draws bounding boxes and labels)
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
