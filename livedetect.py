import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
import sys


def load_model(path):
    try:
        model = YOLO(path)
        print(f"✅ Model loaded: {path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model {path}: {e}")
        sys.exit(1)


def main():
    # Load models
    model1 = load_model("best.pt")
    #model2 = load_model("best2.pt")  # If you plan to use this later

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to open the camera feed!")
        sys.exit(1)

    bounding_box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    prev_time = 0  # For FPS calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        # Run YOLO inference
        results = model1(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        # Add FPS on frame
        cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show annotated image
        cv2.imshow("YOLO Webcam Detection", annotated_image)

        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Program exited successfully")


if __name__ == "__main__":
    main()
