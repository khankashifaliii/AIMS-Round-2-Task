from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Ask for classes
classes_input = input("Enter class names to detect (separated by space, leave blank for all): ").strip()
selected_classes = classes_input.split() if classes_input else None


# Run detection
results = model("01.jpg")

# Filter detections if classes given
if selected_classes:
    for r in results:
        names = r.names
        keep = []
        for box in r.boxes:
            cls_name = names[int(box.cls)]
            if cls_name in selected_classes:
                keep.append(box)
        r.boxes = keep

# Display image
for r in results:
    img = r.plot()  # Draw detections on image
    cv2.imshow("Detection", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

