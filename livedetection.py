import cv2 
import requests
import numpy as np 
# import imutils
import supervision as sv
from ultralytics import YOLO

model = YOLO(f'best.pt')


print("Module Imported successfully")

webcam= cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Unable to open the camera feed!!")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    Success, frame=webcam.read()
    if not Success:
        break 
    
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(scene = frame, detections= detections)
    annotated_image = label_annotator.annotate(scene= annotated_image, detections=detections)
       

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

print(Success)
print("Program Worked with success")

webcam.release()
cv2.destroyAllWindows()


