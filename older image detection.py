import cv2 
import requests
import numpy as np 
# import imutils
import supervision as sv
from ultralytics import YOLO

model = YOLO(f'best.pt')


print("Module Imported successfully")



bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


frame = cv2.imread("01.jpg")

results = model(frame)[0]
detections = sv.Detections.from_ultralytics(results)

annotated_image = bounding_box_annotator.annotate(scene = frame, detections= detections)
annotated_image = label_annotator.annotate(scene= annotated_image, detections=detections)
       

cv2.imshow("Webcam", frame)
cv2.waitKey(0)

print("Program Worked with success")

cv2.destroyAllWindows()


