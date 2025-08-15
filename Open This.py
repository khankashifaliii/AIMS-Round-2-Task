from pipeline import ActivityDetectionPipeline
import cv2

def resize_max_800(image):
    h, w = image.shape[:2]
    scale = min(800 / h, 800 / w)  # Keep aspect ratio
    if scale < 1:  # Only resize if bigger than 800px
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image

# 1. Initialize the pipeline with your trained YOLOv8 model
pipeline = ActivityDetectionPipeline('best.pt')

# 2. Load an image
image = cv2.imread('02.jpg')

# 3. Process with text query
query = "lady standing drinking water calling phone on floor"
detections = pipeline.process(query, image)

# 4. Print results
for i, det in enumerate(detections[:5]):
    print(f"{i+1}. {det.class_name}: confidence={det.confidence:.3f}, "
          f"text_relevance={det.text_relevance:.3f}")

# 5. Visualize results
vis_image = pipeline.visualize_results(image, detections[:2])
vis_image = resize_max_800(vis_image)
cv2.imshow('Results', vis_image)
cv2.waitKey(0)