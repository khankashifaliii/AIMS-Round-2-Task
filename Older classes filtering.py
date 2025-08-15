from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # Change to your model path

# Mapping from YOLO class name → list of possible keywords
class_map = {
    "calling": ["call", "calling", "phone", "talking on phone"],
    "construction": ["construction", "building", "repair", "site", "worker"],
    "Dance": ["dance", "dancing", "performance"],
    "drinking": ["drink", "drinking", "water", "juice", "soda"],
    "exercise": ["exercise", "workout", "gym", "yoga"],
    "fighting": ["fight", "fighting", "punch", "attack", "hit"],
    "football": ["football", "soccer", "playing", "kick"],
    "holding-book": ["book", "reading", "holding book", "study"],
    "person": ["person", "people", "human"],
    "plant": ["plant", "tree", "flower", "vegetation"],
    "restaurants": ["restaurant", "dining", "eating", "cafe", "food court"],
    "sitting": ["sit", "sitting", "seated"],
    "Snatching": ["snatch", "snatching", "steal", "theft", "robbery"],
    "standing": ["stand", "standing", "upright"],
    "texting": ["text", "texting", "message", "chatting", "typing on phone"],
    "using-computer": ["computer", "laptop", "pc", "typing", "using computer"],
    "vehicle": ["vehicle", "car", "bike", "motorcycle", "bus", "truck"],
    "walking": ["walk", "walking", "strolling"]
}

# Function to find relevant classes from sentence
def get_relevant_classes(sentence):
    sentence_lower = sentence.lower()
    matched_classes = []
    for cls, keywords in class_map.items():
        if any(k in sentence_lower for k in keywords):
            matched_classes.append(cls)
    return matched_classes

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    # Example inputs
    sentence = input("Enter your description: ")

    image_path = input("input image path: ")  # Change to your image path

    # Get relevant classes from sentence
    relevant_classes = get_relevant_classes(sentence)

    if not relevant_classes:
        print("No matching classes found for the given sentence.")
    else:
        # print(f"Matched Classes: {relevant_classes}")

        # Map matched classes to YOLO class IDs
        class_ids = [list(model.names.values()).index(c) for c in relevant_classes if c in model.names.values()]

        # Run YOLO prediction only for matched classes
        results = model.predict(source=image_path, classes=class_ids, conf=0.45)

        # Display results
        annotated_img = results[0].plot()
        
        cv2.imshow("Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
