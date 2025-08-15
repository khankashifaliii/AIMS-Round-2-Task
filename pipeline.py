import torch
import cv2
import numpy as np
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class Detection:
    """Structure to hold detection results"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    text_relevance: float = 0.0

class ActivityDetectionPipeline:
    def __init__(self, yolo_model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the hybrid activity detection pipeline
        
        Args:
            yolo_model_path: Path to your trained YOLOv8 model
            device: Device to run inference on
        """
        # Initialize models
        self.device = device
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize NLP models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spacy en_core_web_sm not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define your 16 classes
        self.classes = [
            'calling', 'construction', 'dance', 'drinking', 'exercise',
            'fighting', 'football', 'holding-book', 'person', 'plant',
            'restaurants', 'sitting', 'snatching', 'standing', 'texting',
            'using-computer', 'vehicle', 'walking'
        ]
        
        # Create expanded vocabulary for better matching
        self.class_synonyms = {
            'calling': ['phone call', 'telephone', 'making call', 'on phone', 'talking phone'],
            'construction': ['building', 'construction work', 'worker', 'hard hat', 'construction site'],
            'dance': ['dancing', 'dancer', 'dancing people', 'dance performance', 'choreography'],
            'drinking': ['drink', 'beverage', 'drinking water', 'drinking coffee', 'sipping'],
            'exercise': ['workout', 'fitness', 'gym', 'training', 'physical activity'],
            'fighting': ['fight', 'combat', 'violence', 'aggressive', 'conflict'],
            'football': ['soccer', 'football game', 'playing football', 'ball game'],
            'holding-book': ['reading', 'book', 'holding book', 'student', 'studying'],
            'person': ['people', 'human', 'individual', 'man', 'woman', 'crowd'],
            'plant': ['tree', 'vegetation', 'flowers', 'garden', 'greenery'],
            'restaurants': ['eating', 'dining', 'food', 'restaurant', 'cafe', 'meal'],
            'sitting': ['seated', 'sit down', 'sitting down', 'chair', 'bench'],
            'snatching': ['stealing', 'grabbing', 'theft', 'snatching bag', 'robbery'],
            'standing': ['stand up', 'standing up', 'upright', 'standing person'],
            'texting': ['text message', 'typing', 'mobile phone', 'smartphone', 'messaging'],
            'using-computer': ['computer', 'laptop', 'typing', 'working computer', 'desktop'],
            'vehicle': ['car', 'bus', 'truck', 'motorcycle', 'automobile', 'transport'],
            'walking': ['walk', 'pedestrian', 'stroll', 'walking person', 'on foot']
        }
        
        # Pre-compute embeddings for all classes and synonyms
        self._precompute_class_embeddings()
    
    def _precompute_class_embeddings(self):
        """Pre-compute embeddings for all classes and their synonyms"""
        self.class_embeddings = {}
        
        for class_name in self.classes:
            # Combine class name with synonyms
            all_terms = [class_name] + self.class_synonyms.get(class_name, [])
            
            # Compute embeddings
            embeddings = self.sentence_model.encode(all_terms)
            # Use mean embedding as class representation
            self.class_embeddings[class_name] = np.mean(embeddings, axis=0)
    
    def extract_relevant_classes(self, sentence: str, threshold: float = 0.3, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Extract relevant classes from input sentence using NLP and semantic similarity
        
        Args:
            sentence: Input text description
            threshold: Minimum similarity threshold
            top_k: Maximum number of classes to return
            
        Returns:
            List of (class_name, relevance_score) tuples
        """
        # Clean and preprocess sentence
        sentence = sentence.lower().strip()
        
        # Extract key phrases using spaCy if available
        key_phrases = self._extract_key_phrases(sentence)
        
        # Combine original sentence with key phrases for better matching
        text_for_matching = sentence + " " + " ".join(key_phrases)
        
        # Compute sentence embedding
        sentence_embedding = self.sentence_model.encode([text_for_matching])
        
        # Calculate similarities with all classes
        class_similarities = []
        for class_name, class_embedding in self.class_embeddings.items():
            similarity = cosine_similarity(sentence_embedding, [class_embedding])[0][0]
            if similarity >= threshold:
                class_similarities.append((class_name, similarity))
        
        # Sort by similarity and return top_k
        class_similarities.sort(key=lambda x: x[1], reverse=True)
        return class_similarities[:top_k]
    
    def _extract_key_phrases(self, sentence: str) -> List[str]:
        """Extract key phrases using spaCy NER and POS tagging"""
        if not self.nlp:
            return []
        
        doc = self.nlp(sentence)
        key_phrases = []
        
        # Extract named entities
        for ent in doc.ents:
            key_phrases.append(ent.text.lower())
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            key_phrases.append(chunk.text.lower())
        
        # Extract important verbs and adjectives
        for token in doc:
            if token.pos_ in ['VERB', 'ADJ'] and not token.is_stop:
                key_phrases.append(token.lemma_.lower())
        
        return list(set(key_phrases))  # Remove duplicates
    
    def run_yolo_detection(self, image: np.ndarray, relevant_classes: List[str] = None, 
                          conf_threshold: float = 0.5) -> List[Detection]:
        """
        Run YOLOv8 detection and filter by relevant classes
        
        Args:
            image: Input image as numpy array
            relevant_classes: List of class names to filter by
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of Detection objects
        """
        # Run YOLOv8 inference
        results = self.yolo_model(image, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Get class name
                    class_id = int(boxes.cls[i])
                    class_name = self.yolo_model.names[class_id]
                    
                    # Filter by relevant classes if specified
                    if relevant_classes is None or class_name in relevant_classes:
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        confidence = float(boxes.conf[i])
                        
                        detection = Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox
                        )
                        detections.append(detection)
        
        return detections
    
    def calculate_text_image_relevance(self, detections: List[Detection], 
                                     sentence: str) -> List[Detection]:
        """
        Calculate relevance scores between detected objects and input text
        
        Args:
            detections: List of Detection objects
            sentence: Input text description
            
        Returns:
            Updated detections with text_relevance scores
        """
        if not detections:
            return detections
        
        sentence_embedding = self.sentence_model.encode([sentence.lower()])
        
        for detection in detections:
            class_embedding = self.class_embeddings.get(detection.class_name)
            if class_embedding is not None:
                similarity = cosine_similarity(sentence_embedding, [class_embedding])[0][0]
                detection.text_relevance = similarity
            else:
                detection.text_relevance = 0.0
        
        return detections
    
    def rank_detections(self, detections: List[Detection], 
                       text_weight: float = 0.4, conf_weight: float = 0.6) -> List[Detection]:
        """
        Rank detections by combined text relevance and detection confidence
        
        Args:
            detections: List of Detection objects
            text_weight: Weight for text relevance score
            conf_weight: Weight for detection confidence
            
        Returns:
            Ranked list of detections
        """
        for detection in detections:
            # Combine text relevance and detection confidence
            combined_score = (text_weight * detection.text_relevance + 
                            conf_weight * detection.confidence)
            detection.combined_score = combined_score
        
        # Sort by combined score
        detections.sort(key=lambda x: getattr(x, 'combined_score', 0), reverse=True)
        return detections
    
    def process(self, sentence: str, image: np.ndarray, 
                max_detections: int = 10, conf_threshold: float = 0.5,
                text_threshold: float = 0.2) -> List[Detection]:
        """
        Main pipeline function to process text and image
        
        Args:
            sentence: Input text description
            image: Input image as numpy array
            max_detections: Maximum number of detections to return
            conf_threshold: YOLOv8 confidence threshold
            text_threshold: Text relevance threshold
            
        Returns:
            List of ranked Detection objects
        """
        print(f"Processing query: '{sentence}'")
        
        # Step 1: Extract relevant classes from text
        relevant_classes_scores = self.extract_relevant_classes(sentence, threshold=text_threshold)
        relevant_classes = [class_name for class_name, _ in relevant_classes_scores]
        
        print(f"Relevant classes identified: {relevant_classes_scores}")
        
        # Step 2: Run YOLOv8 detection
        detections = self.run_yolo_detection(image, relevant_classes, conf_threshold)
        
        print(f"Found {len(detections)} detections")
        
        # Step 3: Calculate text-image relevance
        detections = self.calculate_text_image_relevance(detections, sentence)
        
        # Step 4: Rank detections
        ranked_detections = self.rank_detections(detections)
        
        # Return top results
        return ranked_detections[:max_detections]
    
    def visualize_results(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Visualize detection results on image
        
        Args:
            image: Input image
            detections: List of Detection objects
            
        Returns:
            Image with bounding boxes and labels
        """
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Choose color based on ranking (top detections get brighter colors)
            color_intensity = max(100, 255 - (i * 30))
            color = (0, color_intensity, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if hasattr(detection, 'text_relevance'):
                label += f" (rel: {detection.text_relevance:.2f})"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image

# Example usage
def main():
    # Initialize pipeline
    pipeline = ActivityDetectionPipeline('path/to/your/yolov8_model.pt')
    
    # Load and process image
    image_path = 'path/to/your/image.jpg'
    image = cv2.imread(image_path)
    
    # Example queries
    queries = [
        "Show me people who are sitting in a restaurant",
        "Find someone using a computer or texting",
        "Detect any fighting or violent activities",
        "Show people walking or standing near vehicles"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        # Process query
        detections = pipeline.process(query, image)
        
        # Display results
        for i, detection in enumerate(detections[:5]):  # Show top 5
            print(f"{i+1}. {detection.class_name}: conf={detection.confidence:.3f}, "
                  f"text_rel={detection.text_relevance:.3f}, bbox={detection.bbox}")
        
        # Visualize results
        vis_image = pipeline.visualize_results(image, detections[:5])
        cv2.imshow(f'Results: {query[:30]}...', vis_image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()