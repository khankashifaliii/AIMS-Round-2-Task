Activity Detection Pipeline — YOLOv8 (yolov8m) + SentenceTransformer + spaCy

**Overview**

This repository contains an Activity Detection Pipeline that combines object detection (YOLOv8) with semantic text–image matching (SentenceTransformers + spaCy). The pipeline takes a free-form text query (e.g. "Show me people who are sitting in a restaurant") and an image, finds relevant object detections using a YOLOv8 model, computes semantic relevance between detected classes and the query, ranks detections, and visualizes results.

This README documents how to run, train and extend the code. The pipeline used a YOLOv8m model trained for 20 epochs on a dataset of ~15,000 images (you provided this training configuration).

**Key features**

Hybrid detection: YOLOv8 object detector + semantic matching using SentenceTransformer.

Text-guided detection: the input query suggests relevant object classes to filter and rank detections.

Pre-computed class embeddings & synonyms to improve semantic matching quality.

Visualization helper that draws ranked bounding boxes and textual relevance scores.
**Model & training details**

Base model: yolov8m (Ultralytics YOLOv8 medium variant)

Training epochs: 20 (as requested)

Dataset size: ~15,000 images

Model weight: https://drive.google.com/drive/folders/1h2PIzm2oMfC2MHrN8fujFYR4wLHpU4bQ?usp=sharing

Demo Video Link- https://youtu.be/ZZK3iW--zHE?si=E1GtGgDrkJY5fcoZ

Documentation Link- https://docs.google.com/document/d/1Hq6aIGVPaydoQ8USpluEi7yp2t7lxUaelzcP9AvA1oM/edit?usp=sharing
