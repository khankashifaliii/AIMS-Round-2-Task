Activity Detection Pipeline — YOLOv8 (yolov8m) + SentenceTransformer + spaCy

Overview

This repository contains an Activity Detection Pipeline that combines object detection (YOLOv8) with semantic text–image matching (SentenceTransformers + spaCy). The pipeline takes a free-form text query (e.g. "Show me people who are sitting in a restaurant") and an image, finds relevant object detections using a YOLOv8 model, computes semantic relevance between detected classes and the query, ranks detections, and visualizes results.

This README documents how to run, train and extend the code. The pipeline used a YOLOv8m model trained for 20 epochs on a dataset of ~15,000 images (you provided this training configuration).
