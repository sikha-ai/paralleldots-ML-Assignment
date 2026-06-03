# Autonomous Retail Shelf Intelligence Inference Pipeline

## Project Description
[cite_start]This repository contains a modular, production-ready computer vision inference pipeline designed to automate retail shelf intelligence tasks, specifically **On-Shelf Availability (OSA)** and **Share of Shelf (SOS)** tracking[cite: 3, 4, 7]. 

Moving away from traditional, rigid end-to-end classifiers that require thousands of custom-labeled domain images and break whenever a brand updates its packaging, this pipeline utilizes a unique **Zero-Shot Hybrid Architecture**. [cite_start]By decoupling object localization from semantic identification, the pipeline dynamically detects generic product shapes, classifies multi-category store brands via text descriptions, and runs localized text recognition (OCR) alongside geometric spatial row clustering to deliver robust business metrics[cite: 14, 18, 33].

---

## Technical Architecture & Pipeline Flow

[cite_start]The workflow is engineered sequentially to optimize inference speeds and handle data modularly[cite: 40, 45]:

[Input Shelf Image]
│
▼
[Stage 1: Object Localization (YOLOv8x)] ──────► Total Product Count
│
├── (Crop Dynamic Product Patches)
▼
[Stage 2: Semantic Identity (CLIP ViT)] ────────► Brand-Wise Classification Counts
│
├── (Coordinate Extractions)
▼
[Stage 3: Spatial Row Segmentation (1D DBSCAN)] ─► Share of Shelf (SOS) Analytics
│
▼
[Stage 4: Text Processing Engine (EasyOCR)] ────► Extracted Price Tag / Label Context

### Step-by-Step Pipeline Mechanics
1. **Product Detection (Localization):** The input image is passed through a deep learning object detector to extract bounding boxes for all visible product facings, completely independent of their brand[cite: 19, 20].
2. **Brand Classification (Semantics):** The pipeline dynamically crops each detected product bounding box. These image patches are passed to a vision-language model that matches the visual features against textual brand descriptions (`"Coca-Cola"`, `"Pepsi"`, `"Lay's"`, `"Doritos"`) on the fly[cite: 22, 23, 24, 25, 26, 27].
3. **Spatial Row Segmentation (Analytics):** To estimate shelf space without heavy instance segmentation masks, the pipeline extracts the vertical ($Y$-center) coordinates of all bounding boxes and groups them using a density-based clustering algorithm. This maps items into distinct horizontal rows[cite: 31, 32].
4. **Text Extraction (OCR):** The system processes the image to find text blocks, applying string filters to capture numbers and currency patterns matching shelf price tags[cite: 29, 30].

---

## Model Selection Rationale & Trade-offs

| Pipeline Stage | Chosen Open-Source Tool | Engineering Advantages | Technical Trade-offs & Limitations [cite: 66] |
| :--- | :--- | :--- | :--- |
| **1. Product Detection** [cite: 9] | `YOLOv8x` (Pre-trained on COCO) | High-density feature extraction; exceptionally fast at localizing dense shapes (bottles, packets, cartons)[cite: 66, 68]. | Stock weights do not classify fine-grained retail boundaries out of the box. |
| **2. Brand Classification** [cite: 10] | `CLIP` (ViT-B/32 by OpenAI) | Zero-shot text-to-image embeddings. Eliminates the data-labeling bottleneck, scaling to new brands instantly via text prompts. | Inference time scales linearly with the number of text label comparisons. |
| **3. Shelf Segmentation** [cite: 12] | `1D DBSCAN` (Heuristic Density Clustering) | Zero computational/GPU overhead. Groups items horizontally by mathematical $Y$-center proximity[cite: 66]. | Assumes camera angle is relatively parallel to the shelving unit. |
| **4. Label OCR** [cite: 11] | `EasyOCR` | Lightweight; robust at parsing scene text, numbers, and currency symbols from varying backgrounds. | Raw scene OCR can pick up noisy text from product packaging if not filtered contextually. |

### CPU vs. GPU Considerations & Deployment Practicality [cite: 66]
* **Development Context:** The pipeline leverages GPU acceleration (CUDA) if available (e.g., Google Colab runtimes) to speed up batch processing, but easily falls back to CPU execution[cite: 66].
* **Production Edge Scalability:** For large-scale in-store deployments, the deep learning components (`YOLOv8x` and `CLIP`) can be exported to highly optimized **ONNX** or **OpenVINO** runtimes. This reduces the compute footprint drastically, allowing the entire pipeline to execute efficiently on cost-effective, CPU-only edge gateways or smart store cameras[cite: 66].

---

## System Assumptions and Practical Limitations 
* **Package Deformation & Occlusion:** Tightly packed or deformed snack bags (as seen in `img_2.jpg`) can cause bounding box overlap. This is addressed by tuning the Non-Maximum Suppression (NMS) confidence thresholds to minimize false merges.
* **Text Noise Mitigation:** Product descriptions on retail boxes can sometimes be misread as price tags. To filter this out, the text processing pipeline uses validation filtering looking specifically for numerical patterns and retail symbols (e.g., digits, currency signs).

---

## Environment Setup & Execution Instructions 

Follow these instructions to install dependencies and execute the pipeline prototype:

### 1. Installation
Ensure you have Python 3.8+ installed. Install the frameworks via your terminal:
```bash
pip install ultralytics easyocr transformers torch torchvision opencv-python-headless scikit-learn
