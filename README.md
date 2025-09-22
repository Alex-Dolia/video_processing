# Video Processing Pipeline with Face Tracking and Embedding

## Overview

This pipeline is designed for video processing with face detection, tracking, embedding, and identification. 

The advantage of this approach is that it reduces the number of faces that might mistakenly be identified as a single face using a tracking algorithm. If there are two faces that belong to different tracks, they are excluded from the merging procedure. The code serves as a playground or starting point to learn how to use tracking, embedding, and Pearson identification.

Another approach, which is simpler but not implemented here, involves removing all detected faces with poor quality and then using face segmentation based solely on embeddings. This method does not use the information that two faces could belong to different tracks and only relies on embeddings.

### Example of Input video:
Place this video into alex_movies directory
[Input Video](https://drive.google.com/file/d/10oki6UShU9k30IWa2y9NhUe1FRi1v49M/view?usp=sharing) generated using GenAI [Kling](app.klingai.com)] using my face.

## NEW_pipeline_wo_out_fn_FINAL_v2

This function processes video input and performs several steps:

- **Tracking (Yolov8)**: Detects and tracks objects/faces in video frames.
- **Shot Detection**: Splits video into shots using `TransNetV2`.
- **Face Embedding (Facenet512)**: Extracts embeddings for detected faces. At the moment it uses `Facenet512` embeddings, but there are plans to support [InsightFace](https://github.com/deepinsight/insightface), [AdaFace](https://github.com/swghosh/adaface) and [MagFace](https://github.com/IrvingMeng/MagFace) embeddings.
- **Face Quality Assessment**: Evaluates face quality and adds scores.
- **Track Splitting**: Splits tracks when faces of good quality are different (e.g., different people in the same chair without a gap).
- **Track Merging**: Merges tracks where appropriate using embedding similarity.

The function returns a DataFrame with all processed tracks and a dictionary of computational times for each step.

### Example Output Columns

- `n`: frame number
- `height`, `width`, `channels`: face image dimensions
- `bounding_box`: coordinates of detected face
- `confidence`: detection confidence
- `face_quality`, `sharp_score`, `blur_score`: face quality metrics
- `track_id`, `split_track_id`, `merge_track_id`: tracking and merging identifiers

## Video Processing with YOLOv8 Tracking

This repository provides tools to track faces and objects in videos using YOLOv8 and additional tracking utilities. The workflow allows loading videos from a local directory or S3, processing frames, and performing tracking with detailed outputs.

## Installation

1. **Create conda environment**:

```bash
conda env create -f video_torch_gpu.yml -n video_torch_gpu
```

2. **Install PyTorch with CUDA support**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Install additional packages**:

```bash
pip install ultralytics-thop>=2.0.0
pip install tensorflow[and-cuda]
pip install jupyterlab
pip install ipywidgets
pip install imageio[ffmpeg]
```

## Directory Structure

```
video_processing/
│
├── demo_count_unique_faces.ipynb
├── model/     <-- Upload models here from Google Drive links provided below
│   ├── ediffiqaL.pth
│   ├── pointrend_resnet50.pkl
│   ├── yolov8n-face.pt
│   ├── facenet512_weights.h5
│   ├── osnet_x0_25.pth
│   ├── r100.pth
│   └── mask_rcnn_coco.h5
│
├── src_yolov8_tracking/
├── src_tracking_embedding/
├── src_tracking_face_quality/
├── src_tracking_split/
├── src_tracking_merge/
├── utils_split_merge/
├── shot_detection/
├── deepface_ad/
└── alex_movies/   <-- Input videos
```

## Usage

You can call the YOLOv8 tracking functions from the `demo_count_unique_faces.ipynb` notebook.

### Example: Tracking Faces in a Video

```python
from src_yolov8_tracking import inference as inf
import os

# Set working directories
model_dir = "model"
video_file = "alex_movies/input_video.mp4"
payload = f"LOCAL|LOCAL|{os.path.basename(video_file)}|{os.path.dirname(video_file)}"
content_type = "text/csv"

# Load model
model = inf.model_fn(model_dir)

# Prepare input frames
input_data = inf.input_fn(payload, content_type)

# Run tracking
prediction_output = inf.predict_fn(input_data, model)

# Inspect outputs
print("Status:", prediction_output["status"])
print("Number of unique tracks:", prediction_output["nufs"])
print("Movie tracker output (first 5 entries):")
for track in prediction_output["movie_tracker_output"][:5]:
    print({
        "frame_index": track["n"],
        "track_id": track["track_id"],
        "bounding_box": track["bounding_box"],
        "confidence": track["confidence"],
        "face_shape": (track["height"], track["width"], track["channels"])
    })
```

### `movie_tracker_output` Details

Each element contains:

- `n`: frame index
- `frame`: actual frame (RGB numpy array)
- `face`: cropped face from the frame
- `height`, `width`, `channels`: face dimensions
- `track_id`: unique ID for tracked face/object
- `bounding_box`: `[x1, y1, x2, y2]`
- `confidence`: detection confidence score

This allows for downstream analysis, saving face crops, or further processing with other tracking modules.

## Model Files

This project requires several pretrained model files.  
Download them from the provided Google Drive links and place them into the `model` directory.

### Download Links
- [ediffiqaL.pth](https://drive.google.com/file/d/1w1Kp7AUGfb5nEVfQ6bMda9VrT_4M_7Oe/view?usp=sharing)
- [pointrend_resnet50.pkl](https://drive.google.com/file/d/1lust2O8Vt0THskJDi5ZEd2v29ytEgI5x/view?usp=sharing)
- [yolov8n-face.pt](https://drive.google.com/file/d/1K8grUb2xgkPdWuunKSb0BAALaU__YQCV/view?usp=sharing)
- [facenet512_weights.h5](https://drive.google.com/file/d/1pX4H0dygCnyrpUEx5xf0Kuw-yK2Dqapg/view?usp=sharing)
- [osnet_x0_25.pth](https://drive.google.com/file/d/1QLGFRCDjwiAlCKiaZrxFUYbubAQSZgim/view?usp=sharing)
- [r100.pth](https://drive.google.com/file/d/1yIi2hqYpDqbzh8J9xtp-AHFPKU340R1H/view?usp=sharing)
- [mask_rcnn_coco.h5](https://drive.google.com/file/d/1QZcQf8y4KA2f7aIM7HMB0jdw5ms0cTgc/view?usp=sharing)

### Copy the downloaded files to `model` directory:

```bash
mkdir -p model
cp /path/to/downloaded/files/* model/
```

(Replace `/path/to/downloaded/files` with the location where the files were downloaded.)

## Notes

- Ensure GPU is available; PyTorch will detect CUDA automatically.
- Input videos should be placed in `alex_movies/` or configured for S3 input.
- The notebook `demo_count_unique_faces.ipynb` demonstrates end-to-end usage.

## TransNet V2: Shot Boundary Detection Neural Network  

**TransNet V2** is a fast and effective deep network architecture for shot transition detection.  
- [TransNet V2 Paper on arXiv](https://arxiv.org/abs/2008.04838)  
- [TransNet V2 Inference GitHub](https://github.com/soCzech/TransNetV2/tree/master/inference)