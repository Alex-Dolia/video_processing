# Video Face Tracking with YOLOv8

This repository provides a Python-based solution for **face detection and tracking in videos** using the [YOLOv8](https://ultralytics.com/) model. The pipeline supports local and S3-based video input and performs frame-by-frame tracking with bounding boxes, confidence scores, and track IDs.

---

## Features

- Load YOLOv8 face detection model.
- Supports video input from local directories or AWS S3.
- Converts videos to frames for tracking.
- Performs multi-object tracking with detailed outputs.
- Returns tracking results including bounding boxes, frame number, face crop, and confidence.
- GPU-accelerated using PyTorch and CUDA if available.

---

## Requirements

### Conda Environment

A full Conda environment with PyTorch and required packages is provided in `environment.yml`. GPU support via CUDA is included.

```bash
conda env create -f environment.yml
conda activate tf
```

### Python Packages

The repository uses the following major Python packages:

- `torch`, `torchvision`, `torchaudio`
- `ultralytics` (YOLOv8)
- `opencv-python`
- `boto3` (for S3 access)
- `numpy`, `pandas`
- `json`, `datetime`

---

## Usage

### 1. Import the Library

```python
from video_processing import inference as inf
```

### 2. Load Model

```python
model_dir = "model"
model = inf.model_fn(model_dir)
```

### 3. Prepare Input

The input should be a CSV string specifying S3 or local video location:

```
s3_bucket|s3_key|video_filename|local_directory
```

```python
payload = "my_bucket|videos|video1.mp4|/tmp/videos"
content_type = "text/csv"
input_data = inf.input_fn(payload, content_type)
```

### 4. Run Prediction

```python
prediction_output = inf.predict_fn(input_data, model)
print(prediction_output["status"])
print(f"Number of unique faces: {prediction_output['nufs']}")
```

### 5. Handle Output

The output contains:

- `movie_tracker_output`: List of dicts, one per detected face per frame.
- `nufs`: Number of unique tracked faces.
- `status`: Processing log.

#### `movie_tracker_output` Structure

Each element in `movie_tracker_output` is a dictionary containing:

| Key            | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `n`            | Frame number in the video                                                   |
| `frame`        | The full frame as an RGB NumPy array                                        |
| `face`         | Cropped face image from the frame                                           |
| `height`       | Height of the cropped face                                                  |
| `width`        | Width of the cropped face                                                   |
| `channels`     | Number of channels in the cropped face (typically 3 for RGB)               |
| `track_id`     | Unique ID assigned to each tracked face                                     |
| `bounding_box` | Bounding box coordinates `[x1, y1, x2, y2]`                                 |
| `confidence`   | Detection confidence score for the face                                     |

Example:

```json
{
  "n": 12,
  "frame": [[...]],      # RGB NumPy array
  "face": [[...]],       # Cropped face
  "height": 128,
  "width": 128,
  "channels": 3,
  "track_id": 5,
  "bounding_box": [34, 50, 162, 178],
  "confidence": 0.87
}
```

---

## Video Tracking Workflow

1. **Model Loading** – Load YOLOv8 face detection model.
2. **Input Processing** – Download video from S3 or local directory, extract frames.
3. **Tracking** – Frame-by-frame face detection and tracking.
4. **Output Generation** – Structured JSON output with frame-level tracking info.

---

## AWS S3 Support

If using AWS S3, make sure your environment has the correct **AWS credentials**. The function `write_read_s3_movie()` handles downloading videos from S3.

---

## Notes

- GPU acceleration is enabled if a CUDA-capable device is available.
- Frames are generated as RGB arrays.
- Temporary files are saved locally in the specified directory.
- Tracking configuration (tracker type, confidence threshold, and IoU) can be adjusted in `predict_fn`.

---

## License

MIT License

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Boto3](https://boto3.amazonaws.com/)

