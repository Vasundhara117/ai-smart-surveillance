# Member 1: Detection and Tracking

This standalone Python module reads a video one frame at a time, detects objects with a pretrained Ultralytics YOLO model, gives detections stable IDs using class-aware IoU matching, and writes the agreed per-frame JSON interface. It does not classify activities or generate alerts.

## Input video

Provide a local CCTV or ordinary video file such as MP4, AVI, or MOV that can be opened by your OpenCV build. For example, use a short, consented, fixed-camera clip in which a person walks through view and a vehicle is visible. The tracker can preserve an ID while detections overlap between consecutive frames; an ID may change after a long occlusion or abrupt movement.

The repository does not include a video recording. The supplied `sample_detection.json` demonstrates the output format; its example detections are illustrative, not model output.

## Setup

Python 3.10 or later is recommended. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r member1_detection\requirements.txt
```

On first run, Ultralytics downloads the selected pretrained weights (default `yolo11n.pt`). Use `--model` to select another compatible local or pretrained Ultralytics model. Inference uses the model's automatic device selection by default; `--device cpu` forces CPU execution.

## Run

From the project root:

```powershell
python member1_detection\video_processor.py --input path\to\input.mp4 --output member1_detection\detection.json --annotated member1_detection\annotated.mp4
```

The annotated output is optional. To detect every class available in the selected model rather than the default surveillance-relevant subset:

```powershell
python member1_detection\video_processor.py --input path\to\input.mp4 --output member1_detection\detection.json --classes all
```

You can select class names, confidence threshold, or tracker retention explicitly:

```powershell
python member1_detection\video_processor.py --input path\to\input.mp4 --output member1_detection\detection.json --classes person,car,bicycle --confidence 0.3 --max-age 30 --min-iou 0.3 --device cpu
```

The default classes are `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `backpack`, `handbag`, and `suitcase`, subject to the selected model's class labels. The output is a JSON array containing one record per decoded frame, including frames with no objects. Frame numbers are 1-based, timestamps are seconds from the start of the clip, and boxes are `[x1, y1, x2, y2]` pixel coordinates.

## Output interface

Each array entry follows the fixed Member 2 interface:

```json
{
  "frame": 120,
  "timestamp": 4.0,
  "objects": [
    {
      "id": 7,
      "class": "person",
      "confidence": 0.91,
      "bbox": [120, 80, 250, 400]
    }
  ]
}
```

`id` is a tracker-assigned integer for the detected object, `class` is the model's class label, and `confidence` is the detector confidence. IDs are local to one processed video and can change when an object is missed longer than the configured `--max-age` or moves too far for IoU matching.

## Detection and tracking approach

- **Detection:** pretrained Ultralytics YOLO inference is run independently on each decoded BGR frame. No training is performed by this module.
- **Tracking:** detections are greedily matched to recent tracks of the same class using bounding-box intersection over union (IoU). Unmatched detections get new IDs; stale tracks are discarded. This lightweight tracker is suitable for a course-project baseline, not guaranteed identity recognition through severe occlusion or crowded crossings.
- **Boundary:** output describes detected objects only. It makes no determination that an action is suspicious; downstream activity analysis is outside Member 1's scope.

## Test and performance

Run the dependency-free tracker and output-interface tests from the project root:

```powershell
python -m unittest discover -s member1_detection -p "test_*.py" -v
```

When processing a video, the command prints elapsed time and end-to-end processed FPS. Actual throughput depends on video resolution, model size, CPU/GPU, and hardware; YOLO nano weights are the lightweight default. For reliable comparison, use the same input clip, model, and device.
