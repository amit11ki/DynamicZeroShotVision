# Zero-Shot Object Detection with OWL-ViT

This project demonstrates real-time **zero-shot object detection** using the [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) model from Hugging Face Transformers. The system can detect **objects described by arbitrary text prompts** without retraining.

---

## 🧠 Overview

**DynamicZeroShotVision** is designed to:

- Accept input from a webcam or a local video file.
- Use custom object categories (e.g., lightbulb, matchstick, monitor, lion, gaming console) that are not part of the COCO dataset.
- Process each video frame using a zero-shot model to detect objects.
- Display annotated video with bounding boxes, labels, and confidence scores.
- Allow live updating of detection classes during runtime.
- Log predictions for further analysis.
- Video Recording: Save processed videos with detection overlays to file

---


## 🖥️ Example Classes

The system is initialized with some example classes:

```text
lightbulb, matchstick, monitor, lion, gaming console
```

You can change these live by pressing `c` during detection.

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/amit11ki/DynamicZeroShotVision.git
cd DynamicZeroShotVision
```

2. **Install dependencies**
```bash
pip install torch opencv-python transformers numpy pillow
```

---

## ▶️ Usage

### Command Line Arguments
```bash
python main.py --source 0                      # Use webcam (default)
python main.py --source path/to/video.mp4      # Use video file
python main.py --no-jit                        # Disable TorchScript acceleration
python main.py --save-video                    # Save processed video to output directory
python main.py --output path/to/output.mp4     # Specify output video path
```

## ⌨️ Keyboard Controls
While the application is running:

- `q`: Quit the program
- `c`: Change detection classes dynamically

---

## 📝 Logs

Detection results are saved to:
```
detections_log.json
```

Each entry includes the frame number, timestamp, and detected objects.

---

## 📁 File Structure

```text
├── main.py              # Main detection script
├── requirements.txt     # Python dependencies
├── detections_log.json  # Output logs
└── README.md            # Project documentation
```

---

## 🧠 Model Info

- Model: `google/owlvit-base-patch32`
- Source: [Hugging Face](https://huggingface.co/google/owlvit-base-patch32)
- Framework: PyTorch + Transformers

---

## 💡 Notes

- Performance may vary based on GPU/CPU.
- Resize dimensions for detection: **320x320**.
- Supports only custom classes.

---

## 📜 License

This project is for educational/research purposes. Respect model and dataset licenses.



## 🔮 Future Improvements

- **Live Prompt Editing:** Change detection classes without pausing the stream.
- **Performance Optimizations:** Speed up frame processing and inference.
- **Dashboard UI:** Visualize detections and stats via a browser-based dashboard.
- **Model Acceleration:** Explore ONNX or advanced TorchScript improvements.

