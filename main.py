import cv2  # OpenCV for video processing
import torch  # PyTorch for tensor computations and neural networks
import time  # For measuring execution time
import json  # For working with JSON data (for logging)
import numpy as np
import threading
import queue
import argparse  # For handling command-line arguments
from PIL import Image  # For image manipulation
from transformers import OwlViTProcessor, OwlViTForObjectDetection  # Hugging Face Transformers for OWL-ViT
from torch.jit import trace  # For model acceleration

# ----------- Configuration -----------
RESIZE_WIDTH, RESIZE_HEIGHT = 320, 320  # Resize dimensions for frames
CONFIDENCE_THRESHOLD = 0.04  # Minimum confidence for valid detection
DEFAULT_CLASSES = ["lightbulb", "matchstick", "monitor", "lion", "gaming console"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
LOG_FILE = "detections_log.json"
WINDOW_NAME = "Zero-Shot Object Detection"
PROCESS_EVERY_N = 2  # Process every Nth frame
MAX_QUEUE_SIZE = 10  # Max size for frame queue in threading
# -------------------------------------

COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
}

class ZeroShotDetector:
    """Class to handle zero-shot object detection using OWL-ViT."""
    def __init__(self, model_name="google/owlvit-base-patch32", use_jit=True):
        """Initialize with model and processor."""
        print(f"[INFO] Loading model on {DEVICE}...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(model_name).to(DEVICE)
        model.eval()

        self.coco_classes = COCO_CLASSES
        print("[INFO] Loaded COCO dataset class names from provided set.")

        self.use_jit = use_jit
        if use_jit and DEVICE == "cuda":
            print("[INFO] Creating TorchScript model for acceleration...")
            try:
                self.model = self._create_traced_model(model)
                print("[INFO] TorchScript acceleration enabled")
            except Exception as e:
                print(f"[WARNING] Failed to create TorchScript model: {e}")
                print("[INFO] Falling back to standard model")
                self.model = model
                self.use_jit = False
        else:
            self.model = model

        self.custom_classes = DEFAULT_CLASSES.copy()
        self.logs = []
        self.frame_count = 0
        self.processed_frames = 0
        self.fps = 0

    def _create_traced_model(self, model):
        """Create a TorchScript traced model for faster inference."""
        dummy_processor = self.processor(text=["dummy"], images=Image.new('RGB', (RESIZE_WIDTH, RESIZE_HEIGHT)), return_tensors="pt").to(DEVICE)
        traced_model = trace(model, example_inputs=(dummy_processor))
        return traced_model

    def preprocess_frame(self, frame):
        """Resize and convert OpenCV frame to PIL format."""
        resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), resized.shape[:2]

    def detect_objects(self, frame):
        """Perform object detection on a single frame."""
        pil_img, (resized_h, resized_w) = self.preprocess_frame(frame)
        orig_h, orig_w = frame.shape[:2]

        inputs = self.processor(text=self.custom_classes, images=pil_img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            start_inf = time.time()
            outputs = self.model(**inputs)
            inf_time = time.time() - start_inf

        target_sizes = torch.tensor([[resized_h, resized_w]], device=DEVICE)
        results = self.processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD)[0]

        scale_factors = (orig_h / resized_h, orig_w / resized_w)

        return results, scale_factors, inf_time

    def draw_detections(self, frame, results, scale_factors):
        """Draw bounding boxes and labels on the original frame."""
        detections = []
        scale_y, scale_x = scale_factors

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            if score_val < CONFIDENCE_THRESHOLD:
                continue

            class_name = self.custom_classes[label]
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            color = (0, int(255 * min(score_val, 1)), 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{class_name}: {score_val:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, max(y1 - text_size[1] - 10, 0)), (x1 + text_size[0], max(y1, 0)), color, -1)
            cv2.putText(frame, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            detections.append({"class": class_name, "score": float(score_val), "box": [x1, y1, x2, y2]})

        return frame, detections

    def update_classes(self, new_classes_str):
        """Update detection classes at runtime, excluding COCO classes."""
        if new_classes_str:
            new_classes = [x.strip() for x in new_classes_str.split(",")]
            added_classes = []
            skipped_classes = []
            for cls in new_classes:
                if cls.lower() in [c.lower() for c in self.coco_classes]:
                    skipped_classes.append(cls)
                else:
                    if cls not in self.custom_classes:
                        self.custom_classes.append(cls)
                        added_classes.append(cls)

            if added_classes:
                print("[INFO] Added custom classes:", added_classes)
            if skipped_classes:
                print(f"[INFO] Skipped COCO dataset classes:", skipped_classes)
            return True
        return False

    def log_detection(self, frame_idx, detections):
        """Add detection to logs."""
        self.logs.append({"frame": frame_idx, "timestamp": time.time(), "detections": detections})

    def save_logs(self):
        """Save detection logs to JSON file."""
        with open(LOG_FILE, "w") as f:
            json.dump(self.logs, f, indent=4)
        print(f"[INFO] Detections saved to {LOG_FILE}")


class ThreadedVideoProcessor:
    """Handles video processing in separate threads."""
    def __init__(self, detector, video_source=0):
        """Initialize with detector and video source."""
        self.detector = detector
        self.video_source = video_source
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.cap = None
        self.is_running = False
        self.threads = []

    def start(self):
        """Start the video processing threads."""
        self.is_running = True
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.video_source}")
            return False

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[INFO] Video source: {self.video_source} ({self.frame_width}x{self.frame_height} @ {self.fps:.2f}fps)")

        self.threads.append(threading.Thread(target=self._capture_thread))
        self.threads.append(threading.Thread(target=self._detection_thread))

        for thread in self.threads:
            thread.daemon = True
            thread.start()

        return True

    def _capture_thread(self):
        """Thread to capture frames from the video source."""
        frame_count = 0
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.is_running = False
                break

            frame_count += 1
            if frame_count % PROCESS_EVERY_N == 0:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put((frame_count, frame), block=False)
                except queue.Full:
                    pass

    def _detection_thread(self):
        """Thread to perform detection on queued frames."""
        while self.is_running:
            try:
                frame_count, frame = self.frame_queue.get(timeout=1.0)
                results, scale_factors, inf_time = self.detector.detect_objects(frame)
                annotated_frame, detections = self.detector.draw_detections(frame.copy(), results, scale_factors)
                self.detector.processed_frames += 1
                self.detector.log_detection(frame_count, detections)
                self.result_queue.put((frame_count, annotated_frame, inf_time))
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Detection error: {e}")
                continue

    def stop(self):
        """Stop all threads and release resources."""
        self.is_running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        print("[INFO] Video processing stopped")


def create_ui(frame, fps, inf_time):
    """Add UI elements to the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Inference: {inf_time*1000:.1f}ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit, 'c' to change classes", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def run_detection(video_source=0, use_jit=True):
    """Main function to run the detection pipeline."""
    detector = ZeroShotDetector(use_jit=use_jit)
    processor = ThreadedVideoProcessor(detector, video_source)
    if not processor.start():
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    start_time = time.time()
    fps_start_time = start_time
    fps_counter = 0
    current_fps = 0
    avg_inf_time = 0

    try:
        while processor.is_running:
            try:
                frame_idx, frame, inf_time = processor.result_queue.get(timeout=0.1)
                fps_counter += 1
                avg_inf_time = 0.9 * avg_inf_time + 0.1 * inf_time if avg_inf_time > 0 else inf_time

                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_counter / (current_time - fps_start_time)
                    detector.fps = current_fps
                    fps_counter = 0
                    fps_start_time = current_time

                frame = create_ui(frame, current_fps, avg_inf_time)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    cv2.destroyWindow(WINDOW_NAME)
                    new_classes = input("Enter new custom classes (comma-separated): ").strip()
                    detector.update_classes(new_classes)
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        processor.stop()
        cv2.destroyAllWindows()
        detector.save_logs()
        total_time = time.time() - start_time
        print(f"[INFO] Processed {detector.processed_frames} frames in {total_time:.2f} seconds.")
        print(f"[INFO] Average FPS: {detector.fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Object Detection")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--no-jit", action="store_true", help="Disable TorchScript acceleration")
    args = parser.parse_args()

    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    run_detection(video_source=video_source, use_jit=not args.no_jit)
