import cv2
import numpy as np
import time
import argparse
from collections import deque
import os

class EnhancedDetector:
    """Enhanced detector with multiple detection methods"""

    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold

        # Multiple background subtractors for robustness
        self.bg_subtractors = [
            cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True),
            cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
        ]

        # Lower minimum area for better small object detection
        self.min_contour_area = 100  # Reduced from 500
        self.max_contour_area = 50000

        # Enhanced class names
        self.classes = ["object", "person", "vehicle", "animal", "unknown"]
        self.colors = [
            (0, 255, 255),  # Yellow for general objects
            (255, 0, 0),  # Blue for people
            (0, 0, 255),  # Red for vehicles
            (0, 255, 0),  # Green for animals
            (255, 255, 255)  # White for unknown
        ]

        # Detection history for stability
        self.detection_history = deque(maxlen=10)
        print("Enhanced detector initialized")

    def detect(self, frame):
        """Enhanced object detection with multiple methods"""
        height, width = frame.shape[:2]
        all_detections = []

        # Method 1: Background Subtraction
        bg_detections = self._detect_with_background_subtraction(frame)
        all_detections.extend(bg_detections)

        # Method 2: Color-based detection (for yellow objects)
        yellow_detections = self._detect_yellow_objects(frame)
        all_detections.extend(yellow_detections)

        # Method 3: Edge-based detection
        edge_detections = self._detect_with_edges(frame)
        all_detections.extend(edge_detections)

        # Remove duplicates using NMS
        final_detections = self._non_max_suppression(all_detections)

        return final_detections

    def _detect_with_background_subtraction(self, frame):
        """Detect moving objects using background subtraction"""
        detections = []

        for bg_subtractor in self.bg_subtractors:
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)

            # Enhanced noise removal
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_contour_area < area < self.max_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate confidence based on area and solidity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    confidence = min(0.8, (area / 2000) * solidity)

                    if confidence > self.confidence_threshold:
                        class_name = self._classify_object(w, h, area, solidity)
                        detections.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'class_id': self.classes.index(class_name),
                            'class_name': class_name,
                            'method': 'motion'
                        })

        return detections

    def _detect_yellow_objects(self, frame):
        """Specifically detect yellow-colored objects"""
        detections = []

        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define yellow color range in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create mask for yellow objects
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in yellow mask
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Lower threshold for color-based detection
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate confidence based on color consistency
                roi = frame[y:y + h, x:x + w]
                if roi.size > 0:
                    yellow_ratio = np.sum(yellow_mask[y:y + h, x:x + w] > 0) / (w * h)
                    confidence = min(0.9, yellow_ratio * 0.8)

                    if confidence > self.confidence_threshold:
                        detections.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'class_id': 0,  # "object"
                            'class_name': "yellow_object",
                            'method': 'color'
                        })

        return detections

    def _detect_with_edges(self, frame):
        """Detect objects using edge detection"""
        detections = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 10000:  # Reasonable size range for objects
                x, y, w, h = cv2.boundingRect(contour)

                # Simple confidence based on edge density
                roi_edges = edges[y:y + h, x:x + w]
                edge_density = np.sum(roi_edges > 0) / (w * h) if (w * h) > 0 else 0
                confidence = min(0.7, edge_density * 2)

                if confidence > self.confidence_threshold:
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'class_id': 0,  # "object"
                        'class_name': "edge_object",
                        'method': 'edges'
                    })

        return detections

    def _classify_object(self, width, height, area, solidity):
        """Classify object based on features"""
        aspect_ratio = width / height if height > 0 else 0

        if 0.8 < aspect_ratio < 1.2 and 1000 < area < 10000:
            return "person"
        elif aspect_ratio > 1.5 and area > 2000:
            return "vehicle"
        elif 0.5 < aspect_ratio < 2.0 and area > 1500:
            return "animal"
        else:
            return "object"

    def _non_max_suppression(self, detections, threshold=0.5):
        """Remove duplicate detections using NMS"""
        if len(detections) == 0:
            return []

        # Extract bounding boxes and confidences
        boxes = np.array([det['bbox'] for det in detections])
        confidences = np.array([det['confidence'] for det in detections])

        # Convert to [x1, y1, x2, y2] format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = confidences.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]


class YOLODetector:
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model_loaded = False

        # Use enhanced detector as primary fallback
        self.enhanced_detector = EnhancedDetector(confidence_threshold)

        try:
            if self._check_yolo_files():
                self.net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

                with open('coco.names', 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]

                self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
                self.model_loaded = True
                print("YOLO model loaded successfully!")
            else:
                raise Exception("YOLO files not available")

        except Exception as e:
            print(f"YOLO model failed to load: {e}")
            print("Using enhanced detector as primary")
            self.model_loaded = False

    def _check_yolo_files(self):
        """Check if YOLO files exist"""
        required_files = ['yolov3.cfg', 'yolov3.weights', 'coco.names']
        return all(os.path.exists(f) for f in required_files)

    def detect(self, frame):
        """Detect objects using YOLO or enhanced detector"""
        if not self.model_loaded:
            return self.enhanced_detector.detect(frame)

        # YOLO detection code
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]],
                    'method': 'yolo'
                })

        return detections


class DeepSORTTracker:
    def __init__(self, max_age=30, n_init=3, max_iou_distance=0.7):
        self.next_id = 1
        self.tracks = []
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance

    def iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection coordinates
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        # Calculate intersection area
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def hungarian_matching(self, detections, tracks):
        """Perform Hungarian matching between detections and tracks"""
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))

        # Create cost matrix
        cost_matrix = np.zeros((len(detections), len(tracks)))

        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou_score = self.iou(det['bbox'], track['bbox'])
                cost_matrix[i, j] = 1 - iou_score  # Convert similarity to distance

        # Simple greedy matching (works without scipy)
        return self.greedy_matching(detections, tracks, cost_matrix)

    def greedy_matching(self, detections, tracks, cost_matrix):
        """Greedy matching implementation that doesn't require scipy"""
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))

        # Create list of potential matches
        potential_matches = []
        for i in range(len(detections)):
            for j in range(len(tracks)):
                if cost_matrix[i, j] < self.max_iou_distance:
                    potential_matches.append((i, j, cost_matrix[i, j]))

        # Sort by cost (lowest first)
        potential_matches.sort(key=lambda x: x[2])

        # Match greedily
        matched_dets = set()
        matched_trks = set()

        for i, j, cost in potential_matches:
            if i not in matched_dets and j not in matched_trks:
                matches.append((i, j))
                matched_dets.add(i)
                matched_trks.add(j)

        # Update unmatched lists
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [j for j in range(len(tracks)) if j not in matched_trks]

        return matches, unmatched_detections, unmatched_tracks

    def update(self, detections):
        """Update tracker with new detections"""
        # Update existing tracks age
        for track in self.tracks:
            track['age'] += 1
            track['time_since_update'] += 1

        # Match detections to tracks
        matches, unmatched_dets, unmatched_trks = self.hungarian_matching(detections, self.tracks)

        # Update matched tracks
        for det_idx, track_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]

            # Update track with new detection
            track['bbox'] = det['bbox']
            track['confidence'] = det['confidence']
            track['class_name'] = det['class_name']
            track['hits'] += 1
            track['time_since_update'] = 0

            # Add method information to track
            if 'method' in det:
                track['method'] = det['method']

            if track['hits'] >= self.n_init and not track['confirmed']:
                track['confirmed'] = True

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = {
                'id': self.next_id,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_name': det['class_name'],
                'age': 0,
                'hits': 1,
                'time_since_update': 0,
                'confirmed': False,
                'color': tuple(np.random.randint(0, 255, 3).tolist())
            }

            # Add method information if available
            if 'method' in det:
                new_track['method'] = det['method']

            self.tracks.append(new_track)
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [track for track in self.tracks
                       if track['time_since_update'] <= self.max_age]

        # Return only confirmed tracks
        confirmed_tracks = [track for track in self.tracks if track['confirmed']]
        return confirmed_tracks


class ObjectDetectionTracker:
    def __init__(self, source=0, output_file=None, confidence=0.3):
        self.source = source
        self.output_file = output_file

        # Initialize detector with enhanced capabilities
        self.detector = YOLODetector(confidence_threshold=confidence)
        self.tracker = DeepSORTTracker()

        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer if output file specified
        self.writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))

        # Performance tracking
        self.frame_count = 0
        self.fps_deque = deque(maxlen=30)

        print(f"Video source: {source}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print("Enhanced object detection ready!")

    def draw_detections(self, frame, tracks):
        """Draw bounding boxes and labels on frame with method info"""
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            class_name = track['class_name']
            confidence = track['confidence']
            color = track['color']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with detection method
            method = track.get('method', 'unknown')
            label = f"{class_name} ID:{track_id} {confidence:.2f} ({method})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_stats(self, frame, processing_time, num_objects):
        """Draw performance statistics on frame"""
        # Calculate current FPS
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        self.fps_deque.append(current_fps)
        avg_fps = np.mean(self.fps_deque)

        # Draw stats panel
        stats_bg = np.zeros((140, frame.shape[1], 3), dtype=np.uint8)
        frame[0:140, 0:frame.shape[1]] = cv2.addWeighted(frame[0:140, 0:frame.shape[1]],
                                                         0.3, stats_bg, 0.7, 0)

        detector_type = "Enhanced Detector" if not self.detector.model_loaded else "YOLO"

        stats = [
            f"FPS: {avg_fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Objects: {num_objects}",
            f"Detector: {detector_type}",
            f"Processing: {processing_time * 1000:.1f}ms",
            "Press 'q': quit, 'p': pause, 's': save frame",
            "Yellow objects should be detected now!"
        ]

        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def run(self):
        """Main loop for object detection and tracking"""
        print("Starting enhanced object detection and tracking...")
        print("Press 'q' to quit, 'p' to pause, 's' to save frame")

        paused = False

        while True:
            if not paused:
                start_time = time.time()

                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream")
                    break

                self.frame_count += 1

                try:
                    # Detect objects
                    detections = self.detector.detect(frame)

                    # Update tracker
                    tracks = self.tracker.update(detections)

                    # Draw results
                    self.draw_detections(frame, tracks)

                    # Calculate processing time
                    processing_time = time.time() - start_time

                    # Draw statistics
                    self.draw_stats(frame, processing_time, len(tracks))

                    # Write frame if output specified
                    if self.writer:
                        self.writer.write(frame)

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Draw error message on frame
                    cv2.putText(frame, f"Error: {str(e)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display frame
                cv2.imshow('Object Detection & Tracking', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                status = "Paused" if paused else "Resumed"
                print(status)
                # Show pause status on frame
                cv2.putText(frame, status, (self.width // 2 - 50, self.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Enhanced Object Detection & Tracking', frame)
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

        # Cleanup
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

        print("Processing completed!")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Object Detection and Tracking')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path (optional)')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Detection confidence threshold (0.1-0.9)')

    args = parser.parse_args()

    # Convert source to int if webcam
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    try:
        # Initialize and run the tracker
        tracker = ObjectDetectionTracker(source=source, output_file=args.output, confidence=args.confidence)
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. For webcam: Make sure webcam is connected and not used by another application")
        print("2. For video file: Check if file exists and is a supported format")
        print("3. Install required packages: pip install opencv-python numpy")


if __name__ == "__main__":
    main()