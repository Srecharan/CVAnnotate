import rosbag
import cv2
import numpy as np
import json
from cv_bridge import CvBridge
import os
from datetime import datetime
from ultralytics import YOLO

class BinDetector:
    def __init__(self, conf_threshold_person=0.2, person_overlap_threshold=0.5):
        self.person_model = YOLO("person_best.pt")
        self.bridge = CvBridge()
        self.conf_threshold_person = conf_threshold_person
        self.person_overlap_threshold = person_overlap_threshold
        self.background_subtractors = {}
        self.min_contour_area = 1000
        self.learning_rate = 0.005
        self.gaussian_kernel = (5, 5)
        self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.var_threshold = 50

        self.warmup_frames = 30  
        self.warmup_counters = {}  

        self.counters = {
            'drywall': 0,
            'aggregate': 0,
            'wood': 0,
            'cardboard': 0,
            'total': 0
        }
        
        self.previous_valid_contours = {}
        self.cooldown_frames = 10
        self.roi_cooldowns = {}
        
    def get_material_type(self, topic, roi_idx):
        if topic == "/arena_camera_node_0/image_raw":
            return "drywall"  # Both ROIs are drywall
        elif topic == "/arena_camera_node_2/image_raw":
            return "aggregate"  # Both ROIs are aggregate
        elif topic == "/arena_camera_node_3/image_raw":
            if roi_idx in [0, 1]:  # ROI 1 and 3 (0-based indexing)
                return "wood"
            else:  # ROI 2 and 4
                return "cardboard"
        return None

    def save_detection_frame(self, frame, contour_box, material_type, topic, roi_idx, timestamp, output_dir):
        """Save frame with highlighted detection when count increases."""
        save_frame = frame.copy()
        x, y, w, h = map(int, contour_box)
        cv2.rectangle(save_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        text = f"{material_type} - Bin {roi_idx + 1}"
        cv2.putText(save_frame, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        topic_dir = os.path.join(output_dir, topic.strip('/').replace('/', '_'))
        os.makedirs(topic_dir, exist_ok=True)
        
        filename = f"{topic.split('/')[-1]}_{timestamp}_{material_type}_bin{roi_idx+1}.jpg"
        save_path = os.path.join(topic_dir, filename)
        cv2.imwrite(save_path, save_frame)
        print(f"Saved detection image: {save_path}")

    def update_counters(self, material_type, topic, roi_idx, frame, contour_box, timestamp, output_dir):
        roi_key = f"{topic}_{roi_idx}"

        if roi_key in self.roi_cooldowns and self.roi_cooldowns[roi_key] > 0:
            return False

        if material_type:
            self.counters[material_type] += 1
            self.counters['total'] += 1
            self.roi_cooldowns[roi_key] = self.cooldown_frames
            
            # Save the detection frame
            self.save_detection_frame(frame, contour_box, material_type, topic, roi_idx, timestamp, output_dir)
            return True
            
        return False

    def detect_objects(self, frame, mask, bg_subtractor, person_boxes, roi_offset, topic, roi_idx, timestamp, output_dir):
        key = f"{topic}_{roi_idx}"

        if self.warmup_counters[key] < self.warmup_frames:
            blurred = cv2.GaussianBlur(frame, self.gaussian_kernel, 0)
            _ = bg_subtractor.apply(blurred, learningRate=self.learning_rate)
            self.warmup_counters[key] += 1
            return [], [], np.zeros_like(mask)  
        
        blurred = cv2.GaussianBlur(frame, self.gaussian_kernel, 0)
        
        fg_mask = bg_subtractor.apply(blurred, learningRate=self.learning_rate)
        fg_mask = cv2.bitwise_and(fg_mask, mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morphology_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morphology_kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        overlapping_contours = []
        x_offset, y_offset = roi_offset
        
        tracker_key = f"{topic}_{roi_idx}"
        if tracker_key not in self.previous_valid_contours:
            self.previous_valid_contours[tracker_key] = []
        
        current_valid_centers = []
        bin_key = f"{topic}_{roi_idx}"
        bin_can_count = bin_key not in self.roi_cooldowns or self.roi_cooldowns[bin_key] == 0
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = int(x + w/2), int(y + h/2)
            if not (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and mask[center_y, center_x] > 0):
                continue
            
            contour_box = [x + x_offset, y + y_offset, w, h]
            max_overlap = 0

            for person_box in person_boxes:
                overlap = self.calculate_person_overlap(contour_box, person_box)
                max_overlap = max(max_overlap, overlap)

            center = (center_x + x_offset, center_y + y_offset)
            is_new = True
            for prev_center in self.previous_valid_contours[tracker_key]:
                dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                if dist < 50:  
                    is_new = False
                    break

            if max_overlap > self.person_overlap_threshold:
                overlapping_contours.append(contour_box)
            else:
                valid_contours.append(contour_box)
                current_valid_centers.append(center)
                
                if is_new and bin_can_count:
                    material_type = self.get_material_type(topic, roi_idx)
                    if self.update_counters(material_type, topic, roi_idx, frame, contour_box, timestamp, output_dir):
                        bin_can_count = False  # Prevent additional counts for this bin this frame

        self.previous_valid_contours[tracker_key] = current_valid_centers
        self.update_cooldowns()
        
        return valid_contours, overlapping_contours, fg_mask

    def update_cooldowns(self):
        for roi_key in list(self.roi_cooldowns.keys()):
            if self.roi_cooldowns[roi_key] > 0:
                self.roi_cooldowns[roi_key] -= 1
                if self.roi_cooldowns[roi_key] == 0:
                    del self.roi_cooldowns[roi_key]


    def init_bg_subtractor(self, topic, roi_idx):
        key = f"{topic}_{roi_idx}"
        if key not in self.background_subtractors:
            self.background_subtractors[key] = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=self.var_threshold,
                detectShadows=False
            )
            self.warmup_counters[key] = 0
        return self.background_subtractors[key]
    
    
    def load_rois(self, json_path='bins_mb_2.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def get_person_boxes(self, frame):
        results = self.person_model(frame, conf=self.conf_threshold_person)
        person_boxes = []
        for det in results[0].boxes.data:
            x1, y1, x2, y2, conf, _ = map(float, det)
            width = x2 - x1
            reduction = width * 0.15
            x1 = x1 + reduction/2
            x2 = x2 - reduction/2
            person_boxes.append([x1, y1, x2, y2])
        return person_boxes

    def get_bin_crop(self, frame, polygon_points):
        points_array = np.array(polygon_points)
        x_min, y_min = np.min(points_array, axis=0)
        x_max, y_max = np.max(points_array, axis=0)
        
        cropped = frame[y_min:y_max, x_min:x_max]
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        shifted_points = points_array - [x_min, y_min]
        cv2.fillPoly(mask, [shifted_points.astype(int)], 255)
        
        masked_crop = cropped.copy()
        masked_crop[mask == 0] = 0
        
        return masked_crop, mask, (x_min, y_min, x_max, y_max)

    def calculate_person_overlap(self, contour_box, person_box):
        # Calculate overlap between contour and person box
        x1 = max(contour_box[0], person_box[0])
        y1 = max(contour_box[1], person_box[1])
        x2 = min(contour_box[0] + contour_box[2], person_box[2])
        y2 = min(contour_box[1] + contour_box[3], person_box[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        contour_area = contour_box[2] * contour_box[3]
        
        return intersection / contour_area if contour_area > 0 else 0
    
    def draw_counters(self, frame):
        # Define the starting position and formatting
        y_start = 400
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        cv2.putText(frame, f"Drywall: {self.counters['drywall']}", 
                   (500, y_start), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(frame, f"Aggregate: {self.counters['aggregate']}", 
                   (500, y_start + line_height), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(frame, f"Wood: {self.counters['wood']}", 
                   (500, y_start + 2 * line_height), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(frame, f"Cardboard: {self.counters['cardboard']}", 
                   (500, y_start + 3 * line_height), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(frame, f"Total: {self.counters['total']}", 
                   (500, y_start + 4 * line_height), font, font_scale, (0, 255, 255), font_thickness)
        
        
def main():
    detector = BinDetector()
    rois_data = detector.load_rois()
    
    topics = [
        "/arena_camera_node_0/image_raw",
        "/arena_camera_node_2/image_raw",
        "/arena_camera_node_3/image_raw"
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bin_detections_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    bag_file = "dataset.bag"
    print(f"Processing {bag_file}...")
    
    cv2.namedWindow("Detection Visualization", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Motion Debug", cv2.WINDOW_NORMAL)
    
    with rosbag.Bag(bag_file) as bag:
        for topic in topics:
            print(f"\nProcessing topic: {topic}")
            
            topic_dir = os.path.join(output_dir, topic.strip('/').replace('/', '_'))
            os.makedirs(topic_dir, exist_ok=True)
            
            topic_rois = [roi for roi in rois_data if roi["topic_name"] == topic]
            
            if not topic_rois:
                continue
            
            for _, msg, t in bag.read_messages(topics=[topic]):
                try:
                    
                    frame_timestamp = datetime.fromtimestamp(t.to_sec()).strftime("%Y%m%d_%H%M%S_%f")
                    
                    frame = detector.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    vis_frame = frame.copy()
                    
                    person_boxes = detector.get_person_boxes(frame)
                    for box in person_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    debug_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                    
                    for roi_data in topic_rois:
                        for roi_idx, (roi_type, roi_coords) in enumerate(roi_data["rois"]):
                            if roi_type == 'polygon':
                                cv2.polylines(vis_frame, [np.array(roi_coords, np.int32)], 
                                            True, (255, 255, 0), 2)
                                
                                material_type = detector.get_material_type(topic, roi_idx)
                                cv2.putText(vis_frame, f"{material_type} #{roi_idx+1}", 
                                          (roi_coords[0][0], roi_coords[0][1]-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                
                                bin_crop, mask, (x_min, y_min, x_max, y_max) = detector.get_bin_crop(
                                    frame, roi_coords
                                )
                                
                                bg_subtractor = detector.init_bg_subtractor(topic, roi_idx)
                                
                                valid_contours, overlapping_contours, fg_mask = detector.detect_objects(
                                    bin_crop, mask, bg_subtractor,
                                    person_boxes, (x_min, y_min),
                                    topic, roi_idx, frame_timestamp, output_dir
                                )

                                fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                                debug_region = debug_frame[y_min:y_max, x_min:x_max]
                                debug_region[mask > 0] = fg_mask_bgr[mask > 0]
                                
                                cv2.polylines(debug_frame, [np.array(roi_coords, np.int32)], 
                                            True, (0, 255, 255), 2)

                                for x, y, w, h in valid_contours:
                                    x1, y1 = int(x), int(y)
                                    x2, y2 = int(x + w), int(y + h)
                                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                for x, y, w, h in overlapping_contours:
                                    x1, y1 = int(x), int(y)
                                    x2, y2 = int(x + w), int(y + h)
                                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    
                    detector.draw_counters(debug_frame)
                    detector.draw_counters(vis_frame)
                    
                    cv2.imshow("Detection Visualization", vis_frame)
                    cv2.imshow("Motion Debug", debug_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Detections saved to {output_dir}")
    print("\nFinal Counts:")
    for material, count in detector.counters.items():
        print(f"{material.capitalize()}: {count}")

if __name__ == "__main__":
    main()