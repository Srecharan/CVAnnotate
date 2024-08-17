import cv2
import rosbag
from cv_bridge import CvBridge
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch
import os
from pathlib import Path

class ROIMaskSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.scale = 0.6
        self.roi_points = None
        self.frame = None
        self.points = []
        self.frame_counter = 0
        self.save_dir = "trash"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load YOLO model
        model_path = Path('SAM_best.pt')
        if not model_path.exists():
            print(f"Error: Model file not found at {model_path}")
            model_path = Path(input("Please enter the correct path to SAM_best.pt: "))
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
        
        print("Loading YOLO model...")
        self.model = YOLO(str(model_path)).cuda()
        print("Model loaded successfully!")

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                if len(self.points) > 1:
                    cv2.line(self.frame, self.points[-2], self.points[-1], (0, 255, 0), 2)
                if len(self.points) == 4:
                    cv2.line(self.frame, self.points[-1], self.points[0], (0, 255, 0), 2)
                    cv2.imshow('Select ROI Corners', self.frame)
                print(f"[DEBUG] Point added: {x}, {y}. Total points: {len(self.points)}")

    def select_roi_region(self, first_frame):
        """GUI for selecting ROI corners."""
        print("\nSelect the four corners of the ROI:")
        print("- Click 4 points in clockwise order starting from top-left")
        print("- Press 'r' to reset points")
        print("- Press 'c' to confirm selection")
        print("- Press 'q' to quit")
        
        self.frame = first_frame.copy()
        self.points = []
        
        cv2.namedWindow('Select ROI Corners')
        cv2.setMouseCallback('Select ROI Corners', self.click_event)
        
        while True:
            cv2.imshow('Select ROI Corners', self.frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                print("[DEBUG] Resetting points")
                self.frame = first_frame.copy()
                self.points = []
            elif key == ord('c'):
                print(f"[DEBUG] 'c' pressed, points collected: {len(self.points)}")
                if len(self.points) == 4:
                    print("[DEBUG] Four points confirmed, proceeding with ROI")
                    cv2.destroyWindow('Select ROI Corners')
                    return np.array(self.points)
                else:
                    print(f"[DEBUG] Need 4 points, but only {len(self.points)} selected. Keep clicking points.")
            elif key == ord('q'):
                print("[DEBUG] ROI selection cancelled")
                cv2.destroyWindow('Select ROI Corners')
                return None

    def process_image(self, image):
        try:
            print("[DEBUG] Running model inference")
            results = self.model(image, show_boxes=False)
            if results and results[0].masks:
                print(f"[DEBUG] Model detected {len(results[0].masks.data)} objects")
                return results[0].masks.data
            print("[DEBUG] No masks detected in this frame")
            return None
        except Exception as e:
            print(f"Error in process_image: {e}")
            return None

    def save_individual_masks(self, original_image, masks, frame_num):
        if self.roi_points is None or masks is None:
            return
        
        try:
            roi_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [self.roi_points], 1)
            
            saved_count = 0
            for idx, mask_tensor in enumerate(masks):
                mask = mask_tensor.cpu().numpy().astype(np.uint8)
                if mask.shape != (original_image.shape[0], original_image.shape[1]):
                    mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
                
                masked_result = cv2.bitwise_and(mask, roi_mask)

                if not np.any(masked_result):
                    continue
                
                coords = cv2.findNonZero(masked_result)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    cropped_mask = masked_result[y:y+h, x:x+w]
                    cropped_image = original_image[y:y+h, x:x+w]
                    output = np.zeros_like(cropped_image)
                    output[cropped_mask > 0] = cropped_image[cropped_mask > 0]
                    
                    save_path = os.path.join(self.save_dir, f'mask_{frame_num}_{idx}.png')
                    cv2.imwrite(save_path, output)
                    saved_count += 1
            
            if saved_count > 0:
                print(f"[DEBUG] Saved {saved_count} masks for frame {frame_num}")
        
        except Exception as e:
            print(f"Error saving masks: {e}")

    def process_bag(self, bag_file, topic_name):
        try:
            print(f"[DEBUG] Opening bag file: {bag_file}")
            bag = rosbag.Bag(bag_file)
            print("[DEBUG] Successfully opened bag file")
            
            print("[DEBUG] Getting first message")
            msg = next(bag.read_messages(topics=[topic_name]))
            print("[DEBUG] Converting first message to CV2")
            first_frame = self.bridge.imgmsg_to_cv2(msg.message, desired_encoding='bgr8')
            print("[DEBUG] First frame conversion successful")
            
            self.roi_points = self.select_roi_region(first_frame)
            if self.roi_points is None:
                print("[DEBUG] ROI selection cancelled")
                return
            print(f"[DEBUG] ROI points selected: {self.roi_points}")
            
            bag.close()
            print("[DEBUG] Reopening bag for processing")
            bag = rosbag.Bag(bag_file)
            
            message_count = bag.get_message_count(topic_filters=[topic_name])
            print(f"[DEBUG] Found {message_count} messages to process")
            
            for _, msg, _ in bag.read_messages(topics=[topic_name]):
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    
                    # Process image every 5 frames
                    if self.frame_counter % 5 == 0:
                        print(f"[DEBUG] Processing frame {self.frame_counter}")
                        masks = self.process_image(cv_image)
                        if masks is not None:
                            self.save_individual_masks(cv_image, masks, self.frame_counter)
                            
                            # Visualize masks
                            display_img = cv_image.copy()
                            combined_mask = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8)
                            if combined_mask.shape != (display_img.shape[0], display_img.shape[1]):
                                combined_mask = cv2.resize(combined_mask, (display_img.shape[1], display_img.shape[0]))
                            mask_overlay = np.zeros_like(display_img)
                            mask_overlay[combined_mask > 0] = [0, 0, 255]  # Red overlay
                            display_img = cv2.addWeighted(display_img, 1, mask_overlay, 0.3, 0)
                            cv2.polylines(display_img, [self.roi_points], True, (0, 255, 0), 2)
                            display_img_resized = cv2.resize(display_img, (0, 0), fx=self.scale, fy=self.scale)
                            cv2.imshow('Processing...', display_img_resized)
                            cv2.waitKey(1)
                    
                    if cv2.waitKey(1) == 27:  # ESC key
                        print("[DEBUG] ESC pressed, stopping")
                        break
                    
                    self.frame_counter += 1
                    
                except Exception as e:
                    print(f"[DEBUG] Error processing frame {self.frame_counter}: {e}")
                    continue
                
        except Exception as e:
            print(f"[DEBUG] Error during processing: {e}")
        finally:
            cv2.destroyAllWindows()
            bag.close()
            print("[DEBUG] Processing complete")

def main():
    processor = ROIMaskSaver()
    bag_file = 'dataset.bag'
    topic_name = '/arena_camera_node_3/image_raw'
    
    if not os.path.exists(bag_file):
        print(f"Error: Bag file not found at {bag_file}")
        return
    
    processor.process_bag(bag_file, topic_name)
    print(f"\nProcessing complete. Individual masked images saved in '{processor.save_dir}' directory.")

if __name__ == "__main__":
    main()