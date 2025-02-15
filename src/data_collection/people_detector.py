# Save people masks from rosbag
# Uses color based detection to detect people and saves images with bounding boxes

import cv2
import rosbag
from cv_bridge import CvBridge
import numpy as np
import os
import time

dirs = ['person/images', 'person/labels', 'person/debug']
for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)  

bridge = CvBridge()

# Used hsv values 
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

def color_detection(roi_img):
    """Detect yellow objects in the image"""
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = [(x + w//2, y + h//2) for c in contours if cv2.contourArea(c) > 200 
                 for x, y, w, h in [cv2.boundingRect(c)]]
    return detections, mask

def group_detections(detections, max_distance):
    """Group nearby detections together"""
    grouped = []
    detections = detections.copy()
    while detections:
        base = detections.pop(0)
        group = [base]
        i = 0
        while i < len(detections):
            if np.linalg.norm(np.array(base) - np.array(detections[i])) < max_distance:
                group.append(detections.pop(i))
            else:
                i += 1
        grouped.append(group)
    return [group for group in grouped if len(group) >= 2]

def get_bounding_box(points, padding=150, frame_shape=None):
    """Get bounding box coordinates"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    min_x = min(x_coords) - padding
    min_y = min(y_coords) - padding
    max_x = max(x_coords) + padding
    max_y = max(y_coords) + padding
    
    if frame_shape is not None:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame_shape[1], max_x)
        max_y = min(frame_shape[0], max_y)
    
    return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

def convert_to_yolo_format(box, img_width, img_height):
    """Convert bounding box to YOLO format"""
    x, y, w, h = box
    return [
        (x + w/2) / img_width,  
        (y + h/2) / img_height, 
        w / img_width,         
        h / img_height          
    ]

def process_frame(frame, frame_num, topic_num):
    """Process single frame and save results"""
    try:
        yellow_detections, yellow_mask = color_detection(frame)

        debug_image = frame.copy()

        frame_name = f'{frame_num}_top{topic_num}'
        cv2.imwrite(f'person/images/{frame_name}.jpg', frame)
        
        if yellow_detections:
            person_groups = group_detections(yellow_detections, 200)
            yolo_annotations = []
            
            for group in person_groups:
                box = get_bounding_box(group, frame_shape=frame.shape)
                x, y, w, h = box
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

                label = f"person {len(group)}"
                cv2.putText(debug_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                yolo_box = convert_to_yolo_format(box, frame.shape[1], frame.shape[0])
                yolo_annotations.append(f'0 {" ".join([str(x) for x in yolo_box])}')

            if yolo_annotations:
                with open(f'person/labels/{frame_name}.txt', 'w') as f:
                    f.write('\n'.join(yolo_annotations))
        else:
            with open(f'person/labels/{frame_name}.txt', 'w') as f:
                pass

        cv2.imwrite(f'person/debug/{frame_name}_debug.jpg', debug_image)
        return True
        
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return False

print("Starting ROS bag processing...")
bag_file = input("Enter the bag file name (without extension): ") + '.bag'
bag = rosbag.Bag(bag_file)

topics = [
    '/arena_camera_node_0/image_raw',
    '/arena_camera_node_2/image_raw',
    '/arena_camera_node_3/image_raw'
]

for topic_idx, topic in enumerate(topics, 1):
    print(f"\nProcessing topic {topic_idx}: {topic}")
    save_count = 1  
    frame_count = 0
    
    try:
        total_messages = bag.get_message_count(topic_filters=[topic])
        print(f"Total frames in topic: {total_messages}")
        
        for _, msg, _ in bag.read_messages(topics=[topic]):
            if frame_count % 5 == 0:  
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
                    if process_frame(cv_image, save_count, topic_idx):
                        if save_count % 20 == 0:
                            print(f"Saved {save_count} frames from topic {topic_idx}")
                        save_count += 1
                except Exception as e:
                    print(f"Error converting frame {frame_count} from topic {topic_idx}: {e}")
            
            frame_count += 1
            
        print(f"Completed topic {topic_idx}: Processed {frame_count} frames, Saved {save_count-1} frames")
        
    except Exception as e:
        print(f"Error processing topic {topic_idx}: {e}")

bag.close()
print("\nProcessing complete! Dataset saved in person directory")