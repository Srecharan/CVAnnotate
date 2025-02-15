#Gets screenshots of the bins from the bag file
#Currently uses bins.json to get the bins
#To get the bins.json file, run the get_coordinates.py code.


import json
import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
import random
import os
from datetime import datetime

def load_bins(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_roi(image, roi):
    shape_type, coords = roi
    if shape_type == "rectangle":
        x, y, w, h = coords["x"], coords["y"], coords["w"], coords["h"]
        return image[y:y+h, x:x+w]
    return None

def process_bag(bag_path, bins, num_frames=10):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bin_screenshots_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    
    topic_bins = {}
    for bin_info in bins:
        topic = bin_info["topic_name"]
        if topic not in topic_bins:
            topic_bins[topic] = []
        topic_bins[topic].append(bin_info)

    bridge = CvBridge()
    
    msg_counts = {}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic in topic_bins.keys():
            msg_counts[topic] = bag.get_message_count(topic)

    frame_indices = {}
    for topic, count in msg_counts.items():
        indices = sorted(random.sample(range(count), min(num_frames, count)))
        frame_indices[topic] = indices

    with rosbag.Bag(bag_path, 'r') as bag:
        topic_counters = {topic: 0 for topic in topic_bins.keys()}
        frame_counters = {topic: 0 for topic in topic_bins.keys()}
        
        for topic, msg, t in bag.read_messages(topics=list(topic_bins.keys())):
            if topic_counters[topic] in frame_indices[topic]:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                for bin_idx, bin_info in enumerate(topic_bins[topic]):
                    roi = bin_info["rois"][0]  # Assuming single ROI per bin
                    roi_image = extract_roi(cv_image, roi)
                    
                    if roi_image is not None:
                        frame_num = frame_counters[topic]
                        filename = f"{output_dir}/frame_{frame_num}_topic_{topic.split('/')[-2]}_bin_{bin_idx}.png"
                        cv2.imwrite(filename, roi_image)
                
                frame_counters[topic] += 1
                
                if all(counter >= num_frames for counter in frame_counters.values()):
                    break
                    
            topic_counters[topic] += 1

if __name__ == "__main__":
    bins = load_bins("bins.json")
    
    bag_path = "dataset.bag" 
    process_bag(bag_path, bins, num_frames=10)