#script to draw boxes on the imaege (for bins and other stuff) 
#saves coordinates in json file
#p for polygon, r for rectangle, q to quit , enter to confirm after drawing 

import cv2
import rosbag
from cv_bridge import CvBridge
import numpy as np
import json
import os

scale = 0.7
bridge = CvBridge()

def select_rois(image, scale_factor=scale):
    image_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    rois = []
    
    while True:
        clone = image_resized.copy()
        cv2.putText(clone, "Press 'p' for polygon, 'c' for circle, 'r' for rectangle, 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Select ROI", clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            roi = cv2.selectROI("Select Circle ROI", image_resized, fromCenter=False, showCrosshair=True)
            if roi != (0, 0, 0, 0):
                center = (int((roi[0] + roi[2] // 2) / scale_factor), int((roi[1] + roi[3] // 2) / scale_factor))
                radius = int(min(roi[2], roi[3]) // 2 / scale_factor)
                rois.append(('circle', {'center': center, 'radius': radius}))
        elif key == ord('r'):
            roi = cv2.selectROI("Select Rectangle ROI", image_resized, fromCenter=False, showCrosshair=True)
            if roi != (0, 0, 0, 0):
                x, y, w, h = [int(v / scale_factor) for v in roi]
                rois.append(('rectangle', {'x': x, 'y': y, 'w': w, 'h': h}))
        elif key == ord('p'):
            points = []
            clone = image_resized.copy()
            cv2.putText(clone, "Click to add points. Press 'Enter' to finish.", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
                    if len(points) > 1:
                        cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                    cv2.imshow("Select Polygon Points", clone)

            cv2.imshow("Select Polygon Points", clone)
            cv2.setMouseCallback("Select Polygon Points", click_event)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13 and len(points) >= 3:  # Enter key
                    cv2.line(clone, points[-1], points[0], (0, 255, 0), 2)
                    cv2.imshow("Select Polygon Points", clone)
                    cv2.waitKey(500)
                    break
            
            cv2.destroyWindow("Select Polygon Points")
            
            if len(points) >= 3:
                scaled_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in points]
                rois.append(('polygon', scaled_points))

    cv2.destroyAllWindows()
    return rois

file_input = input("Enter the bag file name: ")
bag_file = file_input + '.bag'
bag = rosbag.Bag(bag_file)

cv_image = None
topic_input = input("Enter the topic name, (eg. /arena_camera_node_x/image_raw): ")

place_input = input("Enter the place (mb or cc): ")

for topic, msg, t in bag.read_messages(topics=[topic_input]):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if topic_input.endswith('/compressed'):
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        break 
    except Exception as e:
        print(f"Error processing image: {e}")
        continue

if cv_image is not None:
    rois = select_rois(cv_image)
    
    roi_data = {
        "topic_name": topic_input,
        "place": place_input,
        "rois": rois
    }
        
    print("Selected ROIs:")
    for roi_type, roi in rois:
        if roi_type == 'circle':
            print(f"Circle ROI: center={roi['center']}, radius={roi['radius']}")
        elif roi_type == 'rectangle':
            print(f"Rectangle ROI: x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}")
        elif roi_type == 'polygon':
            print(f"Polygon ROI points: {roi}")

    for roi_type, roi in rois:
        if roi_type == 'circle':
            center, radius = roi['center'], roi['radius']
            cv2.circle(cv_image, center, radius, (255, 0, 0), 2) 
        elif roi_type == 'rectangle':
            x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif roi_type == 'polygon':
            points = np.array(roi, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(cv_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            
    cv_image_resized = cv2.resize(cv_image, (0, 0), fx=scale, fy=scale)
    
    cv2.imshow("Final Image with ROIs", cv_image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    json_file = 'bins.json'
    existing_data = []

    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

    if not isinstance(existing_data, list):
        existing_data = [existing_data]

    existing_data.append(roi_data)

    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=4)  # Use indent for better readability

else:
    print("No image found in the specified topic.")

bag.close()
