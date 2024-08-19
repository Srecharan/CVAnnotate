# Code to create a dual dataset for object detection and segmentation.
# Currently if training only use object detcetion , we dont need segmentation.
# The dataset is created by overlaying objects on top of background images.
# The objects are resized and scaled to fit the background image.
# The dataset is split into training and validation sets.
# The dataset is saved in the 'Augmented_trash' and 'Augmented_trash_seg' directories.
# The dataset.yaml file is created in each directory to specify the dataset for YOLO training.

import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

def apply_augmentations(image, mask):
    """Apply random augmentations to the image and mask"""
    
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        mask = cv2.warpAffine(mask, rotation_matrix, (width, height))

    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-30, 30)    # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    
    return image, mask

def calculate_scale_factor(obj_size, bg_size, min_scale=0.1, max_scale=0.8):
    """Calculate appropriate scale factor to fit object in background"""
    width_scale = bg_size[0] / obj_size[0] * max_scale
    height_scale = bg_size[1] / obj_size[1] * max_scale
    scale = min(width_scale, height_scale)
    return max(min_scale, min(scale, max_scale))

def create_dual_dataset(objects_dir, backgrounds_dir, output_dir_det, output_dir_seg, split_ratio=0.8, augmentations_per_image=2):
    """Create both detection and segmentation datasets"""
    
    for output_dir in [output_dir_det, output_dir_seg]:
        output_dir = Path(output_dir)
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    
    object_files = list(Path(objects_dir).glob("*.[pj][np][g]"))
    background_files = list(Path(backgrounds_dir).glob("*.[pj][np][g]"))
    
    if not object_files or not background_files:
        raise ValueError("No images found in one or both directories")
    
    print(f"Found {len(object_files)} objects and {len(background_files)} backgrounds")
    
    
    backgrounds = []
    for bg_path in background_files:
        bg_img = cv2.imread(str(bg_path))
        if bg_img is not None:
            backgrounds.append((bg_img, bg_img.shape[:2]))
    
    if not backgrounds:
        raise ValueError("No valid background images found")
    
    total_images = len(object_files) * (augmentations_per_image + 1)
    pbar = tqdm(total=total_images, desc="Processing images")
    
    skipped_images = 0
    processed_images = 0
    
    
    for obj_path in object_files:
        
        obj_img = cv2.imread(str(obj_path))
        if obj_img is None:
            print(f"Failed to read {obj_path}")
            skipped_images += 1
            pbar.update(augmentations_per_image + 1)
            continue
        
        obj_height, obj_width = obj_img.shape[:2]
        if obj_height == 0 or obj_width == 0:
            print(f"Invalid image dimensions for {obj_path}")
            skipped_images += 1
            pbar.update(augmentations_per_image + 1)
            continue
        
        
        gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        for aug_idx in range(augmentations_per_image + 1):
            try:
                bg_img, bg_shape = random.choice(backgrounds)
                bg_height, bg_width = bg_shape
                
                if aug_idx > 0:
                    curr_obj_img, curr_mask = apply_augmentations(obj_img.copy(), mask.copy())
                else:
                    curr_obj_img, curr_mask = obj_img.copy(), mask.copy()

                scale = calculate_scale_factor(
                    (obj_width, obj_height),
                    (bg_width, bg_height)
                )

                scale *= random.uniform(0.8, 1.0)
                new_obj_width = max(10, int(obj_width * scale))
                new_obj_height = max(10, int(obj_height * scale))

                obj_img_resized = cv2.resize(curr_obj_img, (new_obj_width, new_obj_height))
                mask_resized = cv2.resize(curr_mask, (new_obj_width, new_obj_height))

                x_offset = random.randint(0, max(0, bg_width - new_obj_width))
                y_offset = random.randint(0, max(0, bg_height - new_obj_height))
                
                
                composite = bg_img.copy()
                roi = composite[y_offset:y_offset+new_obj_height, x_offset:x_offset+new_obj_width]
                
                mask_inv = cv2.bitwise_not(mask_resized)
                bg_roi = cv2.bitwise_and(roi, roi, mask=mask_inv)
                fg = cv2.bitwise_and(obj_img_resized, obj_img_resized, mask=mask_resized)
                dst = cv2.add(bg_roi, fg)
                composite[y_offset:y_offset+new_obj_height, x_offset:x_offset+new_obj_width] = dst
                
                seg_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
                seg_mask[y_offset:y_offset+new_obj_height, x_offset:x_offset+new_obj_width] = mask_resized
                
                
                x_center = (x_offset + new_obj_width/2) / bg_width
                y_center = (y_offset + new_obj_height/2) / bg_height
                width = new_obj_width / bg_width
                height = new_obj_height / bg_height
                
               
                is_train = random.random() < split_ratio
                subset = "train" if is_train else "val"
                
                
                output_name = f"{obj_path.stem}_aug{aug_idx}" if aug_idx > 0 else obj_path.stem
                
                
                image_path_det = Path(output_dir_det) / "images" / subset / f"{output_name}.jpg"
                cv2.imwrite(str(image_path_det), composite)
                
                label_path_det = Path(output_dir_det) / "labels" / subset / f"{output_name}.txt"
                with open(label_path_det, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
                
                
                image_path_seg = Path(output_dir_seg) / "images" / subset / f"{output_name}.jpg"
                cv2.imwrite(str(image_path_seg), composite)
                
            
                mask_path_seg = Path(output_dir_seg) / "labels" / subset / f"{output_name}.png"
                cv2.imwrite(str(mask_path_seg), seg_mask)
                
                processed_images += 1
                
            except Exception as e:
                print(f"Error processing {obj_path} (aug {aug_idx}): {str(e)}")
                skipped_images += 1
            
            pbar.update(1)
    
    pbar.close()

    for output_dir, task_type in [(output_dir_det, 'detection'), (output_dir_seg, 'segment')]:
        yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

nc: 1  # number of classes
names: ['object']  # class names

task: {task_type} 
        """
        
        with open(Path(output_dir) / "dataset.yaml", 'w') as f:
            f.write(yaml_content.strip())
    
    print(f"\nDataset creation completed:")
    print(f"- Processed images: {processed_images}")
    print(f"- Skipped images: {skipped_images}")
    print(f"- Total attempts: {total_images}")
    print(f"\nDetection dataset saved in {output_dir_det}")
    print(f"Segmentation dataset saved in {output_dir_seg}")
    print("Use dataset.yaml in each directory for training")

if __name__ == "__main__":
    create_dual_dataset(
        objects_dir="trash",
        backgrounds_dir="bins",
        output_dir_det="Augmented_trash",
        output_dir_seg="Augmented_trash_seg",
        augmentations_per_image=2
    )