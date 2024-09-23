import json
import os
from collections import defaultdict
import yaml

def load_coco_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
        
def convert_coco_to_yolo(coco_data, output_dir, dataset_type):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a mapping of image id to file name
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # Create a mapping of category id to category name
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Group annotations by image id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Convert annotations to YOLO format
    for img_id, anns in annotations_by_image.items():
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_width, img_height = img_info['width'], img_info['height']
        
        yolo_anns = []
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = category_id_to_name[cat_id]
            if cat_name == 'severity-damage':
                continue  # Skip the superclass
            
            x, y, w, h = ann['bbox']
            # Convert to YOLO format (centerx, centery, width, height)
            yolo_x = (x + w / 2) / img_width
            yolo_y = (y + h / 2) / img_height
            yolo_w = w / img_width
            yolo_h = h / img_height
            
            # YOLO format uses 0-based index for class id
            yolo_cat_id = cat_id - 1  # Assuming category ids start from 1
            yolo_anns.append(f"{yolo_cat_id} {yolo_x} {yolo_y} {yolo_w} {yolo_h}")

        # Write YOLO annotations to file
        img_name = image_id_to_name[img_id]
        base_name = os.path.splitext(img_name)[0]
        with open(os.path.join(output_dir, f"{base_name}.txt"), 'w') as f:
            f.write("\n".join(yolo_anns))

    print(f"Converted {dataset_type} annotations to YOLO format in {output_dir}")


def generate_yaml(train_data,train_dir, test_dir,val_dir, output_file):
    # Get class names (excluding 'severity-damage')
    class_names = [cat['name'] for cat in train_data['categories'] if cat['name'] != 'severity-damage']

    yaml_content = {
        'train': train_dir,
        'val': val_dir,  # Using test set as validation
        'test': test_dir,
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Generated YAML file: {output_file}")

# Usage
train_json = './annotations/instances_train.json'
val_json = './annotations/instances_val.json'
test_json = './annotations/instances_test.json'
train_output_dir = './labels/train'
val_output_dir = './labels/val'
test_output_dir = './labels/test'
yaml_output = './data.yaml'

train_data = load_coco_data(train_json)
val_data = load_coco_data(val_json)
test_data = load_coco_data(test_json)

# Convert train and test datasets
convert_coco_to_yolo(train_data, train_output_dir, 'train')
convert_coco_to_yolo(val_data, val_output_dir, 'val')
convert_coco_to_yolo(test_data, test_output_dir, 'test')

# Generate YAML file
generate_yaml(train_data,train_output_dir, test_output_dir,val_output_dir, yaml_output)