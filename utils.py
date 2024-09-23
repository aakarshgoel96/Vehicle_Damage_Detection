import yaml
from ultralytics.yolo.utils.plotting import plot_results
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict
import shutil



def prepare_data_yaml(data_yaml_path):
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)


def prepare_data_yaml_balanced(data_yaml_path, max_images_per_class=100):
    root_dir = os.path.dirname(data_yaml_path)  # Get the root directory (app/data)
    
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Load the relative training and validation file paths and append to root_dir
    train_path = os.path.join(root_dir, data['train'])
    val_path = os.path.join(root_dir, data['val'])

    # Parse the class names from the data.yaml file
    class_names = data['names']

    # Dictionary to store images by class
    class_images = defaultdict(list)

    # Read the training dataset
    for root, _, files in os.walk(train_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                # Assuming the label is stored in a corresponding .txt file (YOLO format)
                label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')

                if os.path.exists(label_path):
                    with open(label_path, 'r') as label_file:
                        labels = label_file.readlines()
                        for label in labels:
                            class_id = int(label.split()[0])  # Get the class ID from the label file
                            class_name = class_names[class_id]
                            class_images[class_name].append((image_path, label_path))  # Store image and label together

    # Limit each class to a maximum of 100 images
    filtered_train_images = []
    for class_name, images in class_images.items():
        if len(images) > max_images_per_class:
            filtered_train_images.extend(random.sample(images, max_images_per_class))
        else:
            filtered_train_images.extend(images)

    # Return the filtered image-label pairs and the validation path
    return filtered_train_images, val_path


def copy_filtered_data_to_temp(filtered_train_images, temp_dir='/app/data/temp_train'):
    # Create a temporary directory for filtered images and labels
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)  # Clean up if it exists from a previous run
    os.makedirs(temp_dir, exist_ok=True)

    temp_image_dir = os.path.join(temp_dir, 'images')
    temp_label_dir = os.path.join(temp_dir, 'labels')

    os.makedirs(temp_image_dir, exist_ok=True)
    os.makedirs(temp_label_dir, exist_ok=True)

    # Copy images and labels to the temporary directory
    for image_path, label_path in filtered_train_images:
        # Copy image
        image_name = os.path.basename(image_path)
        temp_image_path = os.path.join(temp_image_dir, image_name)
        shutil.copy(image_path, temp_image_path)

        # Copy label
        label_name = os.path.basename(label_path)
        temp_label_path = os.path.join(temp_label_dir, label_name)
        shutil.copy(label_path, temp_label_path)

    return temp_image_dir, temp_label_dir


def create_temp_data_yaml(temp_image_dir, val_path, output_path='/app/data/temp_data.yaml'):
    # Create a new data.yaml for the filtered dataset
    data = {
        'train': temp_image_dir,  # Path to the temporary directory with filtered training images
        'val': val_path,  # Validation path remains the same
        'nc': 8,  # Number of classes
        'names': ['minor-dent', 'minor-scratch', 'moderate-broken', 'moderate-dent', 'moderate-scratch', 'severe-broken', 'severe-dent', 'severe-scratch']
    }

    with open(output_path, 'w') as file:
        yaml.dump(data, file)

    return output_path