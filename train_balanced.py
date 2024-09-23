import argparse
from ultralytics import YOLO
from utils import visualize_results, prepare_data_yaml_balanced, create_temp_data_yaml, copy_filtered_data_to_temp
import os
import shutil


def train(data_path, epochs, batch_size, img_size):
    # Prepare the data by limiting each class to a maximum of 100 images
    filtered_train_images, val_images = prepare_data_yaml_balanced(data_path)

    # Copy filtered images and labels to a temporary directory
    temp_image_dir, temp_label_dir = copy_filtered_data_to_temp(filtered_train_images)

    # Create a temporary data.yaml with the new training directory
    temp_data_yaml_path = create_temp_data_yaml(temp_image_dir, val_images)

    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model

    # Train the model using the temporary data.yaml
    results = model.train(
        data=temp_data_yaml_path,  # Pass the temporary data.yaml path
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        save=True
    )

    # Clean up the temporary data.yaml file after training
    if os.path.exists(temp_data_yaml_path):
        os.remove(temp_data_yaml_path)

    # Optionally, clean up the temporary directory after training
    if os.path.exists(temp_image_dir):
        shutil.rmtree(os.path.dirname(temp_image_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for vehicle damage detection')
    parser.add_argument('--data', type=str, default='/app/data/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.img_size)