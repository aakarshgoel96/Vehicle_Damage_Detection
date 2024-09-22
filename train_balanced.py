import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from utils import visualize_results, prepare_data_yaml

def train(data_path, epochs, batch_size, img_size, use_focal_loss):
    # Set device to GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model
    model = model.to(device)  # Move model to the appropriate device
    
    # Adjust class weights: more weight to under-represented classes
    class_weights = torch.tensor([10.0, 1.0, 15.0, 10.0, 5.0, 15.0, 10.0, 5.0], device=device)  # Example weights

    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use weighted loss if not focal loss
    
    # Train the model with the modified loss function and augmentations
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        save=True,
        augment=True,  # Enable data augmentations (you can customize this)
        mosaic=True,   # Advanced augmentation (merges 4 images)
        cutmix=True,   # Another augmentation technique to blend images
        loss=criterion  # Pass the custom loss function (either Focal Loss or weighted CrossEntropy)
    )
    
    # Visualize training results
    visualize_results(results)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for vehicle damage detection')
    parser.add_argument('--data', type=str, default='/app/data/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.img_size)