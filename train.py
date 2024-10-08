import argparse
from ultralytics import YOLO
from utils import visualize_results, prepare_data_yaml

def train(data_path, epochs, batch_size, img_size):
    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        save=True
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for vehicle damage detection')
    parser.add_argument('--data', type=str, default='/app/data/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.img_size)