import argparse
from ultralytics import YOLO
from utils import visualize_results, prepare_data_yaml

def evaluate(model_path, data_path):
    # Prepare the data.yaml file
    prepare_data_yaml(data_path)
    
    model = YOLO(model_path)
    
    # Evaluate the model
    model.val(data=data_path, split="test")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 for vehicle damage detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, default='/app/data/data.yaml', help='Path to data.yaml file')
    args = parser.parse_args()

    evaluate(args.model, args.data)
