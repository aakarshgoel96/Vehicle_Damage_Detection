import yaml
from ultralytics.yolo.utils.plotting import plot_results
import matplotlib.pyplot as plt
import os

def visualize_results(results):
    fig, ax = plot_results(results)
    plt.savefig('results.png')
    plt.close()

def prepare_data_yaml(data_yaml_path):
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)