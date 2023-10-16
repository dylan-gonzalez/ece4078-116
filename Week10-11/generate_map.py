import os
import json
from collections import defaultdict

# Function to parse the "slam.txt" file
def parse_slam_file(file_path):
    with open(file_path, 'r') as file:
        slam_data = json.load(file)
    aruco_data = [
        {"tag": tag, "x": x, "y": y}
        for tag, x, y in zip(slam_data["taglist"], slam_data["map"][0], slam_data["map"][1])
    ]
    return aruco_data

# Function to parse the "targets.txt" file
def parse_targets_file(file_path):
    with open(file_path, 'r') as file:
        targets_data = json.load(file)
    return targets_data

# Function to merge and format the aruco and target data
def merge_and_format_data(aruco_data, targets_data):
    aruco_counter = defaultdict(int)
    merged_data = {}

    # Merging aruco data
    for item in aruco_data:
        tag = item["tag"]
        label = f"aruco{tag}_{aruco_counter[tag]}"
        aruco_counter[tag] += 1
        merged_data[label] = {"x": item["x"], "y": item["y"]}

    # Merging targets data
    for label, coordinates in targets_data.items():
        merged_data[label] = {"x": coordinates["x"], "y": coordinates["y"]}
    
    return merged_data

# Function to save the final data to a text file
def save_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Define the paths
folder_path = "lab_output"
slam_file_path = os.path.join(folder_path, "slam.txt")
targets_file_path = os.path.join(folder_path, "targets.txt")
output_file_path = "true_map_full.txt"

# Parse the files
aruco_data = parse_slam_file(slam_file_path)
targets_data = parse_targets_file(targets_file_path)

# Merge and format the data
final_data = merge_and_format_data(aruco_data, targets_data)

# Save the final data to a text file
save_to_file(final_data, output_file_path)

# Print the path to the output file
output_file_path

