import json

# Specify the file paths for slam.txt and targets.txt
slam_file_path = 'lab_output/slam.txt'
targets_file_path = 'lab_output/targets.txt'

# Read the contents of slam.txt and targets.txt
with open(slam_file_path, 'r') as slam_file, open(targets_file_path, 'r') as targets_file:
    slam_data = json.load(slam_file)
    targets_data = json.load(targets_file)

# Extract the taglist from slam_data
taglist = slam_data["taglist"]

# Initialize a dictionary to store the merged data
merged_data = {}

# Extract the map data from slam_data and merge it with targets_data
map_data = slam_data["map"]
for i, tag in enumerate(taglist):
    marker_key = f"aruco{tag}_0"
    marker_data = {"y": map_data[0][i], "x": map_data[1][i]}
    merged_data[marker_key] = marker_data

# Merge the merged_data with targets_data
merged_data.update(targets_data)

# Convert the merged data back to JSON format
merged_json = json.dumps(merged_data, indent=2)

# Write the merged data to an output file
output_file_path = 'merged_map.txt'
with open(output_file_path, 'w') as output_file:
    output_file.write(merged_json)

print("Merged data has been written to 'output.txt'")
