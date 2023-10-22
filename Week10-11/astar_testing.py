#from astar_class3 import AStar
from astar_class4 import CompleteImprovedAStar
import numpy as np
import json
'''
def get_sequential_waypoints_inclusive(true_map_filepath='true_map_full.txt', search_list_filepath='search_list.txt', grid_size=0.05, obstacle_size=0.2, boundary_size=3):
    """
    Get all waypoints for each target in the search list from the starting position.

    Parameters:
    - true_map_filepath (str): The file path to the true map file.
    - search_list_filepath (str): The file path to the search list file.
    - grid_size (float, optional): The size of each grid cell. Defaults to 0.05.
    - obstacle_size (float, optional): The size of the obstacles. Defaults to 0.2.
    - boundary_size (float, optional): The size of the boundary. Defaults to 3.

    Returns:
    - List of lists of waypoints for each target in meters.
    """
    astar = AStar(true_map_filepath, search_list_filepath, grid_size, obstacle_size, boundary_size)
    all_waypoints = []

    # Start from the origin for the first target
    current_start = (0, 0) #Ideally would use the robot pose for this rather than making it fixed also i should get the path sequentially 
    #not all at once

    for target in astar.targets:
        astar.find_path(current_start, target)  # Use the current start point
        waypoints = [(round(x * grid_size - boundary_size / 2, 2), round(y * grid_size - boundary_size / 2, 2)) for x, y in astar.path]
        all_waypoints.append(waypoints)
        print(f"Waypoints for {target}:", waypoints)  # Print waypoints for diagnosis

        # Update the current start to the last waypoint for the next target
        if waypoints:
            current_start = waypoints[-2]  
        else:
            print(f"Failed to find path to {target}.")
            
        astar.visualize_path()

    return all_waypoints

# Testing the updated function with sequential waypoints with the file paths
all_waypoints_inclusive = get_sequential_waypoints_inclusive()
print(all_waypoints_inclusive)  # Display the waypoints for each target
'''
'''
# The get_all_waypoints function
def get_all_waypoints(true_map_filepath='true_map_full.txt', search_list_filepath='search_list.txt', grid_size=0.05, obstacle_size=0.2, boundary_size=3):
    """
    Get all waypoints for each target in the search list from the starting position.

    Parameters:
    - true_map_filepath (str): The file path to the true map file.
    - search_list_filepath (str): The file path to the search list file.
    - grid_size (float, optional): The size of each grid cell. Defaults to 0.05.
    - obstacle_size (float, optional): The size of the obstacles. Defaults to 0.2.
    - boundary_size (float, optional): The size of the boundary. Defaults to 3.

    Returns:
    - List of lists of waypoints for each target in meters.
    """
    astar = AStar(true_map_filepath, search_list_filepath, grid_size, obstacle_size, boundary_size)
    all_waypoints = []

    for target in astar.targets:
        astar.find_path((0, 0), target)
        waypoints = [(round(x * grid_size - boundary_size / 2, 2), round(y * grid_size - boundary_size / 2, 2)) for x, y in astar.path]
        all_waypoints.append(waypoints)
        
        astar.visualize_path()

    return all_waypoints

# Test the function with the same start and target as before
# Correcting the file paths before testing the function
true_map_filepath = 'M4_true_map_full.txt'
search_list_filepath = 'search_list.txt'

# Testing the get_all_waypoints function with the corrected file paths
all_waypoints = get_all_waypoints()
print(all_waypoints)  # Display the waypoints for each target
'''


## FOR ASTAR_CLASS_4
def get_path_to_first_fruit(true_map_file='true_map_full.txt', search_list_file='search_list.txt', start_point=(0,0)):
    astar = CompleteImprovedAStar(true_map_file, search_list_file)
    target_name = astar.targets[0] if astar.targets else None #could create a search index 
    #self input so that it reads the last waypoint of the previous path as the new starting point 
    
    waypoints = None
    if target_name:
        astar.find_path(start_point, target_name)
        waypoints = np.array(astar.path) * astar.grid_size - astar.boundary_size / 2

    # Moved the visualization call outside of the if statement
    astar.visualize_path(smooth=True)

    return waypoints


# Testing the function with the provided files and a start point of (0, 0)
waypoints = get_path_to_first_fruit()

# Printing the waypoints
print(waypoints)
