from typing import List, Tuple
import numpy as np
from astar_class5 import FinalAStar

def ramer_douglas_peucker(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """
    Ramer-Douglas-Peucker algorithm for path smoothing.
    
    Parameters:
    - points: A list of tuples representing the waypoints (x, y).
    - epsilon: The maximum distance from a point to a line formed by two other points.
    
    Returns:
    - A reduced list of tuples representing the smoothed path.
    """
    # Find the point with the maximum distance from the line formed by the start and end points
    start_point, end_point = points[0], points[-1]
    max_distance = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        distance = np.abs(np.cross(np.array(end_point) - np.array(start_point), 
                                   np.array(start_point) - np.array(points[i]))) / np.linalg.norm(np.array(end_point) - np.array(start_point))
        if distance > max_distance:
            index = i
            max_distance = distance
    
    # If the max distance is greater than epsilon, recursively apply the algorithm to the two subarrays
    if max_distance > epsilon:
        left_result = ramer_douglas_peucker(points[:index+1], epsilon)
        right_result = ramer_douglas_peucker(points[index:], epsilon)
        return left_result[:-1] + right_result
    
    return [start_point, end_point]

# Function to ensure that the distance between two consecutive waypoints is not more than a specified maximum distance
def ensure_max_distance(points: List[Tuple[float, float]], max_distance: float) -> List[Tuple[float, float]]:
    """
    Ensure the distance between two consecutive waypoints is not more than the specified max_distance.
    
    Parameters:
    - points: A list of tuples representing the waypoints (x, y).
    - max_distance: The maximum allowed distance between two consecutive waypoints.
    
    Returns:
    - A list of tuples representing the waypoints satisfying the max_distance condition.
    """
    smoothed_points = [points[0]]
    for i in range(1, len(points)):
        distance = np.linalg.norm(np.array(points[i]) - np.array(smoothed_points[-1]))
        if distance > max_distance:
            # Calculate the number of intermediate points needed
            num_intermediate_points = int(np.ceil(distance / max_distance)) - 1
            
            # Calculate the vector from the current point to the next point
            vector = np.array(points[i]) - np.array(smoothed_points[-1])
            
            # Normalize the vector and scale it to the max_distance
            vector = (vector / np.linalg.norm(vector)) * max_distance
            
            # Add the intermediate points
            for j in range(num_intermediate_points):
                intermediate_point = np.array(smoothed_points[-1]) + vector
                smoothed_points.append(tuple(intermediate_point))
        
        smoothed_points.append(points[i])
    
    return smoothed_points

## FOR ASTAR_CLASS_5
def get_path_to_first_fruit(true_map_file='true_map_full.txt', search_list_file='search_list.txt', start_point=(0,0)):
    astar = FinalAStar(true_map_file, search_list_file)
    target_name = astar.targets[0] if astar.targets else None #could create a search index 
    #self input so that it reads the last waypoint of the previous path as the new starting point 
    
    waypoints = None
    if target_name:
        astar.find_path(start_point, target_name)
        waypoints = np.array(astar.path) * astar.grid_size - astar.boundary_size / 2

    # Moved the visualization call outside of the if statement
    #astar.visualize_path(smooth=True)

    return waypoints


# Testing the function with the provided files and a start point of (0, 0)
waypoints = get_path_to_first_fruit()

# Applying the Ramer-Douglas-Peucker algorithm
epsilon = 0.05  # This is the maximum distance a point can be from the line segment connecting two other points
reduced_points = ramer_douglas_peucker(waypoints, epsilon)

# Ensuring that the maximum distance between any two consecutive waypoints is not more than 0.4 units
max_distance = 0.45
smoothed_points = ensure_max_distance(reduced_points, max_distance)

print(len(smoothed_points))
#print(smoothed_points)

