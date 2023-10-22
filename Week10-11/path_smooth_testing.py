from typing import List, Tuple
import numpy as np

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

# Test data
waypoints = [(0.0, 0.0), (0.05, 0.05), (0.1, 0.1), (0.15, 0.15), (0.2, 0.2), (0.25, 0.25),
             (0.3, 0.3), (0.35, 0.35), (0.4, 0.4), (0.45, 0.45), (0.5, 0.5), (0.55, 0.55),
             (0.6, 0.6), (0.65, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.75), (0.8, 0.75),
             (0.85, 0.75), (0.9, 0.75), (0.95, 0.75), (1.0, 0.75), (1.05, 0.75), (1.1, 0.75),
             (1.15, 0.75), (1.2, 0.75), (1.25, 0.75), (1.3, 0.75), (1.35, 0.75), (1.4, 0.75),
             (1.45, 0.75)]

# Applying the Ramer-Douglas-Peucker algorithm
epsilon = 0.05  # This is the maximum distance a point can be from the line segment connecting two other points
reduced_points = ramer_douglas_peucker(waypoints, epsilon)

# Ensuring that the maximum distance between any two consecutive waypoints is not more than 0.4 units
max_distance = 0.2
smoothed_points = ensure_max_distance(reduced_points, max_distance)

print(smoothed_points)
