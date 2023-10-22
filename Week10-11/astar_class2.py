import numpy as np
import matplotlib.pyplot as plt
import json
import re
from queue import PriorityQueue

class AStar:
    def __init__(self, occupancy_grid):
        self.occupancy_grid = occupancy_grid
        self.num_cells = occupancy_grid.shape[0]

    def heuristic(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def a_star_with_obstacle_avoidance(self, start, goal, buffer_cells):
        # Complete A* algorithm implementation including obstacle avoidance
        open_list = PriorityQueue()
        open_list.put((0, start))
        came_from = {}
        g_score = {cell: float('inf') for row in self.occupancy_grid for cell in row}
        g_score[start] = 0
        f_score = {cell: float('inf') for row in self.occupancy_grid for cell in row}
        f_score[start] = self.heuristic(start, goal)

        open_set_hash = {start}

        while not open_list.empty():
            current = open_list.get()[1]
            open_set_hash.remove(current)

            if current == goal or self.heuristic(current, goal) <= buffer_cells:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Return reversed path

            for neighbor in [(0, 1), (1, 0), (-1, 0), (0, -1)]:  # 4-connected grid
                x, y = current[0] + neighbor[0], current[1] + neighbor[1]

                if 0 <= x < self.num_cells and 0 <= y < self.num_cells and self.occupancy_grid[x][y] == 0:
                    temp_g_score = g_score[current] + 1

                    if temp_g_score < g_score[(x, y)]:
                        came_from[(x, y)] = current
                        g_score[(x, y)] = temp_g_score
                        f_score[(x, y)] = temp_g_score + self.heuristic((x, y), goal)
                        if (x, y) not in open_set_hash:
                            open_list.put((f_score[(x, y)], (x, y)))
                            open_set_hash.add((x, y))

        return None  # No path found

    # Function to create the occupancy grid from the true map content
    def create_occupancy_grid(true_map_content, grid_size, map_size):
        occupancy_grid = np.zeros((int(map_size / grid_size), int(map_size / grid_size)))

        for item in true_map_content:
            label, x, y = item
            if 'obstacle' in label:  # Marking obstacles
                i, j = int((x + map_size / 2) / grid_size), int((y + map_size / 2) / grid_size)
                occupancy_grid[i, j] = 1

        return occupancy_grid

    def buffer_obstacles(occupancy_grid, buffer_cells):
        buffered_occupancy_grid = np.copy(occupancy_grid)
        num_cells = occupancy_grid.shape[0]

        for i in range(num_cells):
            for j in range(num_cells):
                if occupancy_grid[i, j] == 1:
                    min_x = max(0, i - buffer_cells)
                    max_x = min(num_cells, i + buffer_cells + 1)
                    min_y = max(0, j - buffer_cells)
                    max_y = min(num_cells, j + buffer_cells + 1)
                    buffered_occupancy_grid[min_x:max_x, min_y:max_y] = 1

        return buffered_occupancy_grid
    
    def visualize_sequential_paths(self, start, targets, paths):
        plt.figure(figsize=(10,10))
        plt.imshow(self.occupancy_grid, cmap='gray_r', origin='lower')
        for path in paths:
            if path:
                y, x = zip(*path)
                plt.plot(x, y, '-o')
        plt.plot(start[1], start[0], 'ro')
        for target in targets:
            plt.plot(target[1], target[0], 'bo')
        plt.show()

if __name__ == "__main__":
    # Load the true map and search list
    with open('M4_true_map_full.txt', 'r') as file:
        true_map_content = file.readlines()

    search_list = np.loadtxt('search_list.txt', dtype=str)

    # Parse the true map content to extract labels and coordinates
    parsed_true_map = json.loads(true_map_content[0])

    # Extract coordinates of the items specified in the search list from the parsed true map
    target_coordinates_from_true_map = {label: coords for label, coords in parsed_true_map.items() 
                                        if label.split('_')[0] in search_list}
    
    # Convert target coordinates to grid indices and define the start position at the center of the map
    map_size = 3.0  # The size of the map in meters
    grid_size = 0.05  # The size of each grid cell in meters
    buffer_distance = 0.3  # The distance to get within each target in meters
    buffer_cells = int(buffer_distance / grid_size)  # The number of cells equivalent to the buffer distance

    target_coordinates_indices = [(int((coord['x'] + map_size / 2) / grid_size), 
                                int((coord['y'] + map_size / 2) / grid_size)) 
                                for coord in target_coordinates_from_true_map.values()]

    start = (int(map_size / (2 * grid_size)), int(map_size / (2 * grid_size)))
    
    # Creating an instance of the AStar class with the provided occupancy grid
    astar = AStar(buffered_occupancy_grid)

    # Assuming buffered_occupancy_grid is defined, replace this with the actual occupancy grid
    example_occupancy_grid = astar.create_occupancy_grid(true_map_content, grid_size, map_size)

    # Create a buffered occupancy grid by expanding obstacles
    buffer_distance = 0.3  # The distance to get within each target in meters
    buffer_cells = int(buffer_distance / grid_size)
    buffered_occupancy_grid = astar.buffer_obstacles(example_occupancy_grid, buffer_cells)


    # Compute paths using the enhanced A* with obstacle avoidance for each target
    paths = []
    for target in target_coordinates_indices:
        path = astar.a_star_with_obstacle_avoidance(start, target, buffer_cells)
        paths.append(path)
        if path:
            start = path[-1]  # Update the start for the next target

    # Visualize the paths on the occupancy grid
    astar.visualize_sequential_paths((int(map_size / (2 * grid_size)), int(map_size / (2 * grid_size))), 
                                     target_coordinates_indices, paths)
