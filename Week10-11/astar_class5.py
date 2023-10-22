import numpy as np
import matplotlib.pyplot as plt
import json
from queue import PriorityQueue

class FinalAStar:
    def __init__(self, true_map_file, search_list_file, grid_size=0.05, obstacle_size=0.2, boundary_size=3):
        self.true_map_file = true_map_file
        self.search_list_file = search_list_file
        self.grid_size = grid_size
        self.obstacle_size = obstacle_size
        self.boundary_size = boundary_size
        self.obstacles = []
        self.targets = []
        self.grid = None
        self.start = None
        self.end = None
        self.path = []
        
        self._parse_true_map()
        self._parse_search_list()
        self._create_grid()
        self._place_obstacles()

    def _parse_true_map(self):
        with open(self.true_map_file, 'r') as file:
            data = json.loads(file.read().strip())
            for name, coords in data.items():
                x = int((coords['x'] + self.boundary_size / 2) / self.grid_size)
                y = int((coords['y'] + self.boundary_size / 2) / self.grid_size)
                self.obstacles.append((x, y, name))

    def _parse_search_list(self):
        with open(self.search_list_file, 'r') as file:
            self.targets = [line.strip() for line in file.readlines()]

    def _create_grid(self):
        grid_size = int(self.boundary_size / self.grid_size)
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

    def _place_obstacles(self):
        obstacle_size_in_cells = int(self.obstacle_size / self.grid_size)
        for x, y, _ in self.obstacles:
            self.grid[x - obstacle_size_in_cells//2:x + obstacle_size_in_cells//2 + 1,
                      y - obstacle_size_in_cells//2:y + obstacle_size_in_cells//2 + 1] = 1
    
    def _heuristic(self, current, goal):
        return np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)

    def _get_neighbors(self, cell):
        neighbors = [
            ((cell[0] - 1, cell[1]), 1), ((cell[0] + 1, cell[1]), 1), 
            ((cell[0], cell[1] - 1), 1), ((cell[0], cell[1] + 1), 1),
            ((cell[0] - 1, cell[1] - 1), 1.4), ((cell[0] + 1, cell[1] + 1), 1.4), 
            ((cell[0] - 1, cell[1] + 1), 1.4), ((cell[0] + 1, cell[1] - 1), 1.4),
        ]
        valid_neighbors = [((x, y), cost) for (x, y), cost in neighbors 
                           if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]]
        return valid_neighbors

    def find_path(self, start, target_name):
        self.start = (int((start[0] + self.boundary_size / 2) / self.grid_size), 
                      int((start[1] + self.boundary_size / 2) / self.grid_size))
        
        target_coords = next((x, y) for x, y, name in self.obstacles if target_name in name)
        distance_cells = int(0.3 / self.grid_size)
        stop_x = target_coords[0]
        stop_y = max(0, target_coords[1] - distance_cells)
        self.end = (stop_x, stop_y)

        open_list = PriorityQueue()
        open_list.put((0, self.start))
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}

        while not open_list.empty():
            _, current = open_list.get()

            if current == self.end:
                break

            for (neighbor, move_cost) in self._get_neighbors(current):
                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    if self.grid[neighbor] == 1:
                        continue
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, self.end)
                    open_list.put((priority, neighbor))
                    came_from[neighbor] = current

        self.path = []
        while current != self.start:
            self.path.append(current)
            current = came_from[current]
        self.path.append(self.start)
        self.path.reverse()

        # Convert grid indices to original coordinates and return the waypoints
        waypoints = [(y * self.grid_size - self.boundary_size / 2, 
                      x * self.grid_size - self.boundary_size / 2) for x, y in self.path]
        return waypoints

    def visualize_path(self, smooth=False):
        plt.imshow(self.grid, cmap='binary')
        plt.plot(self.start[1], self.start[0], 'ro')

        if self.path:
            path_x = [x[1] for x in self.path]
            path_y = [x[0] for x in self.path]
            plt.plot(path_x, path_y, 'g')
            plt.plot(self.end[1], self.end[0], 'gs')

        for x, y, _ in self.obstacles:
            plt.plot(y, x, 'bs')

        plt.show()

'''
# File paths
true_map_file = '/mnt/data/M4_true_map_full.txt'
search_list_file = '/mnt/data/search_list.txt'

# Create an object of the FinalAStar class
astar = FinalAStar(true_map_file, search_list_file)

# Specify the starting position and target name
start_position = (0, 0)
first_target = astar.targets[0]

# Find and visualize the path
waypoints = astar.find_path(start_position, first_target)
astar.visualize_path()

# Waypoints
print("Waypoints:", waypoints)
'''
