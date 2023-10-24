import numpy as np
import matplotlib.pyplot as plt
import json
from queue import PriorityQueue
from scipy.interpolate import CubicSpline

class CompleteImprovedAStar:
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
        for x, y, name in self.obstacles:
            self.grid[x - obstacle_size_in_cells//2:x + obstacle_size_in_cells//2 + 1,
                      y - obstacle_size_in_cells//2:y + obstacle_size_in_cells//2 + 1] = 1
    
    def _heuristic(self, current, goal):
        return np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)

    def _get_neighbors(self, cell):
        neighbors = [
            (cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1),
            (cell[0] - 1, cell[1] - 1), (cell[0] + 1, cell[1] + 1), (cell[0] - 1, cell[1] + 1), (cell[0] + 1, cell[1] - 1),
        ]
        valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]]
        return valid_neighbors
    
    def _smooth_path(self):
        if len(self.path) < 2:
            return self.path
        
        x = np.array([point[1] for point in self.path])
        y = np.array([point[0] for point in self.path])

        t = np.arange(x.shape[0])
        spline_x = CubicSpline(t, x, bc_type='natural')
        spline_y = CubicSpline(t, y, bc_type='natural')

        t_new = np.linspace(0, t[-1], 100)
        x_smooth = spline_x(t_new)
        y_smooth = spline_y(t_new)
        
        return np.column_stack((y_smooth, x_smooth))
    
    def find_path(self, start, target_name):
        self.start = (int((start[0] + self.boundary_size / 2) / self.grid_size), 
                      int((start[1] + self.boundary_size / 2) / self.grid_size))
        
        target_coords = next((x, y) for x, y, name in self.obstacles if target_name in name)
        
        stop_x = target_coords[0]
        stop_y = max(0, target_coords[1] - int(0.3 / self.grid_size))
        self.end = (stop_x, stop_y)

        open_list = PriorityQueue()
        open_list.put((0, self.start))
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}

        while not open_list.empty():
            _, current = open_list.get()

            if current == self.end:
                break

            for neighbor in self._get_neighbors(current):
                new_cost = cost_so_far[current] + 1
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

    def visualize_path(self, smooth=False):
        plt.imshow(self.grid, cmap='binary')
        plt.plot(self.start[1], self.start[0], 'ro')

        if self.path:
            if smooth:
                smoothed_path = self._smooth_path()
                plt.plot(smoothed_path[:, 1], smoothed_path[:, 0], 'g')
            else:
                path_x = [x[1] for x in self.path]
                path_y = [x[0] for x in self.path]
                plt.plot(path_x, path_y, 'g')
            plt.plot(self.end[1], self.end[0], 'gs')

        for x, y, _ in self.obstacles:
            plt.plot(y, x, 'bs')

        plt.show()
"""
# Testing the complete updated AStar class
final_a_star = CompleteImprovedAStar(
    true_map_file='/mnt/data/true_map_full.txt', 
    search_list_file='/mnt/data/search_list.txt'
)

# Find the path
final_a_star.find_path([0,0], [1,1])

# Visualize the smoothed path
final_a_star.visualize_path(smooth=True)
"""
