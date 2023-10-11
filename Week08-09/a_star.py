import numpy as np
from typing import List, Tuple
from queue import PriorityQueue

# A class to represent the nodes in the grid
class Node:
    def __init__(self, x, y, walkable):
        self.x = x
        self.y = y
        self.walkable = walkable
        self.parent = None
        self.g = float('inf')  # The cost to get to this node
        self.h = 0  # The heuristic estimate to the goal
        self.f = 0  # The total cost (g + h)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.f < other.f

# A function to generate the neighbors of a given node
def get_neighbors(node, grid):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x, y = node.x + dx, node.y + dy
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and (dx != 0 or dy != 0):
                neighbors.append(grid[x][y])
    return neighbors

# The complete updated A* function
def astar_pathfinding_complete(grid: List[List[int]], start_coords: Tuple[int, int], goal_coords: Tuple[int, int],
                               stop_distance: float, grid_size: Tuple[int, int], heuristic_function) -> List[Tuple[int, int]]:
    # Creating a 2D array of Node objects from the input grid
    nodes = [[Node(x, y, cell == 0) for y, cell in enumerate(row)] for x, row in enumerate(grid)]
    start_node = nodes[start_coords[0]][start_coords[1]]
    goal_node = nodes[goal_coords[0]][goal_coords[1]]

    # Creating the open list and closed list
    open_list = PriorityQueue()
    open_list.put((0, start_node))
    closed_list = set()

    # Setting the g-cost of the start node to 0
    start_node.g = 0
    start_node.h = heuristic_function(start_node.x, start_node.y, goal_node.x, goal_node.y)
    start_node.f = start_node.h

    while not open_list.empty():
        # Getting the node with the lowest f-cost from the open list
        current_node = open_list.get()[1]

        # Adding the current node to the closed list
        closed_list.add(current_node)

        # If we have reached the goal node, reconstruct the path and return it
        if heuristic_function(current_node.x, current_node.y, goal_node.x, goal_node.y) < stop_distance:
            path = []
            while current_node is not None:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]  # Return the path in the correct order

        # Generating the neighbors of the current node
        neighbors = get_neighbors(current_node, nodes)

        for neighbor in neighbors:
            # If the neighbor is not walkable or is in the closed list, skip it
            if not neighbor.walkable or neighbor in closed_list:
                continue

            # Calculating the new g-cost of the neighbor
            new_g = current_node.g + heuristic_function(current_node.x, current_node.y, neighbor.x, neighbor.y)

            # If the new g-cost is lower than the neighbor's current g-cost, update it
            if new_g < neighbor.g:
                neighbor.g = new_g
                neighbor.h = heuristic_function(neighbor.x, neighbor.y, goal_node.x, goal_node.y)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node
                open_list.put((neighbor.f, neighbor))

    # If there is no path to the goal, return an empty list
    return []