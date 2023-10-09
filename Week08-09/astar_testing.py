from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import json
import numpy as np

def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list

def read_true_map(fname):
        """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

        @param fname: filename of the map
        @return:
            1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
            2) locations of the targets, [[x1, y1], ..... [xn, yn]]
            3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
        """
        with open(fname, 'r') as fd:
            gt_dict = json.load(fd)
            fruit_list = []
            fruit_true_pos = []
            aruco_true_pos = np.empty([10, 2])

            # remove unique id of targets of the same type
            for key in gt_dict:
                x = np.round(gt_dict[key]['x'], 1)
                y = np.round(gt_dict[key]['y'], 1)

                if key.startswith('aruco'):
                    if key.startswith('aruco10'):
                        aruco_true_pos[9][0] = x
                        aruco_true_pos[9][1] = y
                    else:
                        marker_id = int(key[5]) - 1
                        aruco_true_pos[marker_id][0] = x
                        aruco_true_pos[marker_id][1] = y
                else:
                    fruit_list.append(key[:-2])
                    if len(fruit_true_pos) == 0:
                        fruit_true_pos = np.array([[x, y]])
                    else:
                        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)
                        
                    #print(f'fruit: {fruit_list[-1]} at {fruit_true_pos}')

            #return (1), (2), (3)
            return fruit_list, fruit_true_pos, aruco_true_pos


fname = "M4_true_map_full.txt"
fruit_list, fruit_pos, aruco_pos = read_true_map(fname)

obstacles = [] 
for x,y in fruit_pos:
    obstacles.append([x,y])

for x,y in aruco_pos:
    obstacles.append([x,y])

#Generate occupancy grid
width = 3 #m
height = 3 #m
n_cells_y = 20
n_cells_x = 20
matrix = [[1 for _ in range(n_cells_y)] for _ in range(n_cells_x)] #preallocate

def convert_to_grid_space(x,y):
    # Convert real-world coordinates to grid coordinates
    x_grid = int(((x + width/2) / width) * (n_cells_x- 1))
    y_grid = int(((y + height/2) / height) * (n_cells_y - 1))

    if 0 <= x_grid < n_cells_x and 0 <= y_grid < n_cells_y:
    
        # Mark the corresponding grid cell as an obstacle (e.g., set it to 1)
        #matrix[y_grid][x_grid] = 1
        return x_grid, y_grid

print("adding obstacles")
for x,y in obstacles:
    x_grid, y_grid = convert_to_grid_space(x,y)

    matrix[y_grid][x_grid] = 0

for row in matrix:
    print(row)

grid = Grid(matrix=matrix)

start_x, start_y = convert_to_grid_space(0,0)
start = grid.node(start_x, start_y)

search_list = read_search_list()
for idx in range(len(search_list)):
    print(f'Going to fruit {search_list[idx]}')
    
    x_grid, y_grid = convert_to_grid_space(fruit_pos[idx][0] + 0.3, fruit_pos[idx][1] + 0.3)
    end = grid.node(x_grid, y_grid)
    print(f'Start: {start}')
    print(f'End {end}')

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)

    print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start, end=end))
    start = grid.node(x_grid, y_grid)
    grid.cleanup()

