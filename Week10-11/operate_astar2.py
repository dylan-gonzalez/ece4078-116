# teleoperate the robot, perform SLAM and object detection

# Basic python packages
import os
import sys
import time
import cv2
import numpy as np
import math
import json
import copy

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import shutil                       # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import YOLO components 
#from YOLO.detector import Detector

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from astar_class2 import AStar


class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False # False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()

        # M5 Parameter Initialisation
        self.robot_pose = [0.0,0.0,0.0]
        self.waypoints = []
        self.wp = [0,0]
        #self.min_dist = 50 
        self.taglist = []
        self.paths = [[[0,0],[0.5,0.5]],[[0.5,0.5],[1,1]],[[1,1],[1,0.5]]]
        self.path_idx = 0
        self.distance_to_waypoint = 0
        self.target_angle = 0
        self.theta_delta = 0

        #### Maybe add a 'is final waypoint' variable

        # Wheel Control Parameters
        self.drive_ticks = 30
        self.turn_ticks = 10

        # Add known aruco markers and fruits from map to SLAM
        self.fruit_list, self.fruit_true_pos, self.aruco_true_pos = self.read_true_map(args.true_map)
        self.search_list = self.read_search_list()
        self.marker_pos= np.zeros((2,len(self.aruco_true_pos) + len(self.fruit_true_pos)))
        self.marker_pos, self.taglist, self.P = self.parse_slam_map(self.fruit_list, self.fruit_true_pos, self.aruco_true_pos)
        self.ekf.load_map(self.marker_pos, self.taglist, self.P)

        #COUld initialise the path for path planning here. redesign the path planing function so that it outputs the path to self.paths
        #self.astar_path()



    def get_robot_pose(self):
        states = self.ekf.get_state_vector()
        
        robot_pose = [0.0,0.0,0.0]
        robot_pose = states[0:3, :]
        #print(f"robot pose: {robot_pose}")

        return robot_pose
        
    def read_true_map(self, fname):
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
            return fruit_list, fruit_true_pos, aruco_true_pos


    def read_search_list(self):
        """Read the search order of the target fruits

        @return: search order of the target fruits
        """
        search_list = []
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list


    def print_target_fruits_pos(self, search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order

        @param search_list: search order of the fruits
        @param fruit_list: list of target fruits
        @param fruit_true_pos: positions of the target fruits
        """

        print("Search order:")
        n_fruit = 1
        for fruit in search_list:
            for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
                if fruit == fruit_list[i]:
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                    fruit,
                                                    np.round(fruit_true_pos[i][0], 1),
                                                    np.round(fruit_true_pos[i][1], 1)))
            n_fruit += 1    
    
    
    def drive_to_waypoint(self, waypoint, robot_pose, is_final_waypoint):
        '''
            Function that implements the driving of the robot
        '''
        # Read in baseline and scale parameters
        datadir = "calibration/param/"
        scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
        # Read in the waypoint values
        waypoint_x = self.wp[0]
        waypoint_y = self.wp[1]  
        # Read in robot pose values
        robot_pose_x = self.robot_pose[0]
        robot_pose_y, = self.robot_pose[1]
        robot_pose_theta = self.robot_pose[2]

        threshold = 0.3 if is_final_waypoint else 0.05

        # Calculate the target angle and turn the robot
        operate.turn_to_waypoint(waypoint, robot_pose)
        # Ensure the robot is facing the waypoint before starting to drive
        angle_threshold = 5
        while abs(self.target_angle - robot_pose_theta) > angle_threshold:
            self.robot_pose = operate.get_robot_pose()   # update robot pose
            robot_pose_theta = robot_pose[2]
            operate.turn_to_waypoint(waypoint, robot_pose)
            
        # Now that the robot is facing the waypoint we can drive forwards
        while True:
            # Update robot pose
            robot_pose = operate.get_robot_pose()
            robot_pose_x, robot_pose_y = self.robot_pose[:2]

            # Drive straight to waypoint
            self.distance_to_waypoint = math.sqrt((waypoint_x - robot_pose_x)**2 + (waypoint_y - robot_pose_y)**2)
            
            # Check if the robot is close to the waypoint
            if self.distance_to_waypoint < threshold:
                print("Arrived at [{}, {}]".format(waypoint[1], waypoint[0]))
                break
            
            # Drive straight to the waypoint
            drive_time = abs(float((self.distance_to_waypoint) / (self.drive_ticks*scale))) 
            print("Driving for {:.2f} seconds".format(drive_time))
            operate.motion_controller([1,0], self.drive_ticks, drive_time)
        
        update_robot_pose = [waypoint[0], waypoint[1], self.target_angle]
        return update_robot_pose 


    def turn_to_waypoint(self, waypoint, robot_pose):
        '''
            Function that implements the turning of the robot
        '''
        # Read in baseline and scale parameters
        datadir = "calibration/param/"
        scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
        baseline = np.loadtxt("{}baseline.txt".format(datadir), delimiter=',')

        # Read in the waypoint values
        waypoint_x = self.wp[0]
        waypoint_y = self.wp[1]  
        # Read in robot pose values
        robot_pose_x = self.robot_pose[0]
        robot_pose_y, = self.robot_pose[1]
        robot_pose_theta = self.robot_pose[2]

        # Calculate turning varibles
        turn_time = 0
        self.target_angle = math.atan2(waypoint_y - robot_pose_y, waypoint_x - robot_pose_x) # angle from robot's current position to the target waypoint
        self.theta_delta = self.target_angle - robot_pose_theta # How far the robot must turn from current pose
        
        if self.theta_delta > math.pi:
            self.theta_delta -= 2 * math.pi
        elif self.theta_delta < -math.pi:
            self.theta_delta += 2 * math.pi

        # Evaluate how long the robot should turn for
        turn_time = float((abs(self.theta_delta)*baseline)/(2*self.turn_ticks*scale))
        print("Turning for {:.2f} seconds".format(turn_time))
        
        if self.theta_delta == 0:
            print("No turn")
        elif self.theta_delta > 0:
            operate.motion_controller([0,1], self.turn_ticks, turn_time)
        elif self.theta_delta < 0:
            operate.motion_controller([0,-1], self.turn_ticks, turn_time)
        else:
            print("There is an issue with turning function")
            
        return self.theta_delta # delete once we get the robot_pose working and path plannning
        
        
        
    def motion_controller(self, motion, wheel_ticks, drive_time):
        lv,rv = 0.0, 0.0       
        
        if not motion == [0,0]:
            if motion[0] == 0:  # Turn
                lv, rv = ppi.set_velocity(motion, tick=wheel_ticks, time=drive_time)
            else:   # Drive forward
                lv, rv = ppi.set_velocity(motion, tick=wheel_ticks, time=drive_time)
            
            # A good place to add the obstacle detection algorithm
            
            # Run SLAM Update Sequence
            operate.take_pic()
            drive_meas = measure.Drive(lv,-rv,drive_time)
            operate.update_slam(drive_meas)
            #operate.record_data()
            #operate.save_image()
            #operate.detect_target()
    
    def generate_path(self):
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
                
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()

        if self.data is not None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:  # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            # need to convert the colour before passing to YOLO
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)

            # covert the colour back for display purpose
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)

            # self.command['inference'] = False     # uncomment this if you do not want to continuously predict
            self.file_output = (yolo_input_img, self.ekf)

            # self.notification = f'{len(self.detector_output)} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                # image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    def run_slam(self, aruco_true_pos):
        operate.ekf.taglist = np.array([1,2,3,4,5,6,7,8,9,10])
        operate.ekf.markers = aruco_true_pos
        
        ################Run SLAM####################
        n_observed_markers = len(operate.ekf.taglist)
        if n_observed_markers == 0:
            if not operate.ekf_on:
                print('SLAM is running')
                operate.ekf_on = True
            else:
                print('> 2 landmarks is required for pausing')
        elif n_observed_markers < 3:
            print('> 2 landmarks is required for pausing')
        else:
            if not operate.ekf_on:
                operate.request_recover_robot = True
            operate.ekf_on = not operate.ekf_on
            if operate.ekf_on:
                print('SLAM is running')
            else:
                print('SLAM is paused')    
        ###########################################
        
    
# main loop
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--map", type=str, default='true_map_full.txt')
    #parser.add_argument("--map", type=str, default='lab_out/targets.txt')
    
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)
    operate = Operate(args)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = operate.read_true_map(args.map) #list of fruits names, locations of fruits, locations of aruco markers
    search_list = operate.read_search_list() #inputted ordered list of fruits to search 
    operate.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    print(f'fruits_true_pos: {fruits_true_pos}')
    print(f'search list: {search_list}')
    
    # Define starting and robot pose 
    start = np.array([0.0, 0.0])
    robot_pose =  np.array([0.0, 0.0, 0.0])
    waypoint = [0.0,0.0]
    
    # Run Slam
    operate.run_slam(aruco_true_pos)    
    
    # Add obstacles
    obstacles = []
    for x,y in fruits_true_pos:
        obstacles.append([x,y])

    for x,y in aruco_true_pos:
        obstacles.append([x,y])
        
    print(f"Obstacles are: {obstacles}")

    #Generate occupancy grid
    width = 3 #m
    height = 3 #m
    n_cells_y = 20
    n_cells_x = 20
    matrix = [[1 for _ in range(n_cells_y)] for _ in range(n_cells_x)] #preallocate

    def convert_to_grid_space(x,y):
        x_grid = int(((x + width/2) / width) * (n_cells_x - 1))
        y_grid = int(((y + height/2) / height) * (n_cells_y - 1))
        
        # Clamping values to be within the grid
        x_grid = max(0, min(x_grid, n_cells_x - 1))
        y_grid = max(0, min(y_grid, n_cells_y - 1))
        
        return x_grid, y_grid

    
    def convert_to_world_space(x,y):
        return x/(n_cells_x - 1) * width - width/2, y/(n_cells_y-1) * height - height/2
  
    print("adding obstacles")
    for x,y in obstacles:
        print(f"x - {x}, y - {y}")
        x_grid, y_grid = convert_to_grid_space(x,y)
        matrix[y_grid][x_grid] = 0
        
    print("###############################################################################################")

    for row in matrix:
        print(row)

    grid = Grid(matrix=matrix)

    start_x, start_y = convert_to_grid_space(0,0)
    start = grid.node(start_x, start_y)
    
    search_list = operate.read_search_list()
    for idx in range(len(search_list)):
        print(f'Going to fruit {search_list[idx]}')
        
        x_grid, y_grid = convert_to_grid_space(fruits_true_pos[idx][0] + 0.3, fruits_true_pos[idx][1] + 0.3)
        end = grid.node(x_grid, y_grid)
        print(f'Start: {start}')
        print(f'End {end}')

        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, runs = finder.find_path(start, end, grid)
        
        print('operations:', runs, 'path length:', len(path))
        print(grid.grid_str(path=path, start=start, end=end))
        start = grid.node(x_grid, y_grid)
        grid.cleanup()
        
        waypoints = []
        for node in path:
            x,y = convert_to_world_space(node.x, node.y)
            waypoints.append((x,y))
            
        for wp in waypoints:
            print(wp)
        
        for wp in waypoints:
            if wp == waypoints[-1]:
                is_final_waypoint = 1
            else: 
                is_final_waypoint = 0
                
            robot_pose = operate.get_robot_pose()                    
            operate.drive_to_waypoint(wp, robot_pose, is_final_waypoint)
            print(f"robot pose:  {operate.get_robot_pose()}")
            
        # Implement a delay of 2 seconds and update SLAM
        print("Initiating the Delay of 2 seconds") 
        operate.motion_controller([0,0],0,2)
        
        #### WE NEED TO ADD A BUFFER AROUND THE OBSTACLE


sys.exit()