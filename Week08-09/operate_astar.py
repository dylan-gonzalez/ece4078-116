# teleoperate the robot, perform SLAM and object detection

import os
import sys
import time
import cv2
import numpy as np
import math
import json
from typing import Tuple, List



# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations


# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# A* components
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from a_star import astar_pathfinding_complete

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
        self.ekf_on =True # False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        #self.detector_output = np.zeros([240, 320], dtype=np.uint8)

    def get_robot_pose(self):
        states = self.ekf.get_state_vector()
        #print(f"robot pose: {states}")
        return states[0:3, :]
        
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

            #return (1), (2), (3)
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
     
    
    
    
    def drive_to_waypoint(self, waypoint, robot_pose):
        '''
            Function that implements the driving of the robot
        '''
        
        # Read in baseline and scale parameters
        datadir = "calibration/param/"
        scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
        
        # Read in the waypoint values
        waypoint_x = waypoint[0]
        waypoint_y = waypoint[1]
        
        # Read in robot pose values
        robot_pose_x = robot_pose[0]
        robot_pose_y = robot_pose[1]
        
        # Wheel ticks in m/s
        wheel_ticks = 30

        target_angle = operate.turn_to_waypoint(waypoint, robot_pose)

        # Drive straight to waypoint
        distance_to_waypoint = math.sqrt((waypoint_x - robot_pose_x)**2 + (waypoint_y - robot_pose_y)**2)
        drive_time = abs(float((distance_to_waypoint) / (wheel_ticks*scale))) 

        print("Driving for {:.2f} seconds".format(drive_time))
        operate.motion_controller([1,0], wheel_ticks, drive_time)
        print("Arrived at [{}, {}]".format(waypoint[1], waypoint[0]))
        
        update_robot_pose = [waypoint[0], waypoint[1], target_angle]
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
        waypoint_x = waypoint[0]
        waypoint_y = waypoint[1]
        
        # Read in robot pose values
        robot_pose_x = robot_pose[0]
        robot_pose_y = robot_pose[1]
        robot_pose_theta = robot_pose[2]
        
        # Wheel ticks in m/s
        wheel_ticks = 30

        # Calculate turning varibles
        turn_time = 0
        theta_target = math.atan2(waypoint_y - robot_pose_y, waypoint_x - robot_pose_x) # angle from robot's current position to the target waypoint
        theta_delta = theta_target - robot_pose_theta # How far the robot must turn from current pose
        
        if theta_delta > math.pi:
            theta_delta -= 2 * math.pi
        elif theta_delta < -math.pi:
            theta_delta += 2 * math.pi

        # Evaluate how long the robot should turn for
        turn_time = float((abs(theta_delta)*baseline)/(2*wheel_ticks*scale))
        print("Turning for {:.2f} seconds".format(turn_time))
        
        if theta_delta == 0:
            print("No turn")
        elif theta_delta > 0:
            operate.motion_controller([0,1], wheel_ticks, turn_time)
        elif theta_delta < 0:
            operate.motion_controller([0,-1], wheel_ticks, turn_time)
        else:
            print("There is an issue with turning function")
            
        return theta_delta # delete once we get the robot_pose working and path plannning
        
        
        
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
            drive_meas = measure.Drive(lv,rv,drive_time)
            operate.update_slam(drive_meas)
        
    
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

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,
                                position=(h_pad, 240 + 2 * v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                            False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    # keyboard teleoperation, replace with your M1 codes if preferred        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0] + 1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0] - 1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1] + 1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1] - 1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    def indices_to_coordinates(x, y, grid_size, map_size=3):
        """
        Convert grid indices to real world coordinates.
        The center of the grid is mapped to the origin (0,0) in real world coordinates.
        """
        grid_center = grid_size // 2  
        scale = map_size / grid_size  
        return (y - grid_center) * scale, (x - grid_center) * scale


    def navigate_to_fruits_fixed_grid(robot_pose: Tuple[float, float], 
                                    grid: List[List[int]], 
                                    targets: List[Tuple[float, float]], 
                                    stop_distance: float,
                                    heuristic_function) -> None:
        """
        Navigate through multiple waypoints to reach target fruits using A* pathfinding.

        Parameters:
        - robot_pose: Current robot pose in real-world coordinates (x, y).
        - grid: 2D list representing the environment, with 0 for free space and 1 for obstacles.
        - targets: List of tuples representing the target fruits' positions in real-world coordinates.
        - stop_distance: The distance to stop from the goal.
        - heuristic_function: Function to compute the heuristic value.
        """
        FIXED_GRID_SIZE = (30,30)
        
        for target in targets:
            # Convert real-world coordinates to grid indices for the target
            target_indices = operate.coordinates_to_indices(target[0], target[1], FIXED_GRID_SIZE)
            
            # Find the path to the target using A* pathfinding
            path = astar_pathfinding_complete(grid, 
                                            operate.coordinates_to_indices(robot_pose[0], robot_pose[1], FIXED_GRID_SIZE),
                                            target_indices,
                                            stop_distance,
                                            FIXED_GRID_SIZE,
                                            heuristic_function)
            
            # If a path is found, navigate through the waypoints
            if path:
                print(f"Path found to target {target}: {path}")
                for waypoint in path:
                    # Convert grid indices to real-world coordinates for the waypoint
                    waypoint_coordinates = operate.indices_to_coordinates(waypoint[0], waypoint[1], FIXED_GRID_SIZE)
                    
                    # Drive the robot to the waypoint
                    robot_pose = operate.drive_to_waypoint(waypoint_coordinates, robot_pose)
            else:
                print(f"No path found to target {target}")
            print(f"Arrived at target {target}\n")


# main loop
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt')
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

    #***************************Astar Implementation*****************************************************
    '''
    obstacles = []
    for x,y in fruits_true_pos:
        obstacles.append([x,y])

    for x,y in aruco_true_pos:
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

    search_list = operate.read_search_list()
    for idx in range(len(search_list)):
        print(f'Going to fruit {search_list[idx]}')
        
        x_grid, y_grid = convert_to_grid_space(fruits_true_pos[idx][0] + 0.3, fruits_true_pos[idx][1] + 0.3)
        end = grid.node(x_grid, y_grid)
        print(f'Start: {start}')
        print(f'End {end}')

        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, runs = finder.find_path(start, end, grid)
        

        #-----implement drive to point code----
        
        ######################################



        print('operations:', runs, 'path length:', len(path))
        print(grid.grid_str(path=path, start=start, end=end))
        start = grid.node(x_grid, y_grid)
        grid.cleanup()
        '''
        

    # Create a 30x30 grid initialized with zeros
    grid = np.zeros((30, 30))

    # Function to convert real-world coordinates to grid indices
    def coordinates_to_grid_indices(x, y, grid_size=30, map_size=3):
        grid_center = grid_size // 2
        scale = grid_size / map_size
        return int(grid_center + (y * scale)), int(grid_center + (x * scale))

    # Example obstacle data (replace this with actual data)
    obstacles = []
    for x,y in fruits_true_pos:
        obstacles.append([x,y])

    for x,y in aruco_true_pos:
        obstacles.append([x,y])

    # Mark obstacles on the grid
    for obstacle in obstacles:
        x, y = coordinates_to_grid_indices(obstacle["x"], obstacle["y"])
        grid[x, y] = 1

    # Print the grid (optional)
    print(grid)


sys.exit()
