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
import time
from typing import List, Tuple

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

#from astar_class3 import AStar
#from astar_class4 import CompleteImprovedAStar
from astar_class5 import FinalAStar

class PID:
    def __init__(self, kp, ki, kd, max_output=None, min_output=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.prev_error = error

        # Clamping the output if max_output or min_output is set
        if self.max_output is not None and output > self.max_output:
            output = self.max_output
        if self.min_output is not None and output < self.min_output:
            output = self.min_output
            
        return output

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
        #self.paths = self.path_planning()
        self.path = [[0,0]]
        self.search_idx = 0
        self.distance_to_waypoint = 0
        self.target_angle = 0
        self.theta_delta = 0

        #### Maybe add a 'is final waypoint' variable
        self.drive_time = 0
        self.turn_time = 0

        # Wheel Control Parameters
        self.drive_ticks = 25
        self.turn_ticks = 10
        
        #initialise the PID
        self.pid_turn=PID(1,0,0.1)
        self.pid_drive=PID(1,0,0.1)

        # Add known aruco markers and fruits from map to SLAM
        self.fruit_list, self.fruit_true_pos, self.aruco_true_pos = self.read_true_map(args.map)
        self.search_list = self.read_search_list()
        self.marker_pos= np.zeros((2,len(self.aruco_true_pos) + len(self.fruit_true_pos)))
        self.marker_pos, self.taglist = self.merge_slam_map(self.fruit_list, self.fruit_true_pos, self.aruco_true_pos)
        self.obstacles = np.concatenate((self.fruit_true_pos, self.aruco_true_pos))
        #self.ekf.load_map(self.marker_pos, self.taglist, self.P)

        #COUld initialise the path for path planning here. redesign the path planing function so that it outputs the path to self.paths
        self.scale = np.loadtxt("{}scale.txt".format("calibration/param/"), delimiter=',')
        self.baseline = np.loadtxt("{}baseline.txt".format("calibration/param/"), delimiter=',')
    
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
    
    def merge_slam_map(self, fruit_list, fruits_true_pos, aruco_true_pos):
        #adding known aruco markers
        for (i, pos) in enumerate(aruco_true_pos):
            self.taglist.append(i + 1)
            self.marker_pos[0][i] = pos[0]
            self.marker_pos[1][i] = pos[1]

        #adding known fruits
        for (i,pos) in enumerate(fruits_true_pos):
            self.taglist.append(fruit_list[i]) #adding tag
            self.marker_pos[0][i + 10] = pos[0]
            self.marker_pos[1][i + 10] = pos[1]

        return self.marker_pos, self.taglist
    
    def distance(self, current_pose, waypoint):
        return math.sqrt((waypoint[0] - current_pose[0])**2 + (waypoint[1] - current_pose[1])**2)

    def calculate_orientation(self, current_pose, waypoint):
        return math.atan2(waypoint[1] - current_pose[1], waypoint[0] - current_pose[0])
    
    # The get_all_waypoints function
    def path_planning(self, true_map_file='true_map_full.txt', search_list_file='search_list.txt', grid_size=0.1, obstacle_size=0.2, boundary_size=3):
        astar = FinalAStar(true_map_file, search_list_file)
        target_name = astar.targets[self.search_idx] if astar.targets else None #could create a search index 
        #self input so that it reads the last waypoint of the previous path as the new starting point 
        
        waypoints = None
        if target_name:
            #start_point = operate.path[-1] # TODO: Or i could use the robot_pose
            start_point = self.get_robot_pose()
            astar.find_path(start_point, target_name)
            waypoints = np.array(astar.path) * astar.grid_size - astar.boundary_size / 2

        # Moved the visualization call outside of the if statement
        #astar.visualize_path(smooth=True)

        return waypoints
    
    def ramer_douglas_peucker(self,points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
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
            left_result = self.ramer_douglas_peucker(points[:index+1], epsilon)
            right_result = self.ramer_douglas_peucker(points[index:], epsilon)
            return left_result[:-1] + right_result
        
        return [start_point, end_point]

    # Function to ensure that the distance between two consecutive waypoints is not more than a specified maximum distance
    def ensure_max_distance(self,points: List[Tuple[float, float]], max_distance: float) -> List[Tuple[float, float]]:
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
    
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # running in sim
        if args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        # running on physical robot (right wheel reversed)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas
    
    def drive_to_waypoint(self):
        self.robot_pose = self.get_robot_pose()
        self.distance_to_waypoint = self.distance(self.robot_pose, self.wp)

        max_ticks = 40
        min_ticks = 0  # Minimum ticks, adjust this value as needed

        while self.distance_to_waypoint > 0.1:
            speed_multiplier = self.pid_drive.compute(self.distance_to_waypoint)
            
            # Calculate the max speed multiplier based on the distance to the waypoint
            max_speed_multiplier = max_ticks / (self.distance_to_waypoint / (self.drive_ticks * self.scale))
            min_speed_multiplier = min_ticks / (self.distance_to_waypoint / (self.drive_ticks * self.scale))  # if needed

            # Clamping the speed_multiplier to avoid extreme speeds
            speed_multiplier = max(min_speed_multiplier, min(max_speed_multiplier, speed_multiplier))

            self.drive_time = abs(self.distance_to_waypoint / (self.drive_ticks * self.scale * speed_multiplier))
            motion = [1,0]
            self.drive_bot(motion, self.drive_time)
            self.robot_pose = self.get_robot_pose()
            self.distance_to_waypoint = self.distance(self.robot_pose, self.wp)


    def turn_to_waypoint(self):
        self.robot_pose = self.get_robot_pose()
        self.target_angle = self.calculate_orientation(self.robot_pose, self.wp)

        while True:
            self.theta_delta = self.target_angle - self.robot_pose[2]

            # Normalize the angle
            if self.theta_delta > math.pi:
                self.theta_delta -= 2 * math.pi
            elif self.theta_delta < -math.pi:
                self.theta_delta += 2 * math.pi

            # Break the loop if the error is small enough
            if abs(self.theta_delta) <= math.radians(5):
                break

            speed_multiplier = self.pid_turn.compute(self.theta_delta)
            self.turn_time = abs(self.theta_delta * self.baseline) / (2 * self.turn_ticks * self.scale * speed_multiplier)
            motion = [0, 1] if self.theta_delta > 0 else [0, -1]
            self.drive_bot(motion, self.turn_time)  # Using drive_bot to execute the turning motion
            self.robot_pose = self.get_robot_pose()
      
        
    def drive_bot(self,motion, drive_time):
        self.command['motion'] = motion
        self.take_pic()
        #Turn
        operate.control_clock=time.time()
        drive_time += time.time()
        while time.time()<=drive_time:
            drive_meas = self.control()
            self.update_slam(drive_meas)
        
        
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
 

    def get_robot_pose(self):
        states = self.ekf.get_state_vector()
        
        robot_pose = [0.0,0.0,0.0]
        robot_pose = states[0:3, :]
        print(f"robot pose: {robot_pose}")
        return robot_pose
 
    def generate_next_path(self):
        # Path planning
        path_long = self.path_planning()
        epsilon = 0.05  # This is the maximum distance a point can be from the line segment connecting two other points
        reduced_points = self.ramer_douglas_peucker(path_long, epsilon)

        # Ensuring that the maximum distance between any two consecutive waypoints is not more than 0.4 units
        max_distance=0.45
        smoothed_points = self.ensure_max_distance(reduced_points, max_distance)

        # Generation logic here
        return smoothed_points

    def navigate_to_target(self):
        self.current_waypoint_index = 0
        self.search_idx = 0
        self.path = self.generate_next_path()

        while True:
            if not self.path:
                print("Path is empty. Ending navigation.")
                break
            
            self.wp = self.path[self.current_waypoint_index]

            while True:
                self.robot_pose = self.get_robot_pose()
                distance_error = self.distance(self.robot_pose, self.wp)

                if distance_error < 0.1:
                    self.current_waypoint_index += 1
                    if self.current_waypoint_index >= len(self.path):
                        print("Reached final waypoint of current path.")
                        break
                    self.wp = self.path[self.current_waypoint_index]
                    continue

                self.turn_to_waypoint()

                orientation_error = self.calculate_orientation(self.robot_pose, self.wp) - self.robot_pose[2]
                if abs(orientation_error) < math.radians(5):
                    self.drive_to_waypoint()

                print("Waypoint:", self.wp, "Robot Pose:", self.robot_pose)
                time.sleep(0.1)  # Control the loop execution rate

            self.search_idx += 1
            self.path = self.generate_next_path()
            self.current_waypoint_index = 0
            print("Moving to next path.")
 

        
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
    
    # Run SLAM ###########################################
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
    #######################################################
    
    
    # sleep
    time.sleep(0.5)
    
    
    operate.navigate_to_target()

    
    print("COMPLETED THE CODE")   
    sys.exit()

  