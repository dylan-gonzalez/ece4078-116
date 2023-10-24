import os
import sys
import time
import cv2
import numpy as np
import math
import json


#from rrt3 import RrtConnect
#from rrt import RRTC
from Obstacle import *
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

        #return (1), (2), (3)
        return fruit_list, fruit_true_pos, aruco_true_pos
    


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



def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
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



def drive_to_waypoint(verti_dist):
    '''
        Function that implements the driving of the robot
    '''    
    # Read in baseline and scale parameters
    datadir = "calibration/param/"
    scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
    
    drive_ticks = 25
    
    drive_time = 0
    distance_to_waypoint = verti_dist
    drive_time = abs(float((distance_to_waypoint) / (drive_ticks*scale))) 
    print("#--------------------------------------------------------------------#")
    print("Driving for {:.2f} seconds and distance to waypoint is: {}".format(drive_time, distance_to_waypoint))
    motion_controller([1,0], drive_ticks, drive_time)
    #print("Arrived at pose: [{}], Target waypoint is: [{}]".format(self.get_robot_pose(),self.wp))
    print("#--------------------------------------------------------------------#") 


def turn_to_waypoint(target_angle):
    '''
        Function that implements the turning of the robot
    '''
    datadir = "calibration/param/"
    scale = np.loadtxt("{}scale.txt".format(datadir), delimiter=',')
    baseline = np.loadtxt("{}baseline.txt".format(datadir), delimiter=',')
    turn_ticks = 10
    
    
    # Calculate turning varibles
    turn_time = 0
    #target_angle = math.atan2(wp[1], wp[0]) # angle from robot's current position to the target waypoint
    theta_delta = target_angle  # How far the robot must turn from current pose
    
    if theta_delta > math.pi:
        theta_delta -= 2 * math.pi
    elif theta_delta < -math.pi:
        theta_delta += 2 * math.pi

    # Evaluate how long the robot should turn for
    turn_time = float((abs(theta_delta)*baseline)/(2*turn_ticks*scale))
    print("#------------------------------------------------------------#")
    print("Turning for {:.2f} seconds".format(turn_time))
    print("#------------------------------------------------------------#")
    
    if theta_delta == 0:
        print("No turn")
    elif theta_delta > 0:
        motion_controller([0,1], turn_ticks, turn_time)
    elif theta_delta < 0:
        motion_controller([0,-1], turn_ticks, turn_time)
    else:
        print("There is an issue with turning function")

        
    #print("Taget angle pose is: [{}]".format(target_angle))
    
    
def motion_controller(motion, wheel_ticks, drive_time):
    lv, rv = ppi.set_velocity(motion, tick=wheel_ticks, time=drive_time)
     
        
  



# main loop
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--map", type=str, default='true_map_full.txt')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    ############Path Planning######################
    print("Entering fruits loop")

    while True:        
        verti_dist, theta = input("Enter waypoint (vertical,theta): ").split(",")
        verti_dist, theta = float(verti_dist), -math.radians(float(theta))
        #waypoint = [float(waypoint_y), float(waypoint_x)]
        if abs(verti_dist) <= 1 and abs(theta) < 2*np.pi:
            if abs(theta) > 0.05:
                turn_to_waypoint(theta)
            if abs(verti_dist) > 0.05:
                drive_to_waypoint(verti_dist)
        

sys.exit()
    