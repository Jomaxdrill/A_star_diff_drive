# A_star_diff_drive
Implementation of the A* algorithm on a Differential Drive (non-holonomic) TurtleBot3 robot
Repository : https://github.com/Jomaxdrill/A_star_diff_drive

## AUTHORS
jcrespo 121028099
saigopal 119484128
ugaurav 120386640

## DEPENDENCIES and PACKAGES
python 3.11.7 or 3.8
(pip installer used)
numpy 1.26.3
opencv-python 4.9.0.80
tqdm 4.66.2

## LIBRARIES
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import os
import time
import heapq as hq
import numpy as np
import math
import cv2
from tqdm import tqdm

## INSTRUCTIONS

# PART1
-Verify you have ROS2 galatic distribution installed and also CMAKE necessary installations

-Install previously the following packages and any additional in your vlinux distribution running on the terminal the command:
	sudo apt install ffmpeg
    sudo apt install python3-colcon-common-extensions

-Install all necessary dependencies and libraries with pip for insrtance. Recommended to run in your own python environment.

-Using linux terminal locate the folder with the python file 'a_star_robot_part1.py' into the Part01 folder.

-Run the command 'python3 a_star_robot_part1.py'

-Follow the instructions provided:
    1.User will be asked to input each coordinate of the initial and goal state separated by commas. For the goal state add by default any angle(third value).
    2.After clearance in mm must be provided and well left and righ velocities of the wheels.

-While program is running information of the node count will display

-The program informs success or not and seconds it took. If success, it will begin animation creation and by finishing it a video file called 'a_star_robot_part1.mp4' which animates the node exploration and optimal path found  is available.

# PART2

-PART 1 generates a txt file called command_set_solutions.txt used for generate the movement of the robot in gazebo

- Locale folder Part02 and build it. In case compiling problems occurs delete recursively the build,install and log folder present and run the command.

```sh
cd ~\Part02
colcon build --packages-select turtlebot3_project3
```
- Source ROS (Package will be identified)

```sh
 source /opt/ros/galactic/setup.bash

```
-Run the gazebo environment

```sh
ros2 launch turtlebot3_project3 competition_world.launch.py
```
- In other terminal locale Part02 folder ,build it and source package to be identified

```sh
cd ~\Part02
colcon build
source install/setup.bash
```

-Run the following script and see what displays in gazebo
```sh
ros2 run turtlebot3_project3 teleop.py
```
# Video links

Part1 : https://drive.google.com/file/d/1q55cuj-UfM1IgOeoE_GgCgJp8SPTNHo7/view?usp=sharing
Part2: https://drive.google.com/file/d/1YvLhn3QDj3CAMU2yVYhUNc5d1eLu7pxU/view?usp=sharing
