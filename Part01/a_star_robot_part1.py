import heapq as hq
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial
import numpy as np

RADIUS_ROBOT = 220 #mm
RADIUS_WHEELS = 33 #mm
WHEEL_DISTANCE = 287 #mm
MAX_RPM = 27.120002
ACTION_SET = np.array([[50,50], [50,0], [0,50], [50,100], [100,50], [100,100], [0,100], [100,0]])
ACTION_NUMBER = list(range(8))
K_0 = RADIUS_WHEELS / WHEEL_DISTANCE
K_R = RADIUS_WHEELS / 2
STEP_TIME = 0.5/60 #seconds
FREQUENCY = 1
RAD_TIME = 2 * np.pi * STEP_TIME

def coordinate_input(name):
    value_provided = False
    while not value_provided:
        print(f'Enter coordinates of the { name } state:\n')
        try:
            input_coord = input(
                f'Provide horizontal,vertical position and orientation separated by comma ( eg: 3,8,-60 ).')
            # print(input_coord)
            coord_process = input_coord.split(',')
            coord_process = [ int(element) for element in coord_process ]
            # user provided more or less elements than allowed
            if not(len(coord_process) > 1 and len(coord_process) <= 3):
                raise Exception('NotExactElements')
            # coordinate is not valid
            # print(coord_process)
            # print(angle_incorrect)
            confirm = input('Confirm coordinate? (y/n): ')
            if confirm == 'y':
                print(f'The coordinate is: { coord_process }')
                return (coord_process[0], coord_process[1], coord_process[2], coord_process[3])
            else:
                print('*****Must write coordinate again****')
                raise Exception('Repeat')
        except ValueError as error:
            print(error)
            print(
                f'Invalid input for the coordinate. Could not convert to integer all values.')
        except Exception as err:
            args = err.args
            if 'NotExactElements' in args:
                print('Coordinate should have exactly two values. Please try again.')
            elif 'CoordsNotValid' in args:
                print('angle not valid. Please try again.')
            else:
                print(err)

def param_robot_input():
    value_provided = False
    while not value_provided:
        print(f'Enter the following robot paramters:\n')
        try:
            input_clearance = input('Enter clearance of the obstacles in mm')
            clearance_robot = float(input_clearance)
            rpm_wheels = input(
                f'Provide Revolutions per Minute(RPM) for both wheels of the robot separated by commas ( eg: 38,60 ).')
            # print(input_coord)
            rpm_process = rpm_wheels.split(',')
            rpm_process = [ int(element) for element in rpm_process if int(element) < MAX_RPM and int(element) > 0]
            # user provided more or less elements than allowed
            if not len(rpm_process) == 2:
                raise Exception('NotExactElements')
            confirm = input('Confirm params? (y/n):')
            if confirm == 'y':
                params_robot = (clearance_robot, rpm_process[0], rpm_process[1] )
                print(f'robot parameters are {params_robot[0]} clearance, {params_robot[1]} RPM are: {rpm_process}')
                return params_robot
            else:
                print('*****Must write parameters again****')
                raise Exception('Repeat')
        except ValueError as error:
            print(error)
            print(
                f'Invalid input for the parameter. Could not convert to integer all values.')
        except Exception as err:
            args = err.args
            if 'NotInRange' in args:
                print(f'Step size not in range. Please write again all parameters')
            if 'NotExactElements' in args:
                print(f'Velocities musst be or 2 and not satisfy RPM range between 0.1 and {MAX_RPM}. Please write again all parameters')
            else:
                print(err)


#*###############NEW STATE STRUCTURE###########
# ?(cost_total, cost_to_come,cost_to_go, index, parent, x_pos, y_pos, angle, RPM_1, RPM_2)
#!BE AWARE OF INDEXING , MORE ELEMENTS MEANS INDEXES WILL CHANGE
#*Create as subfunctions you need for each one of the principal here declared

#TODO: GAURAV
#? USE AS BASE THE FUNCTIONS ON THE PREVIOUS PROJECT!
def check_in_obstacle(state, border):
	"""
	This function checks if a given state is within the obstacle space.

	Args:
		state (tuple): The horizontal and vertical coordinates of the state.
		border (int): The clearance of the obstacles.

	Returns:
		bool: True if the state is within the obstacle space, False otherwise.

	"""
def is_duplicate(node):
	"""
	This function checks if a node is already in the check matrix.

	Args:
		node (tuple): The node to check, as a tuple of its x and y coordinates and its orientation.
	"""

#TODO: NAGA
#? USE AS BASE THE FUNCTIONS ON THE PREVIOUS PROJECT!
def apply_action(state, type_action):
    """THIS IS AN ATTEMPT OF THE FUNCTION MENTIONED IN STEP1"""
    #TODO: CHECK THIS IS ACTUALLY THE CORRECT WAY TO DO THIS
    x_0, y_0, theta_0 = state
    ACTION_RPM = ACTION_SET[type_action] * RPM_USER
    K_1 =  K_0 * (ACTION_RPM[0] - ACTION_RPM[1])
    K_2 = K_R * (ACTION_RPM[0] + ACTION_RPM[1])
    K_3 = K_2 / K_1
    sin_value = np.sin(K_1*np.radians(RAD_TIME) + np.radians(theta_0))
    cos_value = np.cos(K_1*np.radians(RAD_TIME) + np.radians(theta_0))
    x_pos_new = K_3 * sin_value + x_0
    y_pos_new = -K_3 * cos_value + y_0
    theta_new = (K_1 * RAD_TIME) + theta_0
    if theta_new > 180:
        theta_new = theta_new - 360
    if theta_new < -180:
        theta_new = theta_new + 360
    return (x_pos_new, y_pos_new, theta_new)

def action_move(current_node, action):
    #TODO: Check heuristic function maybe euclidean distance will work better now as actino set is diff
    #* uses apply_action, is_duplicate, check_in obstacle
    """
    Args:
        current_node (Node): Node to move

    Returns:
        Node: new Node with new configuration and state
    """

#TODO: JONATHAN
def create_nodes(initial_state, goal_state):
    """Creates the State space of all possible movements until goal state is reached by applying the A* algorithm.

	Args:
			initial_state (array): multi dimensional array 3x3 that describes the initial configuarion of the puzzle
			goal_state (array): multi dimensional array 3x3 that describes the final configuration the algorithm must find.

	Returns:
			str: 'DONE'. The process have ended thus we have a solution in the tree structure generated.
	"""
def generate_path(node):
	"""Generate the path from the initial node to the goal state.

	Args:
		node (Node): Current node to evaluate its parent (previous move done).
	Returns:
		Boolean: True if no more of the path are available
	"""

def robot_commands_path(goal_path):
    """
    This function is what the ROS publisher will provide to the topic to move the simulated turtlebot waffle.
    Converts every (RPM1,RPM2)-> (linear velocity, angular velocity)

    Args:
        goal_path (list): A list of nodes that define the turtle path.

    Returns:
        None
    """

#TODO: GAURAV
def generated_map():
    """
    Creates a blank image with the outer boundary of the arena drawn on it.
    Draws filled rectangles for the initial and goal states, and outlines for them.
    Defines the polygon points for the rotated hexagon and the polygon.
    Draws the rotated hexagon and the filled polygon, and outlines them.
    Returns:
        np.ndarray: The blank image with the arena drawn on it.
    """
    #TODO: test your map with cv2.imshow()

def divide_array(vect_per_frame, arr_nodes):
	"""
	This function is used to divide an array into chunks of a specified size.

	Args:
		vect_per_frame (int): The number of nodes to include in each chunk.
		arr_nodes (list): A list of nodes to divide.

	Returns:
		list: A list of lists, where each sub-list represents a chunk of nodes.

	"""
	arr_size = len(arr_nodes)
	if arr_size <= vect_per_frame:
			return [ arr_nodes ]
	# Calculate the number of full chunks and the size of the remaining chunk
	number_full_slices  = arr_size // vect_per_frame
	remaining_slice = arr_size % vect_per_frame
	# Slice the array into chunks of the nodes per frame
	sliced_chunks = [ arr_nodes[idx*vect_per_frame:(idx+1)*vect_per_frame]
				for idx in range(number_full_slices) ]
	# Remaining nodes into a separate chunk
	if remaining_slice > 0:
		sliced_chunks.append(arr_nodes[number_full_slices*vect_per_frame:])
	return sliced_chunks


#TODO: JONATHAN
#*Call functions for A* Algorithms execution and code additional logic required
INITIAL_STATE = coordinate_input('INITIAL')
GOAL_STATE = coordinate_input('GOAL')
CLEARANCE, RPM_R, RPM_L = param_robot_input()
RPM_USER = np.array([RPM_R, RPM_L])

#GENERAL VARIABLES FOR A*
generated_nodes = []  # open list
generated_nodes_total = []  # for animation of all
visited_nodes = []  # full list node visited
visited_vectors = {} # for animation
goal_path = [] #nodes shortest path
hq.heapify(generated_nodes)

#TODO: NAGA
#? USE AS BASE THE FUNCTIONS ON THE PREVIOUS PROJECT!
#*Call functions for run animation execution and necessary extra code

#*PIPELINE idea
#?1.define variables required
"""1.1 Determine needed structure for the node exploration,this must be provided by the A* algorithm when finished
    in previous project was a dict with keys the parent nodes position and values array of child positions
    What must change now?
    """
result_frames_vectors = []
result_frames_goal = []
#?2. call generate map and store
#?3.declare and do logic for framing process
    #* use divide array to help you with the ratio of curves to show
    #* create frames of node exploration--> adding x number of nodes per frame
    #? a function for draw these curves so it can be used in the goal path frames??
#TODO: GAURAV
#?4.process frames of goal path --> generate CURVE PATH , the path won't have straight lines always
"""Determine if goal path might need a new structure to satisfy this
    """

# ##add extra frames for the end to display more time the final result
# extra_frames = []
# for idx in range(30):
# 	extra_frames.append(result_frames_goal[-1])

# result_frames_total = result_frames_vectors + result_frames_goal + extra_frames
# try:
# 	video = cv2.VideoWriter(
# 				'a_star_diff_drive_part1.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25, (1200, 500))
# 	for frame in result_frames_total:
# 		video.write(frame)
# 	video.release()
# except Exception as err:
#     print('Video FFMEPG Done')

#TODO: JONATHAN
#*TEST EVERYTHING TOGETHER, CHECK ERRORS
