import heapq as hq
import numpy as np
import time
import math
import cv2

RADIUS_ROBOT = 220 #mm
RADIUS_WHEELS = 33 #mm
WHEEL_DISTANCE = 287 #mm
MAX_RPM = 17.37972 #RPM
MAX_ROT_SPEED = 1.82 # rad/s
MAX_TRANSLATIONAL_SPEED = 0.26 # m/s
ACTION_NUMBER = list(range(8))
K_0 = RADIUS_WHEELS / WHEEL_DISTANCE
K_R = RADIUS_WHEELS / 2
STEP_TIME = 0.1 #seconds
DUR_ACTION = 1
FREQUENCY = 1
RAD_GOAL = 1.5
WIDTH_SPACE = 6000  # Horizontal dimension of space
HEIGHT_SPACE = 2000  # Vertical dimension of space
MM_TO_CM = 5 #threshold to reduce number of squares to check, every square will be 5x5 cm
TRANS_MATRIX = [ [0,-1,HEIGHT_SPACE],[1,0,0],[0,0,1] ] #from origin coord system to image coord system
TRANS_MATRIX_INV = [[0, 1, 0], [-1, 0, HEIGHT_SPACE], [0, 0, 1]]  #from image coord system origin coord system

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
			confirm = input('Confirm coordinate? (y/n): ')
			if confirm == 'y':
				print(f'The coordinate is: { coord_process }')
				return (coord_process[0], coord_process[1], coord_process[2])
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
			input_clearance = input('Enter clearance of the obstacles in mm\n')
			clearance_robot = float(input_clearance)
			rpm_wheels = input(
				f'Provide Revolutions per Minute(RPM) for both wheels of the robot separated by commas ( eg: 17,17 ).\n')
			# print(input_coord)
			rpm_process = rpm_wheels.split(',')
			#*IN BACKEND THE CODE WILL USE RAD/S INSTEAD OF RPM
			rpm_process = [ (np.pi/60) * float(element) for element in rpm_process if float(element) < MAX_RPM and float(element) > 0]
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
	x_pos, y_pos = state
	tl = border
	# Check if the state is outside of the space
	if x_pos < 0 or y_pos < 0:
		return True
	if x_pos >= WIDTH_SPACE or y_pos >= HEIGHT_SPACE:
		return True
	# Check half-plane equations for each obstacle
	in_obstacle = np.zeros(6, dtype=bool)
	#first obstacle
	in_obstacle_0 = (x_pos >= 1500 - tl) and (x_pos <= 1750 + tl) and (y_pos >= 1000 - tl) and (y_pos <= HEIGHT_SPACE)
	if in_obstacle_0:
		#print(f'first obstacle')
		return True
	#second obstacle
	in_obstacle_1 = (x_pos >= 2500 - tl) and (x_pos <= 2750 + tl) and (y_pos >= 0) and (y_pos <= 1000 + tl)
	if in_obstacle_1:
		#print(f'second obstacle')
		return True
	#third_obstacle- circle
	in_obstacle_2 = (x_pos - 4200)**2 + (y_pos - 1200)**2 <= (600 + tl)**2
	if in_obstacle_2:
		print(f'third obstacle')
		return True
	#border wall 1
	walls_1 = np.zeros(3, dtype=bool)
	walls_1[0] = ( x_pos >= 0 and x_pos <= 1500 - tl ) and (y_pos >= HEIGHT_SPACE-tl and y_pos <= HEIGHT_SPACE)
	walls_1[1] = ( x_pos >= 0 and x_pos <= tl ) and (y_pos >= tl and y_pos <= HEIGHT_SPACE-tl)
	walls_1[2] =  ( x_pos >= 0 and x_pos <= 2500 - tl ) and (y_pos >= 0 and  y_pos <= tl )
	in_obstacle_4 = any(walls_1)
	if in_obstacle_4:
		#print(f'walls left detected')
		return True
	#border wall 2
	walls_2 = np.zeros(3, dtype=bool)
	walls_2[0] = ( x_pos >= 1750 + tl and x_pos <= WIDTH_SPACE ) and (y_pos >= HEIGHT_SPACE-tl and y_pos <= HEIGHT_SPACE)
	walls_2[1] = ( x_pos >= WIDTH_SPACE-tl and x_pos <= WIDTH_SPACE ) and ( y_pos >= tl and y_pos <= HEIGHT_SPACE-tl)
	walls_2[2] =  ( x_pos >= 2750 + tl and x_pos <= WIDTH_SPACE ) and ( y_pos >= 0 and y_pos <= tl )
	in_obstacle_5 = any(walls_2)
	if in_obstacle_5:
		#print(f'walls right detected')
		return True
	return False
def convert_to_matrix_coordinates(state):
	x_pos,y_pos = state
	row = int(x_pos // MM_TO_CM)
	col = int(y_pos // MM_TO_CM)
	return (row, col)
def is_duplicate(state):
	row, col = convert_to_matrix_coordinates(state)
	return check_matrix[row][col] == 1

def to_check_matrix(state):
	row, col = convert_to_matrix_coordinates(state)
	check_matrix[row][col] = 1
	return
#TODO: NAGA
#? USE AS BASE THE FUNCTIONS ON THE PREVIOUS PROJECT!
def calculate_cost(action):
    #consider state as 0,0,0
	time_action = 0
	CT_1 = CT1_ACTION_SET[action]
	CT_2 = CT2_ACTION_SET[action]
	theta_new = 0
	cost = 0
	while time_action < DUR_ACTION:
		time_action += STEP_TIME
		delta_x = K_R * CT_1 * math.cos(theta_new) * STEP_TIME
		delta_y = K_R * CT_1 * math.sin(theta_new) * STEP_TIME
		theta_new += K_0 * CT_2 * STEP_TIME
		cost += math.sqrt((delta_x**2) + (delta_y**2))
	return cost
def apply_action(state, action ,full_moves = False):
	time_action = 0
	CT_1 = CT1_ACTION_SET[action]
	CT_2 = CT2_ACTION_SET[action]
	x_0, y_0, theta_0 = state
	x_pos_new = x_0
	y_pos_new = y_0
	theta_new = np.radians(theta_0)
	while time_action < DUR_ACTION:
		time_action += STEP_TIME
		delta_x = K_R * CT_1 * math.cos(theta_new) * STEP_TIME
		delta_y = K_R * CT_1 * math.sin(theta_new) * STEP_TIME
		theta_new += K_0 * CT_2 * STEP_TIME
		x_pos_new += delta_x
		y_pos_new += delta_y
	theta_new = np.degrees(theta_new)
	return x_pos_new, y_pos_new, theta_new


def action_move(current_node, action):
	#TODO: Check heuristic function maybe euclidean distance will work better now as actino set is diff
	#* uses apply_action, is_duplicate, check_in obstacle
	"""
	Args:
		current_node (Node): Node to move

	Returns:
		Node: new Node with new configuration and state
	"""
	state_moved = apply_action(current_node[5:], action)
	# *check new node is in obstacle space
	if check_in_obstacle(state_moved[0:2], BORDER_TOTAL):
		return None
	# *check by the state duplicate values between the children
	node_already_visited = is_duplicate(state_moved)
	if node_already_visited:
		return None
	#create new node
	new_cost_to_come = current_node[1] + COST_ACTION_SET[action]
	new_cost_to_go = distance(state_moved[0:2], GOAL_STATE[0:2]) if state_moved[0] >= WIDTH_SPACE/2 else 0 #heuristic function
	new_cost_to_go *= (1.0 + 1/1000) #adjustments to heuristic
	new_total_cost =  new_cost_to_come + new_cost_to_go
	new_node = (new_total_cost, new_cost_to_come, new_cost_to_go) + (-1, current_node[3]) + state_moved
	return new_node

def get_vector(node_a, node_b):
	"""
	This function returns the vector from node_a to node_b.

	Args:
		node_a (tuple): The first node.
	"""
	return tuple(x - y for x, y in zip(node_a, node_b))

def distance(node_a, node_b):
	"""
	Returns the Euclidean distance between two nodes.

	Args:
		node_a (tuple): The first node.
		node_b (tuple): The second node.

	Returns:
		float: The Euclidean distance between the two nodes.

	"""
	substract_vector = get_vector(node_a, node_b)
	#? Euclidean distance has given better performance
	return math.sqrt(substract_vector[0]**2 + substract_vector[1]**2)
#TODO: JONATHAN
def get_vector(node_a, node_b):
	"""
	This function returns the vector from node_a to node_b.

	Args:
		node_a (tuple): The first node.
	"""
	return tuple(x - y for x, y in zip(node_a, node_b))
# def euclidean_distance(node_a, node_b):
# 	return np.linalg.norm(get_vector(node_a, node_b))
# ?(cost_total, cost_to_come,cost_to_go, index, parent, x_pos, y_pos, angle, RPM_1, RPM_2)
def create_nodes(initial_state, goal_state):
	"""Creates the State space of all possible movements until goal state is reached by applying the A* algorithm.

	Args:
			initial_state (array): multi dimensional array 3x3 that describes the initial configuarion of the puzzle
			goal_state (array): multi dimensional array 3x3 that describes the final configuration the algorithm must find.

	Returns:
			str: 'DONE'. The process have ended thus we have a solution in the tree structure generated.
	"""
	# Start the timer
	start_time = time.time()
	goal_reached = False
	counter_nodes = 0
	distance_init_goal = distance(initial_state[0:2], goal_state[0:2])
	cost_init = (distance_init_goal, 0, distance_init_goal)
	initial_node = cost_init + (0, None) + initial_state
	# Add initial node to the heap
	hq.heappush(generated_nodes, initial_node)
	while not goal_reached and len(generated_nodes) and not counter_nodes > 100000:
		print(counter_nodes)
		current_node = generated_nodes[0]
		hq.heappop(generated_nodes)
		# Mark node as visited
		visited_nodes.append(current_node)
		#for check duplicates
		#*add_to_check_matrix(current_node[5:8])
		#*visited_vectors[current_node[5:7]] = []
		# Check if popup_node is goal state
		goal_reached = distance(current_node[5:], goal_state) < RAD_GOAL
		if goal_reached:
			goal_reached = True
			end_time = time.time()
			return f'DONE in {end_time-start_time} seconds.'
		# Apply action set to node to get new states/children
		for action in range(len(ACTION_SET)):
			child = action_move(current_node, action)
			# If movement was not possible, ignore it
			if not child:
				continue
			#*visited_vectors[current_node[5:7]].append(child[5:7])
			# Check if child is in open list generated nodes
			where_is_node = 0
			is_in_open_list = False
			for node in generated_nodes:
				if node[5:7] == child[5:7]:
					is_in_open_list = True
					break
				where_is_node += 1
			if not is_in_open_list:
				counter_nodes += 1
				child_to_enter = child[0:3] + (counter_nodes,) + child[4:]
				hq.heappush(generated_nodes, child_to_enter)
			# check if cost to come is greater in node in open list
			elif generated_nodes[where_is_node][1] > child[1]:
				# Update parent node and cost of this node in the generated nodes heap
				current_index = generated_nodes[where_is_node][3]
				generated_nodes[where_is_node] = child[0:3] + (current_index,) + child[4:]
	end_time = time.time()
	print(f'No solution found. Process took {end_time-start_time} seconds.')
	return None
def generate_path(node):
	"""Generate the path from the initial node to the goal state.

	Args:
		node (Node): Current node to evaluate its parent (previous move done).
	Returns:
		Boolean: True if no more of the path are available
	"""
	while node is not None:
		goal_path.append(node[5:])
		parent_at = 0
		for node_check in visited_nodes:
			if node_check[3] == node[4]:
				break
			parent_at += 1
		node = visited_nodes[parent_at] if parent_at < len(visited_nodes) else None
	return True

def robot_commands_path(goal_path):
	"""
	This function is what the ROS publisher will provide to the topic to move the simulated turtlebot waffle.
	Converts every (RPM1,RPM2)-> (linear velocity, angular velocity)

	Args:
		goal_path (list): A list of nodes that define the turtle path.

	Returns:
		None
	"""
	commands_robot = []
	for state in goal_path:
		sin_value = np.sin(np.radians(state[7]))
		cos_value = np.cos(np.radians(state[7]))
		#compute jacobian matrix
		jacobian_matrix = [[K_R*cos_value,K_R*cos_value],
							[K_R*sin_value,K_R*sin_value],
							[K_0, -K_0]
						]
		x_dot, y_dot, theta_dot = np.dot(jacobian_matrix, np.array([state[8], state[9]]))
		#turtlebot accepts m/s and rad/s
		commands_robot.append(tuple(x_dot/1000, y_dot/1000, theta_dot))
	return commands_robot[1:-1] #ignore the start position and goal position

def store_command_path(commands_robot):
	"""Stores the actions in order which solves the puzzle for the initial and goal state given by the user.
	"""
	file_nodes_path = open('../Part02/src/turtlebot3_project3/scripts/command_set_solution.txt', "w")
	for command in commands_robot[::-1]:
		command_sequence = ",".join(command)
		file_nodes_path.write(f'{ command_sequence }\n')
	file_nodes_path.close()
	return

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


def coordinate_plane(coord):
	"""
	This function takes a coord of matrix and return the coordinate in the bottom left corner of the plane
		coord (tuple): The state of the robot, as a tuple of its x and y coordinates.

	Returns:
		tuple: The row and column coordinates of the state in the transformed image.

	"""
	row,col = coord
	x_pos, y_pos, _ = np.dot(TRANS_MATRIX_INV, (row, col, 1))
	return x_pos, y_pos


#TODO: JONATHAN
#*Call functions for A* Algorithms execution and code additional logic required
INITIAL_STATE = coordinate_input('INITIAL')
GOAL_STATE = coordinate_input('GOAL')
print(INITIAL_STATE, GOAL_STATE)
CLEARANCE, RPM_L, RPM_R = param_robot_input()
BORDER_TOTAL = RADIUS_ROBOT + CLEARANCE
ACTION_SET = [ [0, RPM_L], [RPM_L, 0], [RPM_L, RPM_L], [0, RPM_R],
				[RPM_R, 0], [RPM_R, RPM_R], [RPM_L, RPM_R], [RPM_R, RPM_L] ]
CT1_ACTION_SET = [ vel[0] + vel[1] for vel in ACTION_SET]
CT2_ACTION_SET = [ vel[1] - vel[0] for vel in ACTION_SET]
COST_ACTION_SET = [ calculate_cost(action) for action in range(len(ACTION_SET)) ]
print(INITIAL_STATE, GOAL_STATE)
verify_initial_position = check_in_obstacle(INITIAL_STATE[0:2], BORDER_TOTAL)
verify_goal_position = check_in_obstacle(GOAL_STATE[0:2], BORDER_TOTAL)
if verify_initial_position:
	print("START HITS OBSTACLE!! Please run the program again.")
	exit(0)
if verify_goal_position:
	print("GOAL HITS OBSTACLE!! Please run the program again.")
	exit(0)
check_matrix = np.zeros((HEIGHT_SPACE // MM_TO_CM, WIDTH_SPACE // MM_TO_CM))
matrix_check_obstacle = np.zeros((HEIGHT_SPACE, WIDTH_SPACE))
for row in tqdm(range(HEIGHT_SPACE), desc ="Setting obstacle space..."):
	for col in range(WIDTH_SPACE):
		x_pos, y_pos = coordinate_plane((row,col))
		if check_in_obstacle((x_pos, y_pos),BORDER_TOTAL):
			matrix_check_obstacle[row][col] = 1
#GENERAL VARIABLES FOR A*
generated_nodes = []  # open list
generated_nodes_total = []  # for animation of all
visited_nodes = []  # full list node visited
visited_vectors = {} # for animation
goal_path = [] #nodes shortest path
hq.heapify(generated_nodes)

# solution = create_nodes(INITIAL_STATE, GOAL_STATE)
# print(solution)
# if not solution:
# 	exit(0)
# generate_path(visited_nodes[1])
# turtlebot3_commands = robot_commands_path(goal_path)
# #generate a file that node publisher will read to send info to simulation via the topic
# file_velocities_path = open("velocitiesPath.txt", "w")
# for command in turtlebot3_commands[::-1]:
# 	command_sequence = ",".join(command)
# 	file_velocities_path.write(f'{ command_sequence }\n')
# file_velocities_path.close()
# #TODO: NAGA
# #? USE AS BASE THE FUNCTIONS ON THE PREVIOUS PROJECT!
# #*Call functions for run animation execution and necessary extra code

# #*PIPELINE idea
# #?1.define variables required
# """1.1 Determine needed structure for the node exploration,this must be provided by the A* algorithm when finished
# 	in previous project was a dict with keys the parent nodes position and values array of child positions
# 	What must change now?
# 	"""
# result_frames_vectors = []
# result_frames_goal = []
# #?2. call generate map and store
# #?3.declare and do logic for framing process
# 	#* use divide array to help you with the ratio of curves to show
# 	#* create frames of node exploration--> adding x number of nodes per frame
# 	#? a function for draw these curves so it can be used in the goal path frames??
# #TODO: GAURAV
# #?4.process frames of goal path --> generate CURVE PATH , the path won't have straight lines always
# """Determine if goal path might need a new structure to satisfy this
# 	"""

# # ##add extra frames for the end to display more time the final result
# # extra_frames = []
# # for idx in range(30):
# # 	extra_frames.append(result_frames_goal[-1])

# # result_frames_total = result_frames_vectors + result_frames_goal + extra_frames
# # try:
# # 	video = cv2.VideoWriter(
# # 				'a_star_diff_drive_part1.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25, (1200, 500))
# # 	for frame in result_frames_total:
# # 		video.write(frame)
# # 	video.release()
# # except Exception as err:
# #     print('Video FFMEPG Done')

# #TODO: JONATHAN
# #*TEST EVERYTHING TOGETHER, CHECK ERRORS
