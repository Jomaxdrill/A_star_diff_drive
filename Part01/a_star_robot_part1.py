import heapq as hq
import numpy as np
import time
import math
import cv2
from tqdm import tqdm

#*robot specifications for turtlebot waffle
RADIUS_ROBOT = 220 #mm
RADIUS_WHEELS = 33 #mm
WHEEL_DISTANCE = 287 #mm
MAX_ROT_SPEED = 1.82 # rad/s
MAX_LIN_SPEED = 0.26 # m/s
K_0 = RADIUS_WHEELS / WHEEL_DISTANCE
K_R = RADIUS_WHEELS / 2 #mm
#*threshold for actions and general dimensions
STEP_TIME = 0.1 #seconds
DUR_ACTION = 1 #seconds
RADIUS_GOAL = 100 #mm
WIDTH_SPACE = 6000  # Horizontal dimension of space
HEIGHT_SPACE = 2000  # Vertical dimension of space
MM_TO_CM = 50 #threshold to reduce number of squares to check, every square will be these dimensions
ANGLE_DIV = 24 #threshhold angle to reduce the possible orientations
ANGLE_MIN = 360 // ANGLE_DIV
#*change of coordinates
TRANS_MATRIX = [ [0,-1, HEIGHT_SPACE],[1, 0, 0],[0, 0, 1] ] #from origin coord system to image coord system
TRANS_MATRIX_INV = [[0, 1, 0], [-1, 0, HEIGHT_SPACE], [0, 0, 1]]  #from image coord system origin coord system
#*animation
COLOR_OBSTACLE = (255, 0, 0)
COLOR_CURVES = (0, 255, 0)
COLOR_GOAL = (0, 0, 255)
COLOR_BORDER = (255, 255, 255)
COLOR_RADIUS_GOAL = (255, 255 ,0)
COLOR_START = (255, 0, 255)
COLOR_END = (0, 255, 255)
FPS = 25
RESIZE_RATIO = 2
RESIZE_WIDTH = round(WIDTH_SPACE / RESIZE_RATIO)
RESIZE_HEIGHT = round(HEIGHT_SPACE / RESIZE_RATIO)
MAX_FRAME_NODES = 250 #*when frames surpass 300 it can fail the rendering
MAX_GOAL_FRAMES = 80 #*set also limit for frames of goal
NODES_PER_FRAME = 1000
LINES_PER_FRAME = 200
PATH_COMMANDS = '../Part02/src/turtlebot3_project3/scripts/command_set_solution.txt'
def coordinate_input(name):
	"""
	This function is used to input the coordinates of a given state.

	Args:
		name (str): The name of the state.

	Returns:
		tuple: A tuple containing the horizontal and vertical coordinates of the state.

	Raises:
		ValueError: If the input is not a valid number.
		Exception: If the input is not in the expected format.

	"""
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
	"""
	This function is used to input the robot parameters such as the clearance of the obstacles and the RPM of the wheels.

	Args:
		None

	Returns:
		params_robot (tuple): A tuple containing the clearance of the obstacles and the RPM of the left and right wheels.

	Raises:
		ValueError: If the input is not a valid number.
		Exception: If the input is not in the expected format.

	"""
	value_provided = False
	while not value_provided:
		print(f'Enter the following robot paramters:\n')
		try:
			input_clearance = input('Enter clearance of the obstacles in mm\n')
			clearance_robot = float(input_clearance)
			rpm_wheels = input(
				f'Provide Revolutions per Minute(RPM) for left and right wheels of the robot separated by commas ( eg: 50,100 ).\n')
			# print(input_coord)
			rpm_process = rpm_wheels.split(',')
			#*IN BACKEND THE CODE WILL USE RAD/S INSTEAD OF RPM
			rpm_process_to_rad_s = [ (np.pi/60) * float(element) for element in rpm_process ]
			# user provided more or less elements than allowed
			if not len(rpm_process_to_rad_s) == 2:
				raise Exception('NotExactElements')
			if rpm_process_to_rad_s[0] == rpm_process_to_rad_s[1]:
				raise Exception('EqualsElements')
			confirm = input('Confirm params? (y/n):')
			if confirm == 'y':
				params_robot = (clearance_robot, rpm_process_to_rad_s[0], rpm_process_to_rad_s[1] )
				print(f'robot parameters are {params_robot[0]} clearance, RPM are: left {rpm_process[0]}, right {rpm_process[1]}')
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
				print(f'Velocities must be or 2 and not equal. Please write again all parameters')
			if 'EqualsElements' in args:
				print(f'Velocities must not be equal. Please write again all parameters')
			else:
				print(err)


#*###############NEW STATE STRUCTURE###########
# ?(cost_total, cost_to_come,cost_to_go, index, parent, x_pos, y_pos, angle, action)

def check_in_obstacle(state, border):
	"""
	This function checks if a given state is within the obstacle space.

	Args:
		state (tuple): The horizontal and vertical coordinates of the state.
		border (int): The clearance of the obstacles.

	Returns:
		bool: True if the state is within the obstacle space, False otherwise.

	"""
	sc = 1
	tl = border/sc
	x_pos, y_pos = state
	x_pos = x_pos/sc
	y_pos = y_pos/sc
	# Check if the state is outside of the space
	if x_pos < 0 or y_pos < 0:
		return True
	if x_pos >= WIDTH_SPACE/sc or y_pos >= HEIGHT_SPACE/sc:
		return True
	#first obstacle
	in_obstacle_0 = (x_pos >= 1500/sc - tl) and (x_pos <= 1750/sc + tl) and (y_pos >= 1000/sc - tl) and (y_pos <= HEIGHT_SPACE/sc)
	if in_obstacle_0:
		#print(f'first obstacle')
		return True
	#second obstacle
	in_obstacle_1 = (x_pos >= 2500/sc - tl) and (x_pos <= 2750/sc + tl) and (y_pos/sc >= 0) and (y_pos <= 1000/sc + tl)
	if in_obstacle_1:
		#print(f'second obstacle')
		return True
	#third_obstacle- circle
	in_obstacle_2 = (x_pos - 4200/sc)**2 + (y_pos - 1200/sc)**2 <= (600/sc + tl)**2
	if in_obstacle_2:
		#print(f'third obstacle')
		return True
	#border wall 1
	walls_1 = np.zeros(3, dtype=bool)
	walls_1[0] = ( x_pos >= 0 and x_pos <= 1500/sc - tl ) and (y_pos >= HEIGHT_SPACE/sc - tl and y_pos <= HEIGHT_SPACE/sc)
	walls_1[1] = ( x_pos >= 0 and x_pos <= tl ) and (y_pos >= tl and y_pos <= HEIGHT_SPACE/sc - tl)
	walls_1[2] =  ( x_pos >= 0 and x_pos <= 2500/sc - tl ) and (y_pos >= 0 and  y_pos <= tl )
	in_obstacle_4 = any(walls_1)
	if in_obstacle_4:
		#print(f'walls left detected')
		return True
	#border wall 2
	walls_2 = np.zeros(3, dtype=bool)
	walls_2[0] = ( x_pos >= 1750/sc + tl and x_pos <= WIDTH_SPACE/sc ) and (y_pos >= HEIGHT_SPACE/sc - tl and y_pos <= HEIGHT_SPACE/sc)
	walls_2[1] = ( x_pos >= WIDTH_SPACE/sc - tl and x_pos <= WIDTH_SPACE/sc ) and ( y_pos >= tl and y_pos <= HEIGHT_SPACE/sc - tl)
	walls_2[2] =  ( x_pos >= 2750/sc + tl and x_pos <= WIDTH_SPACE/sc ) and ( y_pos >= 0 and y_pos <= tl )
	in_obstacle_5 = any(walls_2)
	if in_obstacle_5:
		#print(f'walls right detected')
		return True
	return False
def convert_to_matrix_coordinates(state):
	"""
	This function converts a state from cartesian coordinates to matrix coordinates.

	Args:
		state (tuple): A tuple containing the x and y coordinates and the angle of the state.

	Returns:
		tuple: A tuple containing the row, column, and angle index of the state in the matrix.

	"""
	x_pos,y_pos, angle = state
	row = round(x_pos / MM_TO_CM)
	col = round(y_pos / MM_TO_CM)
	angle_index = round(angle / ANGLE_MIN)
	return (row, col, angle_index)
def is_duplicate(state):
	"""
	This function checks if a given state is a duplicate of a previously visited state.

	Args:
		state (tuple): The state to check for duplicates.

	Returns:
		bool: True if the state is a duplicate, False otherwise.
	"""
	row, col, angle_index = convert_to_matrix_coordinates(state)
	return check_matrix[row, col, angle_index] == 1

def add_to_check_matrix(state):
	"""
	This function adds the given state to the obstacle check matrix.
	It marks the square in the matrix that corresponds to the given state as occupied.
	This function is used to prevent the robot from visiting the same state multiple times.

	Args:
		state (tuple): The state to add to the check matrix.
	"""
	row, col, angle_index = convert_to_matrix_coordinates(state)
	#check_matrix[row, col, angle_index] = 1
	#check_matrix[row, col, angle_index + 1] = 1
	#check_matrix[row, col, angle_index - 1] = 1
	check_matrix[row, col, :] = 1
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
	return round(cost)
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
	#normalize and set angle always between -180,180
	theta_new %= 360
	if theta_new > 180:
		theta_new  -= 360
	elif theta_new < -180:
		theta_new += 360
	#rounds to nearest angle of the possible according to threshold angle
	theta_new = (theta_new // ANGLE_MIN ) * ANGLE_MIN
	return round(x_pos_new), round(y_pos_new), round(theta_new)


def action_move(current_node, action):
	#TODO: Check heuristic function maybe euclidean distance will work better now as actino set is diff
	#* uses apply_action, is_duplicate, check_in obstacle
	"""
	Args:
		current_node (Node): Node to move

	Returns:
		Node: new Node with new configuration and state
	"""
	state_moved = apply_action(current_node[5:8], action)
	#*check when actions could inmediately be out of bounds
	if (state_moved[0] <= 0 or state_moved[1] <= 0) or (state_moved[0] >= WIDTH_SPACE or state_moved[1] >= HEIGHT_SPACE):
		return None
	# *check new node is in obstacle space
	if check_in_obstacle(state_moved[0:2], BORDER_TOTAL):
		return None
	# *check by the state duplicate values between the children
	node_already_visited = is_duplicate(state_moved)
	if node_already_visited:
		return None
	#create new node
	new_cost_to_come = current_node[1] + COST_ACTION_SET[action]
	new_cost_to_go = distance(state_moved[0:2], GOAL_STATE[0:2]) #if state_moved[0] >= WIDTH_SPACE/2 else 0 #heuristic function
	new_total_cost =  new_cost_to_come + new_cost_to_go
	new_node = (new_total_cost, new_cost_to_come, new_cost_to_go) + (-1, current_node[3]) + state_moved + (action,)
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
	return round(math.sqrt(substract_vector[0]**2 + substract_vector[1]**2))
#TODO: JONATHAN
def get_vector(node_a, node_b):
	"""
	This function returns the vector from node_a to node_b.

	Args:
		node_a (tuple): The first node.
	"""
	return tuple(x - y for x, y in zip(node_a, node_b))

# ?(cost_total, cost_to_come,cost_to_go, index, parent, x_pos, y_pos, angle, action)
#*the action is the one who created the node via the parent
def create_nodes(initial_state, goal_state):
	"""Creates the State space of all possible movements until goal state is reached by applying the A* algorithm.

	Args:
			initial_state (tuple): multi dimensional tuple 3x3 that describes the initial configuarion of the puzzle
			goal_state (tuple): multi dimensional tuple 3x3 that describes the final configuration the algorithm must find.

	Returns:
			str: 'DONE'. The process have ended thus we have a solution in the tree structure generated.
			str: 'No solution'. The process ended without finding a solution all the tree was traversed.
	"""
	# Start the timer
	start_time = time.time()
	goal_reached = False
	counter_nodes = 0
	distance_init_goal = distance(initial_state[0:2], goal_state[0:2])
	cost_init = (distance_init_goal, 0, distance_init_goal)
	initial_node = cost_init + (0, None) + initial_state + (None,)
	generated_nodes_indexing[initial_state] = (0, counter_nodes)
	# Add initial node to the heap
	hq.heappush(generated_nodes, initial_node)
	while not goal_reached and len(generated_nodes):
		print(counter_nodes, end="\r")
		current_node = generated_nodes[0]
		hq.heappop(generated_nodes)
		# Mark node as visited
		visited_nodes.append(current_node)
		#for check duplicates
		add_to_check_matrix(current_node[5:8])
		visited_curves[current_node[5:8]] = []
		# Check if popup_node is goal state
		goal_reached = distance(current_node[5:8], goal_state[0:2]) < RADIUS_GOAL
		if goal_reached:
			goal_reached = True
			end_time = time.time()
			return f'DONE in {end_time-start_time} seconds.'
		# Apply action set to node to get new states/children
		for action in ACTION_NUMBER:
			child = action_move(current_node, action)
			# If movement was not possible, ignore it
			if not child:
				continue
			visited_curves[current_node[5:8]].append(action)
			# Check if child is in open list generated nodes
			in_open_list = generated_nodes_indexing.get(child[5:8], None)
			if not in_open_list:
				counter_nodes += 1
				child_to_enter = child[0:3] + (counter_nodes,) + child[4:]
				generated_nodes_indexing[child[5:8]] = (child[1], counter_nodes)
			# check if cost to come is greater in node in open list
			elif in_open_list[0] > child[1]:
				# Update parent node and cost of this node in the generated nodes heap
				#create new node with lowest cost to unqueue fast and achieve it
				child_to_enter = (0,) + child[1:3] + (in_open_list[1],) + child[4:]
			hq.heappush(generated_nodes, child_to_enter)
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
	goal_path_action.append((node[5:8], node[8])) #(parent, action) first
	while node is not None:
		goal_path.append(node[5:8])
		parent_at = 0
		for node_check in visited_nodes:
			if node_check[3] == node[4]:
				break
			parent_at += 1
		if parent_at < len(visited_nodes):
			node = visited_nodes[parent_at]
			goal_path_action.append((node[5:8], node[8])) #(parent, action)
		else:
			node = None
	return True

def robot_commands_path():
	"""
	This function is what the ROS publisher will provide to the topic to move the simulated turtlebot waffle.
	Converts every known origin and action into a set of velocities-> (linear velocity, angular velocity)

	Args:
		goal_path_action (list): A list of nodes referenced by parent and the action it did to generate it.

	Returns:
		None
	"""
	commands_robot = []
	reverse_path_action = goal_path_action[::-1]
	for idx in tqdm(range(len(reverse_path_action)-1), desc= 'Generating commands for gazebo simulation'):
		node_parent,_ = reverse_path_action[idx]
		_, action = reverse_path_action[idx+1]
		#vel_curve = generate_curve_velocities(node_parent, action)
		CT_1 = CT1_ACTION_SET[action]
		CT_2 = CT2_ACTION_SET[action]
		linear_vel = K_R * CT_1
		angular_vel = K_0 * CT_2
		vel_curve = [(linear_vel/1000, angular_vel)]
		for vel in vel_curve:
			commands_robot.append(vel)
	commands_robot.append((0, 0)) #SEND THE STOP COMMAND
	return commands_robot #ignore the start position

def store_command_path(commands_robot, path):
	"""Stores the actions in order which solves the puzzle for the initial and goal state given by the user.
	"""
	file_nodes_path = open(path, "w")
	for command in commands_robot:
		command = [ str(round(x,6)) for x in command ]
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
	print("Generating map...")
	# Create a blank image
	canvas = np.zeros((HEIGHT_SPACE, WIDTH_SPACE, 3), dtype="uint8")
	# Draw the outer boundary
	cv2.rectangle(canvas, (0, 0), (WIDTH_SPACE, HEIGHT_SPACE),COLOR_BORDER, int(CLEARANCE)*2)

	# Upper Rectangle
	draw_rectangle(canvas, (1500, 0), (1750, 1000), COLOR_OBSTACLE, int(CLEARANCE))

	# Lower Rectangle
	draw_rectangle(canvas, (2500, 2000), (2750, 1000), COLOR_OBSTACLE, int(CLEARANCE))

	# Circle
	draw_circle(canvas, (4200, 800), 600, COLOR_OBSTACLE, int(CLEARANCE))

	#draw initial and goal state points and  radius goal
	y_init, x_init = coordinate_image(INITIAL_STATE[0:2])
	y_goal, x_goal = coordinate_image(GOAL_STATE[0:2])
	cv2.circle(canvas, (x_init,y_init) , 10, COLOR_START, thickness=-1)
	cv2.circle(canvas, (x_goal,y_goal) , 10, COLOR_END, thickness=-1)
	cv2.circle(canvas, (x_goal,y_goal), RADIUS_GOAL , COLOR_RADIUS_GOAL, 4)

	return canvas

def draw_rectangle(canvas, pt1, pt2, color, border):
	# Draw filled rectangle
	cv2.rectangle(canvas, pt1, pt2, color, -1)
	# Draw outline
	cv2.rectangle(canvas, pt1, pt2, COLOR_BORDER, border)

def draw_circle(canvas, center, radius, color, border):
	# Draw filled circle
	cv2.circle(canvas, center, radius, color, -1)
	# Draw outline
	cv2.circle(canvas, center, radius, COLOR_BORDER, border)

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

def generate_curve_velocities(state, action):
	"""
	This function generates a list of velocities for a robot to follow a curve based on the current state and action.

	Args:
		state (tuple): The current state of the robot, as a tuple of its x and y coordinates and its orientation.
		action (int): The action to take, which corresponds to an index in the ACTION_SET list.

	Returns:
		list: A list of velocities, where each element is a tuple of linear velocity (in meters per second) and angular velocity (in radians per second).

	"""
	if action is None:
		return [ state ]
	velocities = []
	time_action = 0
	CT_1 = CT1_ACTION_SET[action]
	CT_2 = CT2_ACTION_SET[action]
	while time_action < DUR_ACTION:
		time_action += STEP_TIME
		linear_vel = K_R * CT_1
		angular_vel = K_0 * CT_2
		velocities.append((linear_vel/1000, angular_vel)) #m/s ,rad/s in STEP TIME seconds
	return velocities
def generate_curve_action(state, action):
	"""
	This function generates a list of points for a robot to follow a curve based on the current state and action.

	Args:
		state (tuple): The current state of the robot, as a tuple of its x and y coordinates and its orientation.
		action (int): The action to take, which corresponds to an index in the ACTION_SET list.

	Returns:
		list: A list of points, where each element is a tuple of x and y coordinates.

	"""
	if action is None:
		return [ state ]
	movements = []
	time_action = 0
	CT_1 = CT1_ACTION_SET[action]
	CT_2 = CT2_ACTION_SET[action]
	x_0, y_0, theta_0 = state
	x_pos_new = x_0
	y_pos_new = y_0
	theta_new = np.radians(theta_0)
	movements.append(state)
	while time_action < DUR_ACTION:
		time_action += STEP_TIME
		delta_x = K_R * CT_1 * math.cos(theta_new) * STEP_TIME
		delta_y = K_R * CT_1 * math.sin(theta_new) * STEP_TIME
		theta_new += K_0 * CT_2 * STEP_TIME
		x_pos_new += delta_x
		y_pos_new += delta_y
		#when element is last we are going to round to the proper value when algorithm computed
		if time_action > DUR_ACTION:
			#normalize and set angle always between -180,180
			theta_new %= 360
			if theta_new > 180:
				theta_new  -= 360
			elif theta_new < -180:
				theta_new += 360
			#rounds to nearest angle of the possible according to threshold angle
			theta_new = (theta_new // ANGLE_DIV ) * ANGLE_DIV
			movements.append((round(x_pos_new), round(y_pos_new), round(theta_new)))
		else:
			movements.append((x_pos_new, y_pos_new, theta_new))
	return movements

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

def coordinate_image(state):
	"""
	This function takes a state as input and returns the corresponding row and column for an image
	Args:
		state (tuple): The state of the robot, as a tuple of its x and y coordinates.

	Returns:
		tuple: The row and column coordinates of the state in the transformed image.

	"""
	x_pos, y_pos = state
	row, col, _ = np.dot(TRANS_MATRIX, (x_pos, y_pos, 1))
	return int(row),int(col)

def resize_frames(frames, name):
	final_resized_frames = []
	for frame in tqdm(frames,desc = f'Resizing {name}'):
		resized_frame = cv2.resize(frame,(RESIZE_WIDTH, RESIZE_HEIGHT),interpolation= cv2.INTER_LINEAR)
		final_resized_frames.append(resized_frame)
	return final_resized_frames




if __name__ == '__main__':
#*Call functions for A* Algorithms execution and code additional logic required
	INITIAL_STATE = coordinate_input('INITIAL')
	GOAL_STATE = coordinate_input('GOAL')
	print(INITIAL_STATE, GOAL_STATE)
	CLEARANCE, RPM_L, RPM_R = param_robot_input()
	BORDER_TOTAL = RADIUS_ROBOT + CLEARANCE
	ACTION_SET = [ [0, RPM_L], [RPM_L, 0], [RPM_L, RPM_L], [0, RPM_R],
					[RPM_R, 0], [RPM_R, RPM_R], [RPM_L, RPM_R], [RPM_R, RPM_L] ]
	ACTION_NUMBER = list(range(len(ACTION_SET)))
	CT1_ACTION_SET = [ vel[0] + vel[1] for vel in ACTION_SET]
	CT2_ACTION_SET = [ vel[1] - vel[0] for vel in ACTION_SET]
	COST_ACTION_SET = [ calculate_cost(action) for action in ACTION_NUMBER ]
	verify_initial_position = check_in_obstacle(INITIAL_STATE[0:2], BORDER_TOTAL)
	verify_goal_position = check_in_obstacle(GOAL_STATE[0:2], BORDER_TOTAL)
	#verifiy positions are valid
	print('Checking parameters....')
	if verify_initial_position:
		print("START HITS OBSTACLE!! Please run the program again.")
		exit(0)
	if verify_goal_position:
		print("GOAL HITS OBSTACLE!! Please run the program again.")
		exit(0)
	#verify max linear and rotational velocity don't exceed values
	check_max_lin_vel = np.max( K_R *np.array(CT1_ACTION_SET) )/1000 > MAX_LIN_SPEED #m/s
	check_max_rot_vel = np.max( K_0 *np.array(CT2_ACTION_SET) ) > MAX_ROT_SPEED #rad/s
	if check_max_lin_vel:
		print(f'max RPM velocities of wheels result in a greater linear velocity than de default max of {MAX_LIN_SPEED} m/s\n')
		print("Please run the program again.")
		exit(0)
	if check_max_rot_vel:
		print(f'max RPM velocities of wheels result in a greater linear velocity than de default max of {MAX_ROT_SPEED} rad/s\n')
		print("Please run the program again.")
		exit(0)
	check_matrix = np.zeros((WIDTH_SPACE // MM_TO_CM, HEIGHT_SPACE // MM_TO_CM, ANGLE_DIV))
	#GENERAL VARIABLES FOR A*
	generated_nodes = []  # open list
	generated_nodes_indexing = {} #to track indexing as open list changes
	visited_nodes = []  # full list node visited
	visited_curves = {} # for animation
	goal_path = [] #nodes shortest path
	goal_path_action = [] #parent-action set for create the proper curve
	hq.heapify(generated_nodes)
	print('begin A* algorithm...\n')
	solution = create_nodes(INITIAL_STATE, GOAL_STATE)
	print(solution)
	print('Generating path....\n')
	generate_path(visited_nodes[-1])

	print('Generating command sequence:....\n')
	turtlebot3_commands = robot_commands_path()
	#generate a file that node publisher will read to send info to simulation via the topic
	store_command_path(turtlebot3_commands, PATH_COMMANDS)
	#obtain curve points from the initial to goal
	total_points_goal_path = []
	reverse_path_action = goal_path_action[::-1]
	for idx in tqdm(range(len(reverse_path_action)-1), desc= 'Generating curve goal path'):
		node_parent,_ = reverse_path_action[idx]
		_, action = reverse_path_action[idx+1]
		points_curve = generate_curve_action(node_parent, action)
		#last point is first of next curve
		total_points_goal_path.append(points_curve[:-1])
	#flat the total points list from before goal to initial state
	all_goal_points = []
	for inner_points in total_points_goal_path:
		all_goal_points.extend(inner_points)
	#add goal as last value
	all_goal_points.append(visited_nodes[-1][5:8])
	result_frames_vectors = []
	result_frames_goal = []
	print('Creating animation:...........n')
	time.sleep(2)
	#create space
	arena = generated_map()
	#begin the frame creation process
	result_frames_vectors.append(arena)
	vectors_curves = list(visited_curves.keys())
	ratio_nodes_per_frame = len(vectors_curves) // MAX_FRAME_NODES
	ratio_nodes_per_frame = ratio_nodes_per_frame if ratio_nodes_per_frame > NODES_PER_FRAME else NODES_PER_FRAME
	print(f'Curves per frame will be: {ratio_nodes_per_frame}')
	curves_per_frame = divide_array(ratio_nodes_per_frame, vectors_curves)
	#create the frames for the vectors
	for set_curves in tqdm(curves_per_frame, desc = 'Animating exploration space'):
		plotted_curves = result_frames_vectors[-1].copy()
		#visit every parent start coordinate
		for start in set_curves:
			#get all the valid actions it performed
			action_applied = visited_curves[start]
			#generate the curve for the action and draw it on image
			for action in action_applied:
				points_curve = generate_curve_action(start, action)
				image_points_curve = np.array([ coordinate_image(point[0:2]) for point in points_curve],np.int32)
				#create an array of pair of points representing the curve
				image_lines_curve = []
				for idx in range(len(image_points_curve)-1):
					image_lines_curve.append([image_points_curve[idx],image_points_curve[idx+1]])
				for line in image_lines_curve:
					cv2.line(plotted_curves, (line[0][1],line[0][0]), (line[1][1],line[1][0]), COLOR_CURVES, 4)
		result_frames_vectors.append(plotted_curves)


	#get real coordinates for goal path from goal state to inital state and also get rid of angles
	goal_path_animation = np.array([ coordinate_image(point[0:2]) for point in all_goal_points[::-1] ], np.int32)
	goal_path_lines = []
	#create lines which represent the goal path
	for idx in range(len(goal_path_animation)-1):
		goal_path_lines.append([goal_path_animation[idx], goal_path_animation[idx+1]])
	#Set lines per frame to display goal
	ratio_goal_per_frame = len(goal_path_lines) // MAX_GOAL_FRAMES
	ratio_goal_per_frame = ratio_goal_per_frame if ratio_goal_per_frame > LINES_PER_FRAME else LINES_PER_FRAME
	print(f'Lines of goal per frame will be: {ratio_goal_per_frame}')
	goal_lines_per_frame = divide_array(ratio_goal_per_frame, goal_path_lines)
	first_frame_goal = result_frames_vectors[-1].copy()
	for set_lines in tqdm(goal_lines_per_frame, desc = 'Animating goal path'):
		for line in set_lines:
			cv2.line(first_frame_goal, (line[0][1],line[0][0]), (line[1][1],line[1][0]), COLOR_GOAL, 5)
		result_frames_goal.append(first_frame_goal.copy())
	#final processing
	#downsizing total images to a dimension for encoding video
	resize_result_frames_vectors = resize_frames(result_frames_vectors, 'exploration space frames')
	resize_frames_goal = resize_frames(result_frames_goal, 'goal path frames')
	#add extra frames for the end to display more time the final result
	extra_frames = []
	for idx in range(50):
		extra_frames.append(resize_frames_goal[-1])
	result_frames_total = resize_result_frames_vectors + resize_frames_goal + extra_frames
	try:
		video = cv2.VideoWriter(
					'a_star_robot_part1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (RESIZE_WIDTH, RESIZE_HEIGHT))
		for frame in tqdm(result_frames_total, desc ="Creating video..."):
			video.write(frame)
		video.release()
	except Exception as err:
		print(err)
		print('Problem generation Video. Please check your dependencies and try again.')

