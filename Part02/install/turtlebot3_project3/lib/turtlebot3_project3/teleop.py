#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import os
import time

class GoalPath(Node):

	def __init__(self,commands):
		super().__init__('goal_path')

		self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 3)

	def run_command_sequence(self, vel_commands):
		self.msg = """
		Running A* algorithm solution action command sequence
		---------------------------
		"""
		for vel in vel_commands:
			velocities_msg = Twist()
			velocities_msg.linear.x = vel[0]
			velocities_msg.angular.z = vel[1]
			#publish the message
			print(f"Linear Velocity x {velocities_msg.linear.x} m/s, Steer angle {velocities_msg.angular.z} rad/s") #m/s
			self.cmd_vel_pub.publish(velocities_msg)
			#TODO: # Adjust the delay time as needed to cover the distance wanted according to simulation
			if vel[1] == 0.0:
				time.sleep(5.2)
			else:
				time.sleep(1)

def main(args=None):
	rclpy.init(args=args)
	# Get the current working directory
	current_directory = os.getcwd()
	additional_path = 'src/turtlebot3_project3/scripts/'
	full_path = os.path.join(current_directory, additional_path)
	print(full_path)
	file = open(f'{full_path}command_set_solution.txt', "r")
	data = file.readlines()
	file.close()
	vel_values = [vel.strip().replace('\n','') for vel in data]
	vel_commands = []
	for command in vel_values:
		lin,rot = command.split(',')
		vel_commands.append((float(lin), float(rot)))
	node = GoalPath(len(vel_commands))
	node.run_command_sequence(vel_commands)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()