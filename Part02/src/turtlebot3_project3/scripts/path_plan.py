
#TODO: GAURAV AND NAGA
#* set any configurations and steps to deploy, test the script
#* use the txt file to add random linear and angular velocities and see its actually reading these velocities as expected
#* find the best delay time between each velocities sent
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
class GoalPath(Node):
    def __init__(self):
        super().__init__('goal_path_node')
        self.cmd_vel_a_star = self.create_publisher(Twist, '/cmd_vel', 10)

    def run_command_sequence(self):
        self.msg = """
        Running A* algorithm solution action command sequence
        ---------------------------
        """
        with open('command_set_solution.txt', 'r') as file:
            for line in file:
                #obtain the command and create twist message to turtlebot
                command = line.rstrip().split(',')
                velocities_msg = Twist()
                velocities_msg.linear.x = float(command[0])
                velocities_msg.angular.z = float(command[1])
                #publish the message
                print("Linear Velocity x",velocities_msg.linear.x) #m/s
                print("Steer Angle",velocities_msg.angular.z) #rad/s
                self.cmd_vel_a_star.publish(velocities_msg)
                #TODO: # Adjust the delay time as needed to cover the distance wanted according to simulation
                time.sleep(0.1)

#execute the script
def main(args=None):
    rclpy.init(args=args)
    node = GoalPath()
    node.run_command_sequence()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()