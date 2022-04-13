#! /usr/bin/env python

import rospy
import numpy as np
from time import sleep
import sys
from itertools import product
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

# Real robot
REAL_ROBOT = False

class ControlNode:
    def __init__(self, action_no=4, state_dims=[3,3,3,3,3,3]):
        # self. ... =
        self.actions = self.create_actions(action_no=action_no)
        self.state_space = self.create_state_space(state_dims=state_dims)
        self.data_path = '/home/User/repos/catkin_ws/src/nsc_robot_control/Data' # TODO get it working on every PC
        # self.Q_table_source = self.data_path + '/Log_learning_FINAL'
        # self.Q_table = self.read_Q_table(Q_table_source+'/Qtable.csv')
        # print('Initial Q-table:')
        # print(Q_table[:5])
        self.X_INIT = 0.0
        self.Y_INIT = 0.0
        self.THETA_INIT = 0.0
        self.randomized_start_pos=False
        self.x_odom_pos = 0.0
        self.y_odom_pos = 0.0
        self.x_goal_pos = 0.5
        self.y_goal_pos = 0.5
        self.lidar_distances = None
        self.lidar_angles = None
        self.theta_goal_pos = 0.5
        self.yaw_odom_pos = None
        self.yaw_odom_pos_degree = None

        self.const_linear_speed_forward = 0.08
        self.const_angular_speed_forward = 0.0
        self.const_linear_speed_turn = 0.06
        self.const_angular_speed_turn = 0.4

        rospy.init_node('control_node', anonymous=False)

        self.setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def run_control(self):

        #rospy.Subscriber('/scan', LaserScan , self.get_lidar_readings_callback)  # TODO get lidar working
        rospy.Subscriber('/odom', Odometry, self.get_position_callback)

        rate = rospy.Rate(10)
        (x, y, theta) = self.robot_set_pos(randomized=self.randomized_start_pos)
        Q_table = self.initialize_Q_table(len(self.state_space), len(self.actions))  # TODO self.readQTable(Q_TABLE_SOURCE+'/Qtable.csv')

        while not rospy.is_shutdown():
            (state_ind, current_state) = self.get_state()
            goal_reached = self.check_goal_reached(current_state)
            crash = self.check_crash()
            if goal_reached or crash:
                print("goal_reached or crash", goal_reached, crash)
                rospy.signal_shutdown('End of testing!')
                return

            a_ind = self.getBestAction(Q_table, state_ind)
            self.robotDoAction(a_ind)
            print("current_state", current_state, "action_ind", a_ind)
            rate.sleep()

    def robotGoForward(self):
        velMsg = self.createVelMsg(+self.const_linear_speed_forward, self.const_angular_speed_forward)
        self.velPub.publish(velMsg)

    def robotGoBackward(self):
        velMsg = self.createVelMsg(-self.const_linear_speed_forward, self.const_angular_speed_forward)
        self.velPub.publish(velMsg)

    def robotTurnLeft(self):
        velMsg = self.createVelMsg(self.const_linear_speed_turn, +self.const_linear_speed_turn)
        self.velPub.publish(velMsg)

    def robotTurnRight(self):
        velMsg = self.createVelMsg(self.const_angular_speed_turn, -self.const_angular_speed_turn)
        self.velPub.publish(velMsg)

    def robotStop(self):
        velMsg = self.createVelMsg(0.0, 0.0)
        self.velPub.publish(velMsg)

    # Create rosmsg Twist()
    def createVelMsg(self, v, w):
        velMsg = Twist()
        velMsg.linear.x = v
        velMsg.linear.y = 0
        velMsg.linear.z = 0
        velMsg.angular.x = 0
        velMsg.angular.y = 0
        velMsg.angular.z = w
        return velMsg

    def getBestAction(self, Q_table, state_ind):
        a_ind = np.argmax(Q_table[state_ind, :])
        return a_ind

    def robotDoAction(self, action):
        if action == 0:
            self.robotGoForward()
        elif action == 1:
            self.robotTurnRight()
        elif action == 2:
            self.robotTurnBackward()
        elif action == 3:
            self.robotTurnLeft()
        else:
            print("Invalid action id")

    # Read Q table from path
    def readQTable(self, path):
        Q_table = np.genfromtxt(path, delimiter=' , ')
        return Q_table

    def get_state(self):
        """
        states:
        (3,3,3,3,3,3)

        goal_distance: L2 norm in 3 sectors
        (0-0.1; 0.1-1.0; 0.5-10
        goal_heading: theta_goal in 3 sectors
        (-5-5; 6-180, 181-355)
        lidar sensor: 4 angle sectors with 2 distance sectors -> 8; e.g.
        x0,x1,x2,x3 = (0,1,1,2) ->
        x0 front: nothing, x1 right: distance obstacle
        x2 back: distance obstacle; x3 back: close obstacle


        for non discrtized states (future)
        # robot pos: x,y,theta robot; e.g (0.1, 0.3, 0.2)
        # goal pos: x,y, theta; e.g (0.6, 0.7, 0.9)

        """
        robot_pos_state = self.robot_pos_discretization()
        lidar_state = self.lidar_discretization(self.state_space, self.lidar_distances, self.lidar_angles)
        current_state = list(robot_pos_state) + list(lidar_state)

        ss = np.where(np.all(self.state_space == np.array([current_state]), axis=1))
        state_ind = int(ss[0])
        return state_ind, current_state

    def check_goal_reached(self, current_state):
        if current_state[0] == 0:
            goal_reached = True
        else:
            goal_reached = False
        return goal_reached

    def check_crash(self):

        lidar = 1.0  #TODO get lidar signal

        if lidar < 0.01:
            crash = True
        else:
            crash = False
        return crash

    def lidar_discretization(self, state_space, lidar_distances, lidar_angles):
        """
        TODO
        """
        lidar_state=(0, 0, 0, 0)
        return lidar_state

    def robot_pos_discretization(self):
        print("self.x_odom_pos", self.x_odom_pos, "self.y_odom_pos", self.y_odom_pos)
        self.heading_angle = math.degrees(math.atan2(self.y_goal_pos-self.y_odom_pos, self.x_goal_pos-self.x_odom_pos))
        self.distance_to_goal = ((self.x_odom_pos - self.x_goal_pos)**2 + (self.y_odom_pos - self.y_goal_pos)**2)**0.5
        print("self.heading_angle", self.heading_angle, "self.distance_to_goal", self.distance_to_goal)
        #self.theta_goal_pos
        #self.yaw_odom_pos
        # 0 starts at 90 looking at a compass (see atan2)
        if 95 >= self.heading_angle >= 85:
            self.heading_angle_state = 0
        elif 85 >= self.heading_angle >= -90:
            self.heading_angle_state = 1
        else:
            self.heading_angle_state = 2

        if 0.1 <= self.distance_to_goal <= 0.0:
            self.distance_to_goal_state = 0
        elif 0.1 < self.distance_to_goal <= 1.0:
            self.distance_to_goal_state = 1
        else:
            self.distance_to_goal_state = 2
        return (self.heading_angle_state, self.distance_to_goal_state)

    def robot_set_pos(self, randomized=False):

        if randomized:
            # Set random initial robot position and orientation

            x_range = np.array([-0.4, 0.6, 0.6, -1.4, -1.4, 2.0, 2.0, -2.5, 1.0, -1.0])
            y_range = np.array([-0.4, 0.6, -1.4, 0.6, -1.4, 1.0, -1.0, 0.0, 2.0, 2.0])
            theta_range = np.arange(0, 360, 15)
            # theta_range = np.array([0, 30, 45, 60, 75, 90])

            ind = np.random.randint(0, len(x_range))
            ind_theta = np.random.randint(0, len(theta_range))

            x = x_range[ind]
            y = y_range[ind]
            theta = theta_range[ind_theta]

        else:
            x = self.X_INIT
            y = self.Y_INIT
            theta = self.THETA_INIT

        checkpoint = ModelState()

        checkpoint.model_name = 'nsc_robot'

        checkpoint.pose.position.x = x
        checkpoint.pose.position.y = y
        checkpoint.pose.position.z = 0.0

        [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, math.radians(theta))

        checkpoint.pose.orientation.x = x_q
        checkpoint.pose.orientation.y = y_q
        checkpoint.pose.orientation.z = z_q
        checkpoint.pose.orientation.w = w_q

        checkpoint.twist.linear.x = 0.0
        checkpoint.twist.linear.y = 0.0
        checkpoint.twist.linear.z = 0.0

        checkpoint.twist.angular.x = 0.0
        checkpoint.twist.angular.y = 0.0
        checkpoint.twist.angular.z = 0.0
        print("x , y , theta", x, y, theta)

        self.setPosPub.publish(checkpoint)

        return (x, y, theta)

    def get_lidar_readings_callback(self, data):
        distances = np.array([])
        angles = np.array([])

        for i in range(len(data.ranges)):
            angle = degrees(i * data.angle_increment)
            if (data.ranges[i] > MAX_LIDAR_DISTANCE ):
                distance = MAX_LIDAR_DISTANCE
            elif (data.ranges[i] < data.range_min ):
                distance = data.range_min
                # For real robot - protection
                if data.ranges[i] < 0.01:
                    distance = MAX_LIDAR_DISTANCE
            else:
                distance = data.ranges[i]

            self.lidar_distances = np.append(distances, distance)
            self.lidar_angles = np.append(angles, angle)

        # distances in [m], angles in [degrees]
        return (self.lidar_distances, self.lidar_angles)

    def get_position_callback(self, data):
        self.x_odom_pos = data.pose.pose.position.x
        self.y_odom_pos = data.pose.pose.position.y

        orientation_q = data.pose.pose.orientation
        orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.roll_odom_pos, self.pitch_odom_pos, self.yaw_odom_pos) = euler_from_quaternion(orientation_list)
        self.yaw_odom_pos_degree = math.degrees(self.yaw_odom_pos)
        print("get_position_callback x, y, yaw:", self.x_odom_pos, self.y_odom_pos, self.yaw_odom_pos)

        return (self.x_odom_pos, self.y_odom_pos, self.yaw_odom_pos)

    # Create actions
    def create_actions(self, action_no):
        return np.array(range(action_no))

    # Create state space for Q table
    def create_state_space(self, state_dims=[3, 3, 3, 3, 3, 3]):
        x = []
        for i in range(len(state_dims)):
            x.append(range(state_dims[i]))

        state_space = product(*x)
        return np.array(list(state_space))

    # Read Q table from path
    def read_Q_table(self, path):
        Q_table = np.genfromtxt(path, delimiter=' , ')
        return Q_table

    # Write Q table to path
    def save_Q_table(self, path, Q_table):
        np.savetxt(path, Q_table, delimiter=' , ')

    def initialize_Q_table(self, n_states, n_actions):
        #Q_table = np.random.uniform(low = -0.05, high = 0, size = (n_states,n_actions) )
        Q_table = np.zeros((n_states, n_actions))
        return Q_table


if __name__ == '__main__':
    try:
        CN = ControlNode(action_no=4, state_dims=[3, 3, 3, 3, 3, 3])
        CN.run_control()

    except rospy.ROSInterruptException:
        print('Simulation terminated!')
        pass
