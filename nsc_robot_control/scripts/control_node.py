#! /usr/bin/env python

import rospy
import numpy as np
from time import sleep
import sys
from itertools import product
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

# Real robot
REAL_ROBOT = False

class ControlNode:
    def __init__(self, state_dims=[3,3,3,3,3,3]):
        rospy.init_node('control_node', anonymous=False)
        #TODO
        # self. ... =
        self.actions = [straight, right, left]
        # self.states = [[goal_reached, mid_far_away, far_away]
        #                [straight_ahead, left_from_robot, right_from_robot]
        #                [obstacle_ahead], [obstacle_right], [obstacle_behind], [obstacle_left]

        self.action_space = self.create_actions(action_no=len(self.actions))
        self.state_space = self.create_state_space(state_dims=state_dims)
        self.data_path = '/home/nsc/repos/ros_my_packages/catkin_ws/src/nsc_robot/nsc_robot_control/Data' # TODO get it working on every PC
        self.Q_table_untrained_source = self.data_path + '/Qtable_untrained.csv'
        self.Q_table_trained_source = self.data_path + '/Qtable_trained.csv'
        self.Q_table = None
        self.rate = rospy.Rate(5)
        # self.Q_table = self.read_Q_table(Q_table_source+'/Qtable.csv')
        # print('Initial Q-table:')
        # print(Q_table[:5])
        self.lidar_distances_close = 1.0
        self.lidar_distances_middle = 1.5

        self.epsilon = 0.2
        self.episodes = 20
        self.training_iterations = 50
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.1  # Discount factor

        self.MIN_TIME_BETWEEN_ACTIONS = 0.1

        self.STATE_SPACE_IND_MIN = 0
        self.STATE_SPACE_IND_MAX = len(self.state_space)

        self.X_INIT = 0.0
        self.Y_INIT = 0.0
        self.THETA_INIT = 0.0
        self.randomized_start_pos=True
        self.x_odom_pos = 0.0
        self.y_odom_pos = 0.0
        self.x_goal_pos = 0.2
        self.y_goal_pos = 0.2
        self.lidar_distances = None
        self.lidar_angles = None
        self.MAX_LIDAR_DISTANCE = 5.0

        self.theta_goal_pos = 0.5
        self.yaw_odom_pos = None
        self.yaw_odom_pos_degree = None

        self.const_linear_speed_forward = 0.08
        self.const_angular_speed_forward = 0.0
        self.const_linear_speed_turn = 0.06
        self.const_angular_speed_turn = 0.4

        self.setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


    def run_control(self):
        Q_table = np.random.rand(726, 4)
        np.savetxt(self.Q_table_trained_source, Q_table, delimiter=' , ')

        rospy.Subscriber('/scan', LaserScan, self.get_lidar_readings_callback)
        rospy.Subscriber('/odom', Odometry, self.get_position_callback)
        sleep(1)
        (x, y, theta) = self.robot_set_pos(randomized=self.randomized_start_pos)
        self.Q_table = self.readQtable(self.Q_table_trained_source)
        #print("used self.Q_table", self.Q_table)

        while not rospy.is_shutdown():
            (state_ind, current_state) = self.get_state()
            goal_reached = self.check_goal_reached(current_state)
            crash = self.check_crash()
            if goal_reached or crash:
                print("goal_reached or crash", goal_reached, crash)
                rospy.signal_shutdown('End of testing!')
                return

            a_ind = self.getBestAction(self.Q_table, state_ind)
            self.robotDoAction(a_ind)
            print("current_state", current_state, "action_ind", a_ind)
            self.rate.sleep()

    def run_learning(self):

        rospy.Subscriber('/scan', LaserScan , self.get_lidar_readings_callback)
        rospy.Subscriber('/odom', Odometry, self.get_position_callback)
        sleep(1)
        # init Q_table
        self.Q_table = self.initialize_Q_table(len(self.state_space), len(self.actions))
        while not rospy.is_shutdown():
            for ep in range(self.episodes):

                (x, y, theta) = self.robot_set_pos(randomized=self.randomized_start_pos)
                start_of_loop_time = rospy.Time.now()  # init
                counter = 0
                # initial state
                (next_state_ind, next_state) = self.get_state()
                goal_reached = self.check_goal_reached(next_state)
                crash = self.check_crash()
                if goal_reached or crash:
                    print("goal_reached or crash", goal_reached, crash)

                for i in range(self.training_iterations):
                    end_of_loop_time = start_of_loop_time
                    start_of_loop_time = rospy.Time.now()
                    step_time = (start_of_loop_time - end_of_loop_time).to_sec()
                    print("step_time", step_time)

                    current_state = next_state
                    current_state_ind = next_state_ind


                    action = self.epsiloGreedyExploration(self.Q_table, current_state_ind, self.action_space, self.epsilon)
                    self.robotDoAction(action)
                    counter += 1
                    print(counter)

                    (next_state_ind, next_state) = self.get_state()
                    goal_reached = self.check_goal_reached(next_state)
                    crash = self.check_crash()

                    reward = self.getReward(next_state, action)
                    self.Q_table = self.updateQTable(self.Q_table, current_state_ind, next_state_ind,
                                                         action, reward, self.alpha, self.gamma)

                    if goal_reached:
                        print("goal_reached or crash", goal_reached)
                        # if goal reached go into next iteration
                        break
                    self.rate.sleep()
            self.save_Q_table(self.Q_table_trained_source, self.Q_table)
            rospy.signal_shutdown('End of learning!')

    def epsiloGreedyExploration(self, Q_table, state_ind, actions, epsilon):
        # Epsilog Greedy Exploration action chose
        if np.random.uniform() > epsilon and self.STATE_SPACE_IND_MIN <= state_ind <= self.STATE_SPACE_IND_MAX:

            action = self.getBestAction(Q_table, state_ind)
            print("take best action", action)
        else:

            action = self.getRandomAction(actions)
            print("take random action", action)

        return action

    def getReward(self, current_state, action):
            # distance (goal reached)
            if current_state[0] == 0:
                r_goal_distance = 5
            else:
                r_goal_distance = 0

            # heading (towards goal)
            if current_state[1] == 0:
                r_heading = 5
            else:
                r_heading = 0


            # obstacles too close
            if current_state[2] == 2 or current_state[3] == 2 or current_state[4] == 2 or current_state[5] == 2:
                r_close_obstacle = -2
            else:
                r_close_obstacle = 0

            # the more time needed, the more negative reward
            r_time_duration = -1

            # driving straight is rewarded
            """if action == 0:
                r_action = +0.2
            else:
                r_action = -0.1"""

            total_reward = r_time_duration  + r_goal_distance + r_close_obstacle + r_heading # + r_action
            print("total_reward", total_reward)
            return total_reward

    # Update Q-table values
    def updateQTable(self, Q_table, state_ind, next_state_ind, action, reward, alpha, gamma):
        if self.STATE_SPACE_IND_MIN <= state_ind <= self.STATE_SPACE_IND_MAX and \
                self.STATE_SPACE_IND_MIN <= next_state_ind <= self.STATE_SPACE_IND_MAX:
            Q_table[state_ind, action] = (1 - alpha) * Q_table[state_ind, action] + alpha * \
                                         (reward + gamma * max(Q_table[next_state_ind, :]) - Q_table[state_ind, action])
        else:
            print("state_ind", state_ind, "next_state_ind", next_state_ind, "are NOT ok")
        return Q_table

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
        #print("Q_table", Q_table)
        a_ind = np.argmax(Q_table[state_ind, :])
        return a_ind

    def robotDoAction(self, action):
        if action == 0:
            self.robotTurnLeft() #robotGoForward()
        elif action == 1:
            self.robotTurnRight()
        elif action == 2:
            self.robotGoBackward()
        elif action == 3:
            self.robotTurnLeft()
        elif action == 4:
            self.robotStop()
        else:
            print("Invalid action id")

    # Read Q table from path
    def readQtable(self, path):
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
        lidar_state = self.lidar_discretization(self.lidar_distances, self.lidar_angles)
        current_state = list(robot_pos_state) + list(lidar_state)

        ss = np.where(np.all(self.state_space == np.array([current_state]), axis=1))
        state_ind = int(ss[0])
        print("state_ind, current_state", state_ind, current_state)
        return (state_ind, current_state)

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

    def lidar_discretization(self, lidar_distances, lidar_angles):
        """
        TODO
        """
        number_scans = len(lidar_angles)
        #print("lidar_distances, lidar_angles", lidar_distances, lidar_angles)
        lidar_state_front_max = 0
        lidar_state_right_max = 0
        lidar_state_back_max = 0
        lidar_state_left_max = 0
        for i in range(number_scans):
            if 30 > lidar_angles[i] >= 0 or 360 > lidar_angles[i] > 330:
                if lidar_distances[i] < self.lidar_distances_close:
                    lidar_state_front = 2
                elif lidar_distances[i] < self.lidar_distances_middle:
                    lidar_state_front = 1
                elif lidar_distances[i] >= self.lidar_distances_middle:
                    lidar_state_front = 0
                lidar_state_front_max = max(lidar_state_front_max, lidar_state_front)
            if 120 >= lidar_angles[i] >= 30:
                if lidar_distances[i] < self.lidar_distances_close:
                    lidar_state_right = 2
                elif lidar_distances[i] < self.lidar_distances_middle:
                    lidar_state_right = 1
                elif lidar_distances[i] >= self.lidar_distances_middle:
                    lidar_state_right = 0
                lidar_state_right_max = max(lidar_state_right_max, lidar_state_right)
            if 240 >= lidar_angles[i] > 120:
                if lidar_distances[i] < self.lidar_distances_close:
                    lidar_state_back = 2
                elif lidar_distances[i] < self.lidar_distances_middle:
                    lidar_state_back = 1
                elif lidar_distances[i] >= self.lidar_distances_middle:
                    lidar_state_back = 0
                lidar_state_back_max = max(lidar_state_back_max, lidar_state_back)
            if 330 >= lidar_angles[i] > 240:
                if lidar_distances[i] < self.lidar_distances_close:
                    lidar_state_left = 2
                elif lidar_distances[i] < self.lidar_distances_middle:
                    lidar_state_left = 1
                elif lidar_distances[i] >= self.lidar_distances_middle:
                    lidar_state_left = 0
                lidar_state_left_max = max(lidar_state_left_max, lidar_state_left)
        lidar_state = (lidar_state_front_max, lidar_state_right_max, lidar_state_back_max, lidar_state_left_max)
        return lidar_state

    def robot_pos_discretization(self):
        print("self.x_odom_pos", self.x_odom_pos, "self.y_odom_pos", self.y_odom_pos)
        self.heading_angle = math.degrees(math.atan2(self.y_goal_pos-self.y_odom_pos, self.x_goal_pos - self.x_odom_pos))
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
        return (self.distance_to_goal_state, self.heading_angle_state)

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

        checkpoint.model_name = 'lio_robot'

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
        print("set position to x , y , theta", x, y, theta)

        self.setPosPub.publish(checkpoint)

        return (x, y, theta)

    def get_lidar_readings_callback(self, data):
        print("got lidar data")
        self.lidar_distances = np.array([])
        self.lidar_angles = np.array([])

        for i in range(len(data.ranges)):
            angle = data.angle_min + math.degrees(i * data.angle_increment) #angle_min=0?
            if (data.ranges[i] > self.MAX_LIDAR_DISTANCE ):
                distance = self.MAX_LIDAR_DISTANCE
            elif (data.ranges[i] < data.range_min ):
                distance = data.range_min

            else:
                distance = data.ranges[i]

            self.lidar_distances = np.append(self.lidar_distances, distance)
            self.lidar_angles = np.append(self.lidar_angles, angle)

        # distances in [m], angles in [degrees]
        #print("(self.lidar_distances, self.lidar_angles)", self.lidar_distances, self.lidar_angles)
        return (self.lidar_distances, self.lidar_angles)

    def get_position_callback(self, data):
        self.x_odom_pos = data.pose.pose.position.x
        self.y_odom_pos = data.pose.pose.position.y

        orientation_q = data.pose.pose.orientation
        orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.roll_odom_pos, self.pitch_odom_pos, self.yaw_odom_pos) = euler_from_quaternion(orientation_list)
        self.yaw_odom_pos_degree = math.degrees(self.yaw_odom_pos)
        #print("get_position_callback x, y, yaw:", self.x_odom_pos, self.y_odom_pos, self.yaw_odom_pos)

        return (self.x_odom_pos, self.y_odom_pos, self.yaw_odom_pos)

    def getRandomAction(self, actions):
        n_actions = len(actions)
        a_ind = np.random.randint(n_actions)
        return actions[a_ind]

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
        print("Q_table.size()", Q_table.size)
        return Q_table


if __name__ == '__main__':
    learning = True
    try:
        CN = ControlNode(state_dims=[3, 3, 3, 3, 3, 3])

        if learning is True:
            CN.run_learning()
        elif learning is False:
            CN.run_control()


    except rospy.ROSInterruptException:
        print('Simulation terminated!')
        pass
