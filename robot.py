import socket
import select
import struct
import time
import os
import numpy as np
import utils
import math
from functools import partial, reduce
from scipy.spatial.transform import Rotation as R
from simulation import vrep

# from GPD pkg
import rospy
from gpd.msg import GraspConfigList
#
# global variable to store grasps
grasps = []
#
# Callback function to receive grasps.
def callback(msg):
    global grasps
    grasps = msg.grasps

# ==================== MAIN ====================
# Create a ROS node.
rospy.init_node('get_grasps')

# # Subscribe to the ROS topic that contains the grasps.
sub = rospy.Subscriber('/detect_grasps/clustered_grasps', GraspConfigList, callback)

# Wait for grasps to arrive.
rate = rospy.Rate(1)

def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi


class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):

        self.is_sim = is_sim
        self.workspace_limits = workspace_limits

        # If in simulation...
        if self.is_sim:

            # Define colors for object meshes (Tableau palette)
            self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                                           [89.0, 161.0, 79.0],  # green
                                           [156, 117, 95],  # brown
                                           [242, 142, 43],  # orange
                                           [237.0, 201.0, 72.0],  # yellow
                                           [186, 176, 172],  # gray
                                           [255.0, 87.0, 89.0],  # red
                                           [176, 122, 161],  # purple
                                           [118, 183, 178],  # cyan
                                           [255, 157, 167]]) / 255.0  # pink

            # Read files in object mesh directory
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

            # Make sure to have the server side running in V-REP:
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            #
            # simExtRemoteApiStart(19999)
            #
            # then start simulation, and run this program.
            #
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!

            # MODIFY remoteApiConnections.txt

            # Connect to simulator
            vrep.simxFinish(-1)  # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            sim_ret, self.UR5_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5', vrep.simx_opmode_blocking)
            # sim_ret, self.RF_handle = vrep.simxGetObjectHandle(self.sim_client, 'ReferenceFrame', vrep.simx_opmode_blocking)

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines()
                self.test_obj_mesh_files = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                for object_idx in range(self.num_obj):
                    file_content_curr_object = file_content[object_idx].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir, file_content_curr_object[0]))
                    self.test_obj_mesh_colors.append(
                        [float(file_content_curr_object[1]), float(file_content_curr_object[2]),
                         float(file_content_curr_object[3])])
                    self.test_obj_positions.append(
                        [float(file_content_curr_object[4]), float(file_content_curr_object[5]),
                         float(file_content_curr_object[6])])
                    self.test_obj_orientations.append(
                        [float(file_content_curr_object[7]), float(file_content_curr_object[8]),
                         float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

            # Add objects to simulation environment
            self.add_objects()
            # vrep.simxPauseSimulation(self.sim_client, vrep.simx_opmode_blocking)

        # If in real-settings...
        else:

            # Connect to robot client
            self.tcp_host_ip = tcp_host_ip
            self.tcp_port = tcp_port
            # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Connect as real-time client to parse state data
            self.rtc_host_ip = rtc_host_ip
            self.rtc_port = rtc_port

            # Default home joint configuration
            # self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
            self.home_joint_config = [-(180.0 / 360.0) * 2 * np.pi, -(84.2 / 360.0) * 2 * np.pi,
                                      (112.8 / 360.0) * 2 * np.pi, -(119.7 / 360.0) * 2 * np.pi,
                                      -(90.0 / 360.0) * 2 * np.pi, 0.0]

            # Default joint speed configuration
            self.joint_acc = 8  # Safe: 1.4
            self.joint_vel = 3  # Safe: 1.05

            # Joint tolerance for blocking calls
            self.joint_tolerance = 0.01

            # Default tool speed configuration
            self.tool_acc = 1.2  # Safe: 0.5
            self.tool_vel = 0.25  # Safe: 0.2

            # Tool pose tolerance for blocking calls
            self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

            # Move robot to home pose
            self.close_gripper()
            self.go_home()

            # Fetch RGB-D data from RealSense camera
            from real.camera import Camera
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics

            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp',
                                                            vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1,
                                                           vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1,
                                                                 vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        # print("\n robot.py add_objects() before adding objects")
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + \
                     self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + \
                     self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1],
                                   self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0],
                                      self.test_obj_orientations[object_idx][1],
                                      self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1],
                            self.obj_mesh_color[object_idx][2]]
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                                  'remoteApiCommandServer',
                                                                                                  vrep.sim_scripttype_childscript,
                                                                                                  'importShape',
                                                                                                  [0, 0, 255, 0],
                                                                                                  object_position + object_orientation + object_color,
                                                                                                  [curr_mesh_file,
                                                                                                   curr_shape_name],
                                                                                                  bytearray(),
                                                                                                  vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(2)
        self.prev_obj_positions = []
        self.obj_positions = []
        # print("\n robot.py add_objects() added objects")
        while not rospy.is_shutdown():
            print("\nWaiting for initial grasp samples to arrive...")
            if len(grasps) > 0:
                rospy.loginfo('Received %d grasps.', len(grasps))
                break
            rate.sleep()

    def get_obj_positions(self):
        obj_positions = []
        for handle in self.object_handles:
            obj_positions.append(vrep.simxGetObjectPosition(self.sim_client, handle, -1, vrep.simx_opmode_blocking))
        return np.array(obj_positions)

    def get_obj_orientations(self):
        obj_orientations = []
        for handle in self.object_handles:
            sim_ret, orientation = vrep.simxGetObjectOrientation(self.sim_client, handle, -1, vrep.simx_opmode_blocking)
            obj_orientations.append(orientation)
        # print(obj_positions)
        return np.array(obj_orientations)

    def obj_in_workspace(self, position):
        if position[0] < self.workspace_limits[0][0] or position[0] > self.workspace_limits[0][1]:
            return False
        if position[1] < self.workspace_limits[1][0] or position[1] > self.workspace_limits[1][1]:
            return False
        if position[2] < self.workspace_limits[2][0] or position[0] > self.workspace_limits[2][1]:
            return False
        return True

    # def get_grasp_sample(self):
    #     grasp_samples = []
    #     obj_positions = self.get_obj_positions()
    #     obj_orientations = self.get_obj_orientations()
    #     num_rotations = 4
    #     rotation_inc = 180/num_rotations
    #     # print ("obj_orientations:\n", obj_orientations)
    #     # exit(-1)
    #     for i in range(len(self.object_handles)):
    #         if self.obj_in_workspace(obj_positions[i]):
    #             for j in range(num_rotations):
    #                 orientation = obj_orientations[i]
    #                 orientation[2] = obj_orientations[i][2] + (j*rotation_inc*np.pi/180)
    #                 if orientation[2] > np.pi:
    #                     orientation[2] = orientation[2] - np.pi
    #                 # print (orientation[2])
    #                 obj_pose = np.append(obj_positions[i], orientation)
    #                 grasp_samples.append(obj_pose)
    #     # print ("grasp_samples:\n", grasp_samples)
    #     # exit(-1)
    #     return np.array(grasp_samples)

    def get_grasp_sample(self):
        camera_offset = [0.0, 0.0, 0.5]
        pos_r = R.from_quat([0.65328139066696, 0.65328139066696, -0.27059826254845, -0.27059826254845])
        grasp_samples = []
        # print (len(grasps))
        # print("-------------------grasps detected--------------------\n")
        if len(grasps) != 0:
            for i in range(len(grasps)):
                r = np.array([[-grasps[i].approach.x, -grasps[i].axis.x, grasps[i].binormal.x],
                              [-grasps[i].approach.y, -grasps[i].axis.y, grasps[i].binormal.y],
                              [-grasps[i].approach.z, -grasps[i].axis.z, grasps[i].binormal.z]])
                psi, theta, phi = euler_angles_from_rotation_matrix(r)

                # pos_top = grasps[i].surface
                pos_top = grasps[i].top
                pos_bottom = grasps[i].bottom
                trans_pos_top = pos_r.apply([pos_top.x, pos_top.y, pos_top.z]) + camera_offset
                trans_pos_bottom = pos_r.apply([pos_bottom.x, pos_bottom.y, pos_bottom.z]) + camera_offset

                if self.obj_in_workspace(trans_pos_top):
                    grasp = [trans_pos_top[0], trans_pos_top[1], trans_pos_top[2],
                             psi, theta, phi]
                    grasp_samples.append(grasp)

        if len(grasps) == 0 or len(grasp_samples) == 0:
            grasp_samples.append([-0.5, 0.0, 0.1, np.pi/2, 0.0, np.pi/2])
            # self.get_obj_positions()
        # print("------------------------------------------------------\n")

        return np.reshape(np.asarray(grasp_samples), (-1, 6))

    def get_push_sample(self):
        push_samples = []
        max_x, max_y = 0.0, 0.0
        min_x, min_y = 0.0, 0.0
        obj_positions = self.get_obj_positions()
        # obj_orientations = self.get_obj_orientations()
        obj_on_table_pos = []
        for i in range(len(self.object_handles)):
            if self.obj_in_workspace(obj_positions[i]) == True:
                obj_on_table_pos.append(obj_positions[i])
        # print ("obj_on_table_pos:\n", obj_on_table_pos)
        obj_on_table_pos = np.array(obj_on_table_pos)
        if len(obj_on_table_pos) > 1:
            max_pos = np.amax(np.array(obj_on_table_pos), axis=0)
            min_pos = np.amin(np.array(obj_on_table_pos), axis=0)
            max_x, max_y = max_pos[0] + 0.1, max_pos[1] + 0.1
            min_x, min_y = min_pos[0] - 0.1, min_pos[1] - 0.1
            # max_x, min_x = max(obj_on_table_pos[:][0])+0.1, min(obj_on_table_pos[:][0])-0.1
            # max_y, min_y = max(obj_on_table_pos[:][1])+0.1, min(obj_on_table_pos[:][1])-0.1
        elif len(obj_on_table_pos) == 1:
            obj_on_table_pos = obj_on_table_pos.reshape((3))
            max_x, max_y = obj_on_table_pos[0] + 0.1, obj_on_table_pos[1] + 0.1
            min_x, min_y = obj_on_table_pos[0] - 0.1, obj_on_table_pos[1] - 0.1

        avg_x, avg_y = (max_x + min_x) / 2, (max_y + min_y) / 2

        push_samples.append([min_x, min_y, 0.01, avg_x, avg_y, 0.01])
        push_samples.append([min_x, avg_y, 0.01, avg_x, avg_y, 0.01])
        push_samples.append([min_x, max_y, 0.01, avg_x, avg_y, 0.01])
        push_samples.append([avg_x, min_y, 0.01, avg_x, avg_y, 0.01])
        push_samples.append([avg_x, max_y, 0.01, avg_x, avg_y, 0.01])
        push_samples.append([max_x, min_y, 0.01, avg_x, avg_y, 0.01])
        push_samples.append([max_x, avg_y, 0.01, avg_x, avg_y, 0.01])
        push_samples.append([max_x, max_y, 0.01, avg_x, avg_y, 0.01])

        return np.array(push_samples)

    def check_table_empty(self):
        object_positions = self.get_obj_positions()
        num_object = 0
        for idx in range(len(object_positions)):
            if self.obj_in_workspace(object_positions[idx]):
                num_object = num_object+1
        if num_object == 0:
            return True
        else:
            return False

    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5, 0, 0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                                   vrep.simx_opmode_blocking)

    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < \
                 self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and \
                 gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > \
                 self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()

    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0],  # red
                                    [-0.625, -0.125, 0.0],  # blue
                                    [-0.375, 0.125, 0.0],  # green
                                    [-0.375, -0.125, 0.0]])  # yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1, obj_positions.shape[1], 1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)

    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached

    # def stop_sim(self):
    #     if self.is_sim:
    #         # Now send some data to V-REP in a non-blocking fashion:
    #         # vrep.simxAddStatusbarMessage(sim_client,'Hello V-REP!',vrep.simx_opmode_oneshot)

    #         # # Start the simulation
    #         # vrep.simxStartSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # # Stop simulation:
    #         # vrep.simxStopSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    #         vrep.simxGetPingTime(self.sim_client)

    #         # Now close the connection to V-REP:
    #         vrep.simxFinish(self.sim_client)

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1,
                                                                  vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1,
                                                                  vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1,
                                                                        vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:
            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + \
                     workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + \
                     workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation,
                                          vrep.simx_opmode_blocking)
            time.sleep(2)

    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0,
                                                                           vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float) / 255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle,
                                                                                    vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        else:
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()
            # color_img = self.camera.color_data.copy()
            # depth_img = self.camera.depth_data.copy()

        return color_img, depth_img

    def parse_tcp_state_data(self, state_data, subpackage):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        robot_message_type = data_bytes[4]
        assert (robot_message_type == 16)
        byte_idx = 5

        # Parse sub-packages
        subpackage_types = {'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx + 4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0, 0, 0, 0, 0, 0]
            target_joint_positions = [0, 0, 0, 0, 0, 0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 8):(byte_idx + 16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            return tool_analog_input2

        parse_functions = {'joint_data': parse_joint_data, 'cartesian_info': parse_cartesian_info,
                           'tool_data': parse_tool_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        assert (data_length == 812)
        byte_idx = 4 + 8 + 8 * 48 + 24 + 120
        TCP_forces = [0, 0, 0, 0, 0, 0]
        for joint_idx in range(6):
            TCP_forces[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            byte_idx += 8

        return TCP_forces

    def ompl_move_to(self, position, orientation):

        emptyBuff = bytearray()
        res, target1 = vrep.simxGetObjectHandle(self.sim_client, 'ReferenceFrame', vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectPosition(self.sim_client, target1, -1, position, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, target1, -1, orientation, vrep.simx_opmode_blocking)
        # Retrieve the poses (i.e. transformation matrices, 12 values, last row is implicit) of some dummies in the scene
        res, retInts, target1Pose, retStrings, retBuffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                       'remoteApiCommandServer',
                                                                                       vrep.sim_scripttype_childscript,
                                                                                       'getObjectPose', [target1], [],
                                                                                       [], emptyBuff,
                                                                                       vrep.simx_opmode_oneshot_wait)
        res, retInts, robotInitialState, retStrings, retBuffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                             'remoteApiCommandServer',
                                                                                             vrep.sim_scripttype_childscript,
                                                                                             'getRobotState',
                                                                                             [self.UR5_handle], [], [],
                                                                                             emptyBuff,
                                                                                             vrep.simx_opmode_oneshot_wait)
        # Some parameters:
        approachVector = [0, 0, 1]  # often a linear approach is required. This should also be part of the calculations when selecting an appropriate state for a given pose
        maxConfigsForDesiredPose = 10  # we will try to find 10 different states corresponding to the goal pose and order them according to distance from initial state
        maxTrialsForConfigSearch = 300  # a parameter needed for finding appropriate goal states
        searchCount = 2  # how many times OMPL will run for a given task
        minConfigsForPathPlanningPath = 400  # interpolation states for the OMPL path
        minConfigsForIkPath = 100  # interpolation states for the linear approach path
        collisionChecking = 0  # whether collision checking is on or off

        # Do the path planning here (between a start state and a goal pose, including a linear approach phase):
        inInts = [self.UR5_handle, collisionChecking, minConfigsForIkPath, minConfigsForPathPlanningPath,
                  maxConfigsForDesiredPose, maxTrialsForConfigSearch, searchCount]
        inFloats = robotInitialState + target1Pose + approachVector
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                'remoteApiCommandServer',
                                                                                vrep.sim_scripttype_childscript,
                                                                                'findPath_goalIsPose', inInts, inFloats,
                                                                                [], emptyBuff,
                                                                                vrep.simx_opmode_oneshot_wait)

        if (res == 0) and len(path) > 0:
            # The path could be in 2 parts: a path planning path, and a linear approach path:
            part1StateCnt = retInts[0]
            part2StateCnt = retInts[1]
            path1 = path[:part1StateCnt * 6]

            # Visualize the first path:
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                         'remoteApiCommandServer',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'visualizePath',
                                                                                         [self.UR5_handle, 255, 0, 255],
                                                                                         path1, [], emptyBuff,
                                                                                         vrep.simx_opmode_oneshot_wait)
            line1Handle = retInts[0]

            # Make the robot follow the path:
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                         'remoteApiCommandServer',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'runThroughPath',
                                                                                         [self.UR5_handle], path1, [],
                                                                                         emptyBuff,
                                                                                         vrep.simx_opmode_oneshot_wait)

            # Wait until the end of the movement:
            runningPath = True
            while runningPath:
                res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                             'remoteApiCommandServer',
                                                                                             vrep.sim_scripttype_childscript,
                                                                                             'isRunningThroughPath',
                                                                                             [self.UR5_handle], [], [],
                                                                                             emptyBuff,
                                                                                             vrep.simx_opmode_oneshot_wait)
                runningPath = retInts[0] == 1

    def close_gripper(self, async_=False):

        if self.is_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                                   vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                        vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                            vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.06:  # Block until gripper is fully closed
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                                vrep.simx_opmode_blocking)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True

        else:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_digital_out(8,True)\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            if async:
                gripper_fully_closed = True
            else:
                time.sleep(1.5)
                gripper_fully_closed = self.check_grasp()

        return gripper_fully_closed

    def open_gripper(self, async_=False):

        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                                   vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                        vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                            vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.0336:  # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                            vrep.simx_opmode_blocking)

        else:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_digital_out(8,False)\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            if not async:
                time.sleep(1.5)

    def get_state(self):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        self.tcp_socket.close()
        return state_data

    def move_to(self, tool_position, tool_orientation):

        if self.is_sim:

            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)

            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)

        else:

            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
            tool_position[0], tool_position[1], tool_position[2], tool_orientation[0], tool_orientation[1],
            tool_orientation[2], self.tool_acc, self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            # Block until robot reaches target tool position
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all(
                    [np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
                # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)] + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) for j in range(3,6)])
                tcp_state_data = self.tcp_socket.recv(2048)
                prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                time.sleep(0.01)
            self.tcp_socket.close()

    def guarded_move_to(self, tool_position, tool_orientation):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

        # Read actual tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1  # 1.2 # 0.5

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray([(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < 0.01:
                increment_position = tool_position
            else:
                increment = 0.01 * increment / np.linalg.norm(increment)
                increment_position = np.asarray(actual_tool_pose[0:3]) + increment

            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
            increment_position[0], increment_position[1], increment_position[2], tool_orientation[0],
            tool_orientation[1], tool_orientation[2], self.tool_acc, self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j]) < self.tool_pose_tolerance[j] for j in
                           range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                tcp_state_data = self.tcp_socket.recv(2048)
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                time_snapshot = time.time()
                if time_snapshot - time_start > 1:
                    break
                time.sleep(0.01)

            # Reading TCP forces from real-time client connection
            rtc_state_data = self.rtc_socket.recv(6496)
            TCP_forces = self.parse_rtc_state_data(rtc_state_data)

            # If TCP forces in x/y exceed 20 Newtons, stop moving
            # print(TCP_forces[0:3])
            if np.linalg.norm(np.asarray(TCP_forces[0:2])) > 20 or (time_snapshot - time_start) > 1:
                print('Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' % (
                TCP_forces[0], TCP_forces[1], TCP_forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2  # 1.2 # 0.5

        self.tcp_socket.close()
        self.rtc_socket.close()

        return execute_success

    def move_joints(self, joint_configuration):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(2048)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all(
                [np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(2048)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)

        self.tcp_socket.close()

    def go_home(self):

        self.move_joints(self.home_joint_config)

    # Note: must be preceded by close_gripper()
    def check_grasp(self):

        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        return tool_analog_input2 > 0.26

    def rotate_gripper(self, orientation):
        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                     vrep.simx_opmode_blocking)
        rotation_step_x = 0.3 if (orientation[0] - gripper_orientation[0] > 0) else -0.3
        num_rotation_steps_x = int(np.floor((orientation[0] - gripper_orientation[0]) / rotation_step_x))
        rotation_step_y = 0.3 if (orientation[1] - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps_y = int(np.floor((orientation[1] - gripper_orientation[1]) / rotation_step_y))
        rotation_step_z = 0.3 if (orientation[2] - gripper_orientation[2] > 0) else -0.3
        num_rotation_steps_z = int(np.floor((orientation[2] - gripper_orientation[2]) / rotation_step_z))

        for step_iter in range(max(num_rotation_steps_x, num_rotation_steps_y, num_rotation_steps_z)):
            vrep.simxSetObjectOrientation(self.sim_client,
                                          self.UR5_target_handle, -1,
                                          (gripper_orientation[0] + rotation_step_x * min(step_iter, num_rotation_steps_x),
                                           gripper_orientation[1] + rotation_step_y * min(step_iter, num_rotation_steps_y),
                                           gripper_orientation[2] + rotation_step_z * min(step_iter, num_rotation_steps_z)),
                                          vrep.simx_opmode_blocking)

        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, orientation, vrep.simx_opmode_blocking)

    # Primitives ----------------------------------------------------------

    def grasp(self, position, orientation, workspace_limits):
        # def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
        grasp_success = False

        if self.is_sim:
            # vrep.simxSetObjectPosition(self.sim_client, self.RF_handle, -1, position[0:3], vrep.simx_opmode_blocking)
            # human_label = input()
            # if int(human_label) == 1:
            #     grasp_success = True
            # else:
            #     grasp_success = False

            # Compute tool orientation from heightmap rotation angle
            # tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # position_tip = position[0:3]
            # position_bottom = position[3:6]
            # location_before_grasp = [position[0], position[1], position[2]+0.15]
            # before_grasp_margin = 0.05
            # if position[0]-position[3]>0:
            #     location_before_grasp[0] = location_before_grasp[0] - before_grasp_margin
            # else:
            #     location_before_grasp[0] = location_before_grasp[0] + before_grasp_margin
            #
            # if position[1]-position[4]>0:
            #     location_before_grasp[1] = location_before_grasp[1] - before_grasp_margin
            # else:
            #     location_before_grasp[1] = location_before_grasp[1] + before_grasp_margin
            #
            # if position[2]-position[5]>0:
            #     location_before_grasp[2] = location_before_grasp[2] - before_grasp_margin
            # else:
            #     location_before_grasp[2] = location_before_grasp[2] + before_grasp_margin
            #
            # self.move_to(location_before_grasp, None)
            #
            # self.rotate_gripper(orientation)

            # tool_rotation_angle = orientation
            #
            # # Avoid collision with floor
            # position = np.asarray(position).copy()
            # position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)
            #
            # # Move gripper to location above grasp target
            # grasp_location_margin = 0.15
            # # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            # location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
            #
            # # Compute gripper position and linear movement increments
            # tool_position = location_above_grasp_target
            # sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            # move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            # move_magnitude = np.linalg.norm(move_direction)
            # move_step = 0.05 * move_direction / move_magnitude
            # num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
            #
            # # Compute gripper orientation and rotation increments
            # sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            # rotation_step = 0.3 if (tool_rotation_angle[1] - gripper_orientation[1] > 0) else -0.3
            # num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))
            #
            # # Simultaneously move and rotate gripper
            # for step_iter in range(max(num_move_steps, num_rotation_steps)):
            #     vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
            #     UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
            #     UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
            #     UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            #     vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2), vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            # vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # self.move_to(position, None)

            # Ensure gripper is open
            # self.open_gripper()

            # Approach grasp target
            # self.move_to(position[0:3], None)
            # # self.ompl_move_to(position, orientation)
            #
            # # Close gripper to grasp target
            # gripper_full_closed = self.close_gripper()
            #
            # # Move gripper to location above grasp target
            # # self.move_to(location_above_grasp_target, None)
            # self.move_to(location_before_grasp, None)
            # self.rotate_gripper([np.pi/2, 0, np.pi/2])

            # Check if grasp is successful
            # gripper_full_closed = self.close_gripper()
            # grasp_success = not gripper_full_closed

            object_positions = np.asarray(self.get_obj_positions())
            for idx in range(len(object_positions)):
                pos = object_positions[idx]
                squared_dist = np.sum((np.array(pos[0:2])-np.array(position[0:2]))**2, axis=0)
                dist = np.sqrt(squared_dist)
                if dist < 0.1:
                    grasp_success = True
                    grasped_object_handle = self.object_handles[idx]
                    vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1, (-0.5, 0.5 + 0.05 * float(idx), 0.1), vrep.simx_opmode_blocking)
                    break
            # # Move the grasped object elsewhere
            # if grasp_success:
            #     object_positions = np.asarray(self.get_obj_positions())
            #     object_positions = object_positions[:, 2]
            #     grasped_object_ind = np.argmax(object_positions)
            #     grasped_object_handle = self.object_handles[grasped_object_ind]
            #     vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1,
            #                                (-0.5, 0.5 + 0.05 * float(grasped_object_ind), 0.1),
            #                                vrep.simx_opmode_blocking)
            # print(obj_pos)
            # print(obj_orientation)

        else:

            # Compute tool orientation from heightmap rotation angle
            # grasp_orientation = [1.0,0.0]
            # if heightmap_rotation_angle > np.pi:
            #     heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
            # tool_rotation_angle = heightmap_rotation_angle/2
            # tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            # tool_orientation_angle = np.linalg.norm(tool_orientation)
            # tool_orientation_axis = tool_orientation/tool_orientation_angle
            # tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
            #
            # # Compute tilted tool orientation during dropping into bin
            # tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
            # tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            # tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            # tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])
            #
            # # Attempt grasp
            # position = np.asarray(position).copy()
            # position[2] = max(position[2] - 0.05, workspace_limits[2][0])
            # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            # tcp_command = "def process():\n"
            # tcp_command += " set_digital_out(8,False)\n"
            # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
            # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.1,self.joint_vel*0.1)
            # tcp_command += " set_digital_out(8,True)\n"
            # tcp_command += "end\n"
            # self.tcp_socket.send(str.encode(tcp_command))
            # self.tcp_socket.close()
            #
            # # Block until robot reaches target tool position and gripper fingers have stopped moving
            # state_data = self.get_state()
            # tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            # timeout_t0 = time.time()
            # while True:
            #     state_data = self.get_state()
            #     new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            #     actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            #     timeout_t1 = time.time()
            #     if (tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - position[j]) < self.tool_pose_tolerance[j] for j in range(3)])) or (timeout_t1 - timeout_t0) > 5:
            #         break
            #     tool_analog_input2 = new_tool_analog_input2
            #
            # # Check if gripper is open (grasp might be successful)
            # gripper_open = tool_analog_input2 > 0.26
            #
            # # # Check if grasp is successful
            # # grasp_success =  tool_analog_input2 > 0.26
            #
            # home_position = [0.49,0.11,0.03]
            # bin_position = [0.5,-0.45,0.1]
            #
            # # If gripper is open, drop object in bin and check if grasp is successful
            # grasp_success = False
            # if gripper_open:
            #
            #     # Pre-compute blend radius
            #     blend_radius = min(abs(bin_position[1] - position[1])/2 - 0.01, 0.2)
            #
            #     # Attempt placing
            #     self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #     self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            #     tcp_command = "def process():\n"
            #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (position[0],position[1],bin_position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc,self.joint_vel,blend_radius)
            #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (bin_position[0],bin_position[1],bin_position[2],tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel,blend_radius)
            #     tcp_command += " set_digital_out(8,False)\n"
            #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
            #     tcp_command += "end\n"
            #     self.tcp_socket.send(str.encode(tcp_command))
            #     self.tcp_socket.close()
            #     # print(tcp_command) # Debug
            #
            #     # Measure gripper width until robot reaches near bin location
            #     state_data = self.get_state()
            #     measurements = []
            #     while True:
            #         state_data = self.get_state()
            #         tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            #         actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            #         measurements.append(tool_analog_input2)
            #         if abs(actual_tool_pose[1] - bin_position[1]) < 0.2 or all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            #             break
            #
            #     # If gripper width did not change before reaching bin location, then object is in grip and grasp is successful
            #     if len(measurements) >= 2:
            #         if abs(measurements[0] - measurements[1]) < 0.1:
            #             grasp_success = True
            #
            # else:
            #     self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #     self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            #     tcp_command = "def process():\n"
            #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
            #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
            #     tcp_command += "end\n"
            #     self.tcp_socket.send(str.encode(tcp_command))
            #     self.tcp_socket.close()
            #
            # # Block until robot reaches home location
            # state_data = self.get_state()
            # tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            # actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            # while True:
            #     state_data = self.get_state()
            #     new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            #     actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            #     if (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            #         break
            #     tool_analog_input2 = new_tool_analog_input2
            grasp_success = True

        return grasp_success

    # def push(self, position, push_target, workspace_limits):
    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.024

            # Compute pushing direction
            push_orientation = [1.0, 0.0]
            push_direction = np.asarray([push_orientation[0] * np.cos(heightmap_rotation_angle) - push_orientation[
                1] * np.sin(heightmap_rotation_angle),
                                         push_orientation[0] * np.sin(heightmap_rotation_angle) + push_orientation[
                                             1] * np.cos(heightmap_rotation_angle)])

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)
            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                         vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                              vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                       (tool_position[0], tool_position[1], tool_position[2]),
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0] * push_length, workspace_limits[0][0]),
                           workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1] * push_length, workspace_limits[1][0]),
                           workspace_limits[1][1])
            push_length = np.sqrt(np.power(target_x - position[0], 2) + np.power(target_y - position[1], 2))

            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True

        else:

            # # Compute tool orientation from heightmap rotation angle
            # push_orientation = [1.0,0.0]
            # tool_rotation_angle = heightmap_rotation_angle/2
            # tool_orientation = np.asarray([push_orientation[0]*np.cos(tool_rotation_angle) - push_orientation[1]*np.sin(tool_rotation_angle), push_orientation[0]*np.sin(tool_rotation_angle) + push_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            # tool_orientation_angle = np.linalg.norm(tool_orientation)
            # tool_orientation_axis = tool_orientation/tool_orientation_angle
            # tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
            #
            # # Compute push direction and endpoint (push to right of rotated heightmap)
            # push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle), 0.0])
            # target_x = min(max(position[0] + push_direction[0]*0.1, workspace_limits[0][0]), workspace_limits[0][1])
            # target_y = min(max(position[1] + push_direction[1]*0.1, workspace_limits[1][0]), workspace_limits[1][1])
            # push_endpoint = np.asarray([target_x, target_y, position[2]])
            # push_direction.shape = (3,1)
            #
            # # Compute tilted tool orientation during push
            # tilt_axis = np.dot(utils.euler2rotm(np.asarray([0,0,np.pi/2]))[:3,:3], push_direction)
            # tilt_rotm = utils.angle2rotm(-np.pi/8, tilt_axis, point=None)[:3,:3]
            # tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            # tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            # tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])
            #
            # # Push only within workspace limits
            # position = np.asarray(position).copy()
            # position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
            # position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
            # position[2] = max(position[2] + 0.005, workspace_limits[2][0] + 0.005) # Add buffer to surface
            #
            # home_position = [0.49,0.11,0.03]
            #
            # # Attempt push
            # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            # tcp_command = "def process():\n"
            # tcp_command += " set_digital_out(8,True)\n"
            # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
            # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (push_endpoint[0],push_endpoint[1],push_endpoint[2],tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
            # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.03)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            # tcp_command += "end\n"
            # self.tcp_socket.send(str.encode(tcp_command))
            # self.tcp_socket.close()
            #
            # # Block until robot reaches target tool position and gripper fingers have stopped moving
            # state_data = self.get_state()
            # while True:
            #     state_data = self.get_state()
            #     actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            #     if all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            #         break
            push_success = True
            time.sleep(0.5)

        return push_success

    def restart_real(self):

        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0, 0.0]
        tool_rotation_angle = -np.pi / 4
        tool_orientation = np.asarray(
            [grasp_orientation[0] * np.cos(tool_rotation_angle) - grasp_orientation[1] * np.sin(tool_rotation_angle),
             grasp_orientation[0] * np.sin(tool_rotation_angle) + grasp_orientation[1] * np.cos(tool_rotation_angle),
             0.0]) * np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation / tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi / 4, 0, 0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * np.asarray(
            tilted_tool_orientation_axis_angle[1:4])

        # Move to box grabbing position
        box_grab_position = [0.5, -0.35, -0.12]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (
        box_grab_position[0], box_grab_position[1], box_grab_position[2] + 0.1, tilted_tool_orientation[0],
        tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        box_grab_position[0], box_grab_position[1], box_grab_position[2], tool_orientation[0], tool_orientation[1],
        tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all(
                    [np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in
                     range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

        # Move to box release position
        box_release_position = [0.5, 0.08, -0.12]
        home_position = [0.49, 0.11, 0.03]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        box_release_position[0], box_release_position[1], box_release_position[2], tool_orientation[0],
        tool_orientation[1], tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        box_release_position[0], box_release_position[1], box_release_position[2] + 0.3, tool_orientation[0],
        tool_orientation[1], tool_orientation[2], self.joint_acc * 0.02, self.joint_vel * 0.02)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (
        box_grab_position[0] - 0.05, box_grab_position[1] + 0.1, box_grab_position[2] + 0.3, tilted_tool_orientation[0],
        tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        box_grab_position[0] - 0.05, box_grab_position[1] + 0.1, box_grab_position[2], tool_orientation[0],
        tool_orientation[1], tool_orientation[2], self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        box_grab_position[0], box_grab_position[1], box_grab_position[2], tool_orientation[0], tool_orientation[1],
        tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        box_grab_position[0] + 0.05, box_grab_position[1], box_grab_position[2], tool_orientation[0],
        tool_orientation[1], tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (
        box_grab_position[0], box_grab_position[1], box_grab_position[2] + 0.1, tilted_tool_orientation[0],
        tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (
        home_position[0], home_position[1], home_position[2], tool_orientation[0], tool_orientation[1],
        tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches home position
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all(
                    [np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

    # def place(self, position, orientation, workspace_limits):
    #     print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

    #     # Attempt placing
    #     position[2] = max(position[2], workspace_limits[2][0])
    #     self.move_to([position[0], position[1], position[2] + 0.2], orientation)
    #     self.move_to([position[0], position[1], position[2] + 0.05], orientation)
    #     self.tool_acc = 1 # 0.05
    #     self.tool_vel = 0.02 # 0.02
    #     self.move_to([position[0], position[1], position[2]], orientation)
    #     self.open_gripper()
    #     self.tool_acc = 1 # 0.5
    #     self.tool_vel = 0.2 # 0.2
    #     self.move_to([position[0], position[1], position[2] + 0.2], orientation)
    #     self.close_gripper()
    #     self.go_home()

    # def place(self, position, heightmap_rotation_angle, workspace_limits):
    #     print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

    #     if self.is_sim:

    #         # Compute tool orientation from heightmap rotation angle
    #         tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

    #         # Avoid collision with floor
    #         position[2] = max(position[2] + 0.04 + 0.02, workspace_limits[2][0] + 0.02)

    #         # Move gripper to location above place target
    #         place_location_margin = 0.1
    #         sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
    #         location_above_place_target = (position[0], position[1], position[2] + place_location_margin)
    #         self.move_to(location_above_place_target, None)

    #         sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
    #         if tool_rotation_angle - gripper_orientation[1] > 0:
    #             increment = 0.2
    #         else:
    #             increment = -0.2
    #         while abs(tool_rotation_angle - gripper_orientation[1]) >= 0.2:
    #             vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + increment, np.pi/2), vrep.simx_opmode_blocking)
    #             time.sleep(0.01)
    #             sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
    #         vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

    #         # Approach place target
    #         self.move_to(position, None)

    #         # Ensure gripper is open
    #         self.open_gripper()

    #         # Move gripper to location above place target
    #         self.move_to(location_above_place_target, None)

    #         place_success = True
    #         return place_success
