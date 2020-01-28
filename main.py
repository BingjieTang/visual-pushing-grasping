#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils


def main(args):
    # --------------- Setup options ---------------
    is_sim = args.is_sim  # Run in simulation?
    obj_mesh_dir = os.path.abspath(
        args.obj_mesh_dir) if is_sim else None  # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None  # Number of objects to add to simulation
    tcp_host_ip = args.tcp_host_ip if not is_sim else None  # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None  # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001,
                                                                           0.4]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255,
                                                                       -0.1]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    method = args.method  # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    push_rewards = args.push_rewards if method == 'reinforcement6d' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay  # Use prioritized experience replay?
    heuristic_bootstrap = args.heuristic_bootstrap  # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, test_preset_file)

    # print(robot.get_push_sample())
    # exit(-1)

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    # logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    # logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    # if continue_logging:
    #     trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.5 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'best_push_ind': None,
                          'best_grasp_ind': None,
                          'push_success': False,
                          'grasp_success': False,
                          'grasp_failure_count': 0}

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # Determine whether grasping or pushing should be executed based on network predictions
                best_push_conf = np.max(push_predictions)
                best_grasp_conf = np.max(grasp_predictions)
                print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))
                nonlocal_variables['primitive_action'] = 'grasp'
                explore_actions = False
                if not grasp_only:
                    if is_testing and method == 'reactive':
                        if best_push_conf > 2 * best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    else:
                        if best_push_conf > best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    explore_actions = np.random.uniform() < explore_prob
                    if explore_actions:  # Exploitation (do best action) vs exploration (do other action)
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0, 2) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))

                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
                # NOTE: typically not necessary and can reduce final performance.
                if heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'push' and no_change_count[0] >= 2:
                    print('Change not detected for more than two pushes. Running heuristic pushing.')
                    nonlocal_variables['best_push_ind'] = trainer.push_heuristic(valid_depth_heightmap)
                    no_change_count[0] = 0
                    predicted_value = push_predictions[nonlocal_variables['best_push_ind']]
                    use_heuristic = True
                elif heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'grasp' and no_change_count[1] >= 2:
                    print('Change not detected for more than two grasps. Running heuristic grasping.')
                    nonlocal_variables['best_grasp_ind'] = trainer.grasp_heuristic(valid_depth_heightmap)
                    no_change_count[1] = 0
                    predicted_value = grasp_predictions[nonlocal_variables['best_grasp_ind']]
                    use_heuristic = True
                else:
                    use_heuristic = False

                    # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                    if nonlocal_variables['primitive_action'] == 'push':
                        nonlocal_variables['best_push_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                        predicted_value = np.max(push_predictions)
                    elif nonlocal_variables['primitive_action'] == 'grasp':
                        nonlocal_variables['best_grasp_ind'] = np.argmax(grasp_predictions)
                        predicted_value = np.max(grasp_predictions)

                # Compute 3D position of pixel
                # print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                primitive_position = None
                best_rotation_angle = None
                if nonlocal_variables['primitive_action'] == 'push':
                    best_rotation_angle = np.deg2rad(nonlocal_variables['best_push_ind'][0] * (360.0 / trainer.model.num_rotations))
                    best_pix_x = nonlocal_variables['best_push_ind'][2]
                    best_pix_y = nonlocal_variables['best_push_ind'][1]
                    primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                                          best_pix_y * heightmap_resolution + workspace_limits[1][0],
                                          valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    best_idx = nonlocal_variables['best_grasp_ind']
                    primitive_position = grasp_samples[best_idx]

                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                change_detected = False

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'] = robot.push(primitive_position, best_rotation_angle, workspace_limits)
                    print('Push successful: %r' % (nonlocal_variables['push_success']))
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    nonlocal_variables['grasp_success'] = robot.grasp(primitive_position[0:3], primitive_position[3:6], workspace_limits)
                    if nonlocal_variables['grasp_success'] == False:
                        nonlocal_variables['grasp_failure_count'] = nonlocal_variables['grasp_failure_count'] + 1
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                               robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Reset simulation or pause real-world training if table is empty
        if robot.check_table_empty() == True :
            if is_sim:
                print('Not enough objects in view! Repositioning objects.')
                robot.restart_sim()
                robot.add_objects()
                nonlocal_variables['grasp_failure_count'] = 0
                if is_testing:  # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))

        if nonlocal_variables['grasp_failure_count'] > 9:
            if is_sim:
                print('Failed 10 times. Repositioning objects.')
                nonlocal_variables['grasp_failure_count'] = 0
                robot.restart_sim()
                robot.add_objects()

        # push_samples, grasp_samples = robot.get_push_sample(), robot.get_grasp_sample()
        grasp_samples = robot.get_grasp_sample()

        # print ("\n main.py after robot.get_grasp_sample: ", grasp_samples)

        if not exit_called:
            # Run forward pass with network to get affordances
            push_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap, depth_heightmap, grasp_samples, is_volatile=True)
            # push_predictions, grasp_predictions, state_feat = trainer.forward(push_samples, grasp_samples, is_volatile=True)

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 300
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            # Compute training labels
            label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_push_success,
                                                                     prev_grasp_success,
                                                                     change_detected, prev_push_predictions,
                                                                     prev_grasp_predictions,
                                                                     color_heightmap, valid_depth_heightmap,
                                                                     grasp_samples)

            # Backpropagate
            if prev_primitive_action == 'grasp':
                trainer.backprop(color_heightmap, depth_heightmap, prev_grasp_sample, prev_primitive_action, prev_best_grasp_ind, label_value)
            elif prev_primitive_action == 'push':
                trainer.backprop(color_heightmap, depth_heightmap, prev_grasp_sample, prev_primitive_action, prev_best_push_ind, label_value)

            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration), 0.1) if explore_rate_decay else 0.5

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_push_ind = nonlocal_variables['best_push_ind']
        prev_best_grasp_ind = nonlocal_variables['best_grasp_ind']

        # if len(push_samples) > 0:
        #     prev_push_sample = push_samples.copy()
        if len(grasp_samples) > 0:
            prev_grasp_sample = grasp_samples.copy()
        # if prev_primitive_action == 'push':
        #     prev_best_pose_ind = np.argmax(prev_push_predictions)
        # if prev_primitive_action == 'grasp':
        #     prev_best_pose_ind = np.argmax(prev_grasp_predictions)

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,
                        help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,
                        help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,
                        help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,
                        help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement6d',
                        help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,
                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store',
                        default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,
                        help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,
                        help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,
                        help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,
                        help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,
                        help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
