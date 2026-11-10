#!/usr/bin/env python3

'''
*****************************************************************************************
*
*                   ===============================================
*                            StrataCobot (SC) Theme (eYRC 2026-27)
*                   ===============================================
*
*  Launch file for Task 3B of StrataCobot (SC) Theme (eYRC 2026-27), with logging.
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 1449 ]
# Author List:      [ Atharv Mudse, Abhay Potdar, Riddhi Wagh, Yash Waje ]
# Filename:         task3B_nodes.launch.py
# Functions:        [ generate_launch_description ]
# Nodes:            starts task3B_perception, task3B_manipulation, task3B_planning, task3B_navigation


################### IMPORT MODULES #######################

import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, LogInfo
from launch_ros.actions import Node


##################### TASK CONSTANTS #######################

package_name = 'algorithms'

scripts = [
    'task3B_perception.py',
    'task3B_manipulation.py',
    'task3B_planning.py',
    'task3B_navigation.py',
]

# Every run gets its own folder: ~/task3b_logs/run_<date>_<time>/
log_root = os.path.expanduser('~/task3b_logs')
run_dir = os.path.join(log_root, 'run_' + datetime.now().strftime('%Y%m%d_%H%M%S'))


##################### FUNCTION DEFINITION #######################

def generate_launch_description():
    '''
    Description:    Starts the four Task 3B nodes together and keeps their output both on
                    the terminal and in files under run_dir.
    '''

    os.makedirs(run_dir, exist_ok=True)

    actions = [
        SetEnvironmentVariable('ROS_LOG_DIR', run_dir),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '0'),
        LogInfo(msg=f'Task 3B logs are written to {run_dir}'),
    ]

    actions += [
        Node(package=package_name, executable=script, output='both', emulate_tty=True)
        for script in scripts
    ]

    return LaunchDescription(actions)