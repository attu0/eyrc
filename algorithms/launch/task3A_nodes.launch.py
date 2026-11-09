#!/usr/bin/env python3

'''
*****************************************************************************************
*
*                   ===============================================
*                            StrataCobot (SC) Theme (eYRC 2026-27)
*                   ===============================================
*
*  Launch file for Task 3A of StrataCobot (SC) Theme (eYRC 2026-27), with logging.
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
# Filename:         task3A_nodes.launch.py
# Functions:        [ generate_launch_description ]
# Nodes:            starts task3A_perception, task3A_manipulation, task3A_planning, task3A_navigation


################### IMPORT MODULES #######################

import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, LogInfo
from launch_ros.actions import Node


##################### TASK CONSTANTS #######################

package_name = 'algorithms'

scripts = [
    'task3A_perception.py',
    'task3A_manipulation.py',
    'task3A_planning.py',
    'task3A_navigation.py',
]

# Every run gets its own folder: ~/task3a_logs/run_<date>_<time>/
log_root = os.path.expanduser('~/task3a_logs')
run_dir = os.path.join(log_root, 'run_' + datetime.now().strftime('%Y%m%d_%H%M%S'))


##################### FUNCTION DEFINITION #######################

def generate_launch_description():
    '''
    Description:    Starts the four Task 3A nodes together and keeps their output both on
                    the terminal and in files under run_dir (one <node>-stdout.log per node,
                    plus launch.log). It does not start the world: launch that on its own
                    first, and start this only once the evaluator has printed Ready.

    Returns:
        LaunchDescription:  the four nodes, each logging to the terminal and to a file
    '''

    os.makedirs(run_dir, exist_ok=True)

    actions = [
        SetEnvironmentVariable('ROS_LOG_DIR', run_dir),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '0'),
        LogInfo(msg=f'Task 3A logs are written to {run_dir}'),
    ]

    actions += [
        Node(package=package_name, executable=script, output='both', emulate_tty=True)
        for script in scripts
    ]

    return LaunchDescription(actions)