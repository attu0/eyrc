#!/usr/bin/env python3

'''
*****************************************************************************************
*
*                   ===============================================
*                            StrataCobot (SC) Theme (eYRC 2026-27)
*                   ===============================================
*
*  Launch file for Task 3A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Nodes:            starts ore_tf_publisher, arm manipulation, ebot_planner_node, ebot_nav_node


################### IMPORT MODULES #######################

from launch import LaunchDescription
from launch_ros.actions import Node


##################### TASK CONSTANTS #######################

package_name = 'algorithms'

scripts = [
    'task3A_perception.py',
    'task3A_manipulation.py',
    'task3A_planning.py',
    'task3A_navigation.py',
]


##################### FUNCTION DEFINITION #######################

def generate_launch_description():
    '''
    Description:    Starts the four Task 3A nodes together. It does not start the world:
                    launch that on its own first, and start this only once the evaluator
                    has printed Ready.

    Returns:
        LaunchDescription:  the four nodes, each logging to the terminal
    '''

    nodes = [
        Node(package=package_name, executable=script, output='screen', emulate_tty=True)
        for script in scripts
    ]

    return LaunchDescription(nodes)