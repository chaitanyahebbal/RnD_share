# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:11:06 2016

@author: chaitanya
"""

import numpy as np

def construct_R_matrix(r,d,s):
    R = np.array([[1, -1, -(d+s)],
                  [1, 1, (d+s)],
                  [1, 1, -(d+s)],
                  [1, -1, (d+s)]])
    R = np.dot(1/r,R)
    return R

def calculate_wheel_velocities(r,v):
    #r = construct_R_matrix(0.050,0.150,0.2355)
    #v = np.array([longitudinal_velocity,transversal_velocity,angular_velocity])
    theta_dot = np.dot(r,v)
    return theta_dot


def load_log(filename):
    data = np.genfromtxt(filename,skip_header = 1,delimiter = ';')
    #velocity_values = data[:,17:23]
    #longitudinal_velocity = data[:,17]
    #transversal_velocity = data[:,18]
    #angular_velocity = data[:,19]
    commanded_velocities = data[:,17:20]
    computed_velocities = data[:,13:17] 
    return commanded_velocities,computed_velocities

#for testing purposes
if __name__ == '__main__' :
    #a = calculate_wheel_velocities(0.050,0.150,0.2355)
    r = construct_R_matrix(0.050,0.150,0.2355)
    data,expected_output = load_log('/home/chaitanya/rnd_project/youbot-data/alex/aiciss_logs_06_05_2015/06_05_2015__14_48_51.log')
    output = []
    for v in data:
        theta = calculate_wheel_velocities(r,v)
        output.append(theta)
    
    output = np.array(output)
    
    
    #print output[0]
    