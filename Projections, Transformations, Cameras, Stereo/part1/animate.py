import numpy as np
from math import *
import os
import math

# The following tries to avoid a warning when run on the linux machines via ssh.
if os.environ.get('DISPLAY') is None:
     import matplotlib 
     matplotlib.use('Agg')
       
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# All of the code in this file is just a starting point. Feel free
# to edit anything as you make your animation program.

#Function to find the projection matrix for the points in the scene for each frame
def projection_matrix(f, alpha, beta, gamma, tx, ty, tz):

    #Built these 3 functions for rotations on X, Y, Z planes using the formula provided
    def rotation_matrix_x(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])

    def rotation_matrix_y(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])

    def rotation_matrix_z(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

    #Helpful side function instead of directly doing dot product
    #Helps with the multiplication of 0 padded 4x4 matrices
    def matrix_multiply(A, B):
        return np.dot(A, B)
    
    #alpha, beta, gamma, are defined for rotation planes side to side horizontally (x), up down (y) and side to side vertically (z)
    R_alpha = rotation_matrix_x(alpha)
    R_beta = rotation_matrix_y(beta)
    R_gamma = rotation_matrix_z(gamma)

    #Calculate the value for the 3 rotations as per the formula for further simplification
    R = matrix_multiply(R_alpha, matrix_multiply(R_beta, R_gamma))

    #T matrix as per the formula provided with a padded row for multiplication
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

    #Convert the differently dimensioned matrices into 4x4 for dot product calculation to make it homogeneous
    #0, 0, 0, 1 are added as padding on both last column and row of the matrix
    # referred suggestion of using hstack and vstack from https://stackoverflow.com/questions/53071212/stacking-numpy-arrays-with-padding
    R_homogeneous = np.hstack((R, np.zeros((3, 1))))
    R_homogeneous = np.vstack((R_homogeneous, np.array([0, 0, 0, 1])))
    # print(R_homogeneous, "\n")

    #P array defined as per the formula provided with padding column for multiplication
    P = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, 1, 0]
    ])

    #calculate the projection matrix as per the formula
    Pi = matrix_multiply(P, matrix_multiply(R_homogeneous, T))
    return Pi

# this function gets called every time a new frame should be generated.
#Animating with python using frames: https://www.youtube.com/watch?v=51rCh2Do2EE&ab_channel=Dave%27sSpace
#Animating with if elif and else: https://stackoverflow.com/questions/55560284/pygame-the-best-way-to-slow-down-the-framerate-of-a-certain-animation-without-pa
def animate_above(frame_number):
    global tx, ty, tz, yaw, tilt, twist
    #yaw used here to turn the flight left or right
    #tilt used here to turn the flight up or down
    #twist used here to roll the flight on its axis
    
    #Move on the runway
    if frame_number < 20:
        ty += 10
    
    # 1. Take off and ascend
    elif frame_number < 40:
        ty += 10
        tz -= 4 #start ascending
        tilt -= pi / 200  # Add a slight lift to the camera angle

    # 2. Turn right and move straight for a few frames
    elif frame_number < 60:
        yaw += pi / 30
        tx += 30 * sin(yaw) #turn
        ty += 30 * cos(yaw)
        tz += 1
        twist -= pi / 45  # Add banking roll to the right

    # 3. Turn right and move straight for a few frames
    elif frame_number < 100:
        yaw += pi / 30
        tx += 30 * sin(yaw) #keep turning smoothly
        ty += 30 * cos(yaw)
        twist += pi / 90  # Gradually reduce banking roll
        tilt += pi / 300  # Add a slight lift to the camera angle

    #Start descending for landing    
    elif frame_number < 130:
        ty += 10
        tz += 0.5 #descend
        tilt += pi / 300  # Add a slight dip to the camera angle

    # 6. Descend and land on the runway
    elif frame_number < 140:
        ty += 5
        tz += 3 #rapid descent

    #slow down to a stop
    else:
        ty += 2 #slow down

    #focal length as per the description
    f = 0.002

    #reassign to send to the projection matrix function for clarity
    alpha, beta, gamma = tilt, twist, yaw
    
    #Turned the camera around to show whats happening in the front
    gamma += math.pi

    #Getting the projection matrix for each frame
    Pi = projection_matrix(f, alpha, beta, gamma, tx, ty, tz)

    #points from the project matrix for plotting stored into pts2
    pts2 = []
    for p in pts3:
        #A new coordinate 1 is added to the original 3D coordinates in P to make it homogeneous for next multiplication
        p_homogeneous = np.array([p[0], p[1], p[2], 1])
        #calculate the projected points from P by multiplying the P_homogeneous with projection matrix
        p_projected = np.dot(Pi, p_homogeneous)

        #This finds if the point calculated is infront (z being positive) of the camera and appends it into the pts2 list in 2D by doing xy/z
        if p_projected[2] > 0:
            pts2.append([p_projected[0] / p_projected[2], p_projected[1] / p_projected[2]])

    plt.cla()
    plt.gca().set_xlim([-0.002, 0.002])
    plt.gca().set_ylim([0.002, -0.002])

    #plotting from the list
    #Reference https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
    line, = plt.plot([p[0] for p in pts2], [p[1] for p in pts2], linestyle="", marker=".", markersize=2)
    return line,

with open("airport.pts", "r") as f:
    pts3 = [[float(x) for x in l.split(" ")] for l in f.readlines()]

#initial points for the airplane
(tx, ty, tz) = (0, 0, -20)
(yaw, tilt, twist) = (0, pi / 2, 0)

fig, ax = plt.subplots()
frame_count = 144
ani = animation.FuncAnimation(fig, animate_above, frames=range(0, frame_count))

# ani.save("movie6.gif", writer='pillow', fps=7)
ani.save("movie.mp4", writer='ffmpeg', fps=7)

# uncomment if you want to display the movie on the screen (won't work on the
# remote linux servers if you are connecting via ssh)
#plt.show()


