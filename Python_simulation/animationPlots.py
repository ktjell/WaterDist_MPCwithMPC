#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:21:03 2023

@author: kst
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

x0  = []
y0,y1,y2,y3,y4,y5,y6,y7 = [],[],[],[],[],[],[],[]
col_names = list(mcolors.TABLEAU_COLORS.keys())

dots1 = np.random.randint(0,1000,simu.ite) / 1000
dots2 = np.random.randint(0,1000,simu.ite) / 1000

T = np.mod(simu.TIME, simu.TIME[simu.M])
steps = np.where(T[:simu.ite] == [0, 6*60, 12*60, 18*60])[0]
ticks = np.tile(np.array(['00', '06', '12', '18']),int(len(steps)/4))

f, ax = plt.subplots(5, sharex=True, figsize = (15,12))

ax[0].set_xticks(simu.TIME[steps].flatten(),ticks)
ax[0].set_ylabel('Encrypted Cloud')
# line_dots1 = ax[0].plot([],[])
# line_dots2 = ax[0].plot([],[])

ax[3].set_ylabel('Tank level [m]')
# line_h, = ax[1].plot([],[], label='Level in tank')
ax[3].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*tank.hmin, 'r', linestyle = 'dashed')
ax[3].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*tank.hmax, 'r', linestyle = 'dashed')

# line_q1, = ax[1].plot([], [], label = 'q1')
# line_q2, = ax[1].plot([], [], label = 'q2')
ax[1].set_ylabel('Flow pump 1 [$m^3/h$]')
ax[1].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[0].Qmax, 'r', linestyle = 'dashed')
ax[1].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*0, 'r', linestyle = 'dashed')
# line_d, = ax[1].plot([1], [1], linewidth= 0.5,  label = 'Demand')
# ax[1].set_ylabel('q1, q2, demand')

ax[2].set_ylabel('Flow pump 2 [$m^3/h$]')
ax[2].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[1].Qmax, 'r', linestyle = 'dashed')
ax[2].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*0, 'r', linestyle = 'dashed')
# ax[2].set_ylabel('Extraction per day')
# line_Extq1, = ax[2].plot([], [],)
ax[4].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[0].Vmax, 'r', linestyle = 'dashed')
# line_Extq2, = ax[2].plot([], [],)
# ax[2].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[1].Vmax, 'b', linestyle = 'dashed')
ax[4].set_ylabel('Extraction per day')
# line_p1, = ax[3].plot([], [],)
# line_p2, = ax[3].plot([], [],)
# ax[3].set_xlabel('Time')
# ax[3].set_ylabel('Pressure')
# f.xticks(steps, ticks)#, rotation='vertical')
f.tight_layout()


# function that draws each frame of the animation
def animate(i):
    x0.append(simu.TIME[i])
    y0.append(dots1[i])
    y1.append(dots2[i])
    y2.append(h[i])
    y3.append(q1[i])
    y4.append(q2[i])
    y5.append(cum_q1[i])
    y6.append(cum_q2[i])
    y7.append(simu.d[i])
    
    # ax.clear()
    ax[0].scatter(x0, y0, c = col_names[0])
    ax[0].scatter(x0,y1, c= col_names[1])
    ax[3].plot(x0, y2, col_names[0])
    ax[1].plot(x0,y3, col_names[2])
    # ax[2].plot(x0, y7, col_names[1], linewidth = 0.5)
    ax[2].plot(x0,y4,col_names[2])
    # ax[3].plot(x0, y7, col_names[1], linewidth = 0.5)
    ax[4].plot(x0,y5,col_names[0])
    ax[4].plot(x0,y6,col_names[1])
    # ax.set_xlim([0,])
    # ax.set_ylim([tank.hmin,tank.hmax])

ani = FuncAnimation(f, animate, frames=len(q1), interval=150, repeat=False)

plt.show()
# # # saves the animation in our desktop
# ani.save('FullPlotite3days.mp4', writer = 'ffmpeg', fps = 5)


# # create empty lists for the x and y data
# x = []
# y = []

# # create the figure and axes objects
# fig, ax = plt.subplots(figsize=(15, 10))
# plt.plot(simu.TIME[:simu.ite], np.ones(simu.ite)*sups[0].Qmax, 'r', linestyle = 'dashed')
# T = np.mod(simu.TIME, simu.TIME[simu.M])
# steps = np.where(T[:simu.ite] == [0, 6*60, 12*60, 18*60])[0]
# ticks = np.tile(np.array(['00', '06', '12', '18']),int(len(steps)/4))
# ax.set_xticks(simu.TIME[steps].flatten(),ticks)
# ax.set_xlabel('time')
# ax.set_ylabel('[$m^3/h$]')
# ax.set_title('Pump 1, flow')

# # function that draws each frame of the animation
# def animate(i):
#     x.append(simu.TIME[i])
#     y.append(q1[i])

#     # ax.clear()
#     ax.plot(x, y, col_names[0])
#     # ax.set_xlim([0,])
#     # ax.set_ylim([tank.hmin,tank.hmax])

# ani = FuncAnimation(fig, animate, frames=len(q1), interval=70, repeat=False)

# plt.show()


# # # saves the animation in our desktop
# # ani.save('tanklevel.mp4', writer = 'ffmpeg', fps = 30)






# create empty lists for the x and y data
# x = []
# y = []

# # create the figure and axes objects
# fig, ax = plt.subplots(figsize=(15, 10))
# plt.plot(simu.TIME[:simu.ite], np.ones(simu.ite)*tank.hmin, 'r', linestyle = 'dashed')
# plt.plot(simu.TIME[:simu.ite], np.ones(simu.ite)*tank.hmax, 'r', linestyle = 'dashed')
# T = np.mod(simu.TIME, simu.TIME[simu.M])
# steps = np.where(T[:simu.ite] == [0, 6*60, 12*60, 18*60])[0]
# ticks = np.tile(np.array(['00', '06', '12', '18']),int(len(steps)/4))
# ax.set_xticks(simu.TIME[steps].flatten(),ticks)
# ax.set_xlabel('time')
# ax.set_ylabel('[m]')
# ax.set_title('Tank level')

# # function that draws each frame of the animation
# def animate(i):
#     x.append(simu.TIME[i])
#     y.append(h[i])

#     # ax.clear()
#     ax.plot(x, y, col_names[0])
#     # ax.set_xlim([0,])
#     # ax.set_ylim([tank.hmin,tank.hmax])

# ani = FuncAnimation(fig, animate, frames=len(h), interval=70, repeat=False)

# plt.show()


# # saves the animation in our desktop
# ani.save('tanklevel.mp4', writer = 'ffmpeg', fps = 30)


