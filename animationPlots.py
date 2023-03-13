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
col_names = list(mcolors.TABLEAU_COLORS.keys())
# create empty lists for the x and y data
x = []
y = []

# create the figure and axes objects
fig, ax = plt.subplots()
plt.plot(simu.TIME[:simu.ite], np.ones(simu.ite)*tank.hmin, 'r', linestyle = 'dashed')
plt.plot(simu.TIME[:simu.ite], np.ones(simu.ite)*tank.hmax, 'r', linestyle = 'dashed')
T = np.mod(simu.TIME, simu.TIME[simu.M])
steps = np.where(T[:simu.ite] == [0, 6*60, 12*60, 18*60])[0]
ticks = np.tile(np.array(['00', '06', '12', '18']),int(len(steps)/4))
ax.set_xticks(simu.TIME[steps].flatten(),ticks)
ax.set_xlabel('time')
ax.set_ylabel('[m]')
ax.set_title('Tank level')

# function that draws each frame of the animation
def animate(i):
    x.append(simu.TIME[i])
    y.append(h[i])

    # ax.clear()
    ax.plot(x, y, col_names[0])
    # ax.set_xlim([0,])
    # ax.set_ylim([tank.hmin,tank.hmax])

ani = FuncAnimation(fig, animate, frames=72, interval=100, repeat=False)

plt.show()


# # saves the animation in our desktop
# anim.save('growingCoil.mp4', writer = 'ffmpeg', fps = 30)