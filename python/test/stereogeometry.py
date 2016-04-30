"""Characteristics of a stereo system."""

import matplotlib.pyplot as plt
import numpy as np

__author__ = "Giulio Marin"
__email__ = "giulio.marin@me.com"
__date__ = "2016/04/30"

######################
# Parameters
######################

# Visualization
plt.ion()
LINE_W = 2

# Stereo cameras
SIZE_SENSOR = (480, 640)
FOCAL_H_PXL = 500  # [mm]
FOCAL_V_PXL = FOCAL_H_PXL  # [mm]

BASELINE = [50.0, 70.0]  # [mm]
SUBPIXEL = [1., 1./16]  # pixel

print '------------------'
print 'Parameters'
print 'Sensor size [H,W]] = [%d,%d] pixel' % SIZE_SENSOR
print 'Focal length [H,V] = [%.1f,%.1f] pixel' % (FOCAL_H_PXL, FOCAL_V_PXL)

######################
# Field of View
######################

FOV_H = 2 * np.arctan2(SIZE_SENSOR[1] / 2, FOCAL_H_PXL)
FOV_V = 2 * np.arctan2(SIZE_SENSOR[0] / 2, FOCAL_V_PXL)

print '\n------------------'
print 'Field of view'
print 'FOV_H = %.1f degrees' % (FOV_H / np.pi * 180)
print 'FOV_V = %.1f degrees' % (FOV_V / np.pi * 180)

######################
# Depth resolution
######################

Z_VEC = np.arange(300, 2001, 10)

plt.figure()
for b in BASELINE:
    for s in SUBPIXEL:
        deltaZ = Z_VEC**2 / (FOCAL_H_PXL * b - Z_VEC * s) * s
        currP = plt.plot(Z_VEC, deltaZ, label = ('Baseline=%d\nSubpixel=%.3f' % (b, s)))

plt.title('Depth resolution at different distance')
plt.xlabel('Distance [mm]')
plt.ylabel('Depth resolution [mm]')
plt.legend(loc='upper left')
plt.xlim((Z_VEC[0], Z_VEC[-1]))
plt.grid()
for l in plt.gca().lines:
    plt.setp(l, linewidth = LINE_W)
for l in plt.gca().get_legend().get_lines():
    plt.setp(l, linewidth = LINE_W)
plt.show()
plt.savefig('/Users/giulio/Desktop/depthres.png', transparent = True)

######################
# Disparity vs Depth
######################

Z_VEC = np.arange(300, 2001, 10)

plt.figure()
for b in BASELINE:
    disparity = b * FOCAL_H_PXL / Z_VEC
    plt.plot(Z_VEC, disparity, label = ('Baseline=%d' % b))

plt.title('Disparity at different distance')
plt.xlabel('Distance [mm]')
plt.ylabel('Disparity [pixel]')
plt.legend(loc='upper right')
plt.xlim((Z_VEC[0], Z_VEC[-1]))
plt.grid()
for l in plt.gca().lines:
    plt.setp(l, linewidth = LINE_W)
for l in plt.gca().get_legend().get_lines():
    plt.setp(l, linewidth = LINE_W)
plt.show()
plt.savefig('/Users/giulio/Desktop/disparitydepth.png', transparent = True)

plt.ioff()
plt.show()
