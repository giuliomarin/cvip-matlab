# """
# Convert images from float to 8 bit.
# """
# 
# import cv2
# import os
# import shutil
# import sys
# from matplotlib import pyplot as plt
# 
# # Path the directory of this file
# currPath = os.path.dirname(os.path.realpath(__file__))
# os.chdir(currPath)
# print currPath
# 
# # import Utils
# sys.path.append(currPath)
# import utils
# 
# if __name__ == '__main__':
#     
#     imgPath = 'smaxles/images/depth.png'
#     img = utils.imread32f(imgPath)
#     plt.ion()
#     plt.imshow(img)
#     plt.xticks([]), plt.yticks([])
#     plt.ioff()
#     plt.show()
#     print 'done'

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Path of the directory of this file
currPath = os.path.dirname(os.path.realpath(__file__))
os.chdir(currPath)

# import utils
sys.path.append(currPath)
import utils

def invertColorMap(ax):
    currCM = ax.get_cmap().name
    if currCM.endswith('_r'):
        currCMinv = currCM[0:-2]
    else:
        currCMinv = currCM + '_r'
    ax.set_cmap(currCMinv)

def createImagesc(img):
    
    def press(event):
        sys.stdout.flush()
        if event.key == 'escape':
            sys.exit(0)
        elif event.key == ' ':
            # img8bit = cv2.convertScaleAbs(img, 255.0 / (currMax - currMin), - currMin * 255.0 / (currMax - currMin))
            print 'saved'
    
    minVal, maxVal, _, _ = cv2.minMaxLoc(img)
    print 'Min [%f] - Max [%f]\n' % (minVal, maxVal)
    
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.subplots_adjust(0.15, 0.1, 0.9, 0.98)
    
    plt.xticks([]), plt.yticks([])
    im = plt.imshow(img)
    im.set_clim(minVal, maxVal)
    
    axcolor = 'lightgoldenrodyellow'
    BAR_HEIGHT = 0.03
    axmin = plt.axes([0.2, 0.2 * BAR_HEIGHT, 0.7, BAR_HEIGHT], axisbg = axcolor)
    axmax = plt.axes([0.2, BAR_HEIGHT + 0.4 * BAR_HEIGHT, 0.7, BAR_HEIGHT], axisbg = axcolor)
    smin = Slider(axmin, 'Min', minVal, maxVal, valinit = minVal)
    smax = Slider(axmax, 'Max', minVal, maxVal, valinit = maxVal)
    smin.slidermax = smax
    smax.slidermin = smin
    
    def update(event):
        currMin = smin.val
        currMax = smax.val
        print 'Min [%f] - Max [%f]' % (currMin, currMax)
        im.set_clim(min(currMin, currMax), max(currMin, currMax))
        fig.canvas.draw()
    smin.on_changed(update)
    smax.on_changed(update)
    
    invertax = plt.axes([0.02, 0.2 * BAR_HEIGHT, 0.1, (2.2) * BAR_HEIGHT])
    button = Button(invertax, 'Invert', color=axcolor, hovercolor='0.975')
    
    def invert(event):
        invertColorMap(im)
        fig.canvas.draw()
    button.on_clicked(invert)
    
    rax = plt.axes([0.02, (2.4) * BAR_HEIGHT, 0.1, 0.1], axisbg=axcolor)
    radio = RadioButtons(rax, ('jet', 'gray'), active=0)
    
    def colorfunc(label):
        im.set_cmap(label)
        fig.canvas.draw()
    radio.on_clicked(colorfunc)
    
    plt.show()
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv.append('samples/images/depth.png')
    
    imgPath = sys.argv[1]
    img = utils.imread32f(imgPath)
    createImagesc(img)