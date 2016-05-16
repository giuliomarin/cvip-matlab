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
import dataio

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

    img /= max(img.flatten())

    minVal, maxVal, _, _ = cv2.minMaxLoc(img)
    print 'Min [%f] - Max [%f]\n' % (minVal, maxVal)

    fig_hist, ax_hist = plt.subplots()
    fig_hist.canvas.mpl_connect('key_press_event', press)
    plt.subplots_adjust(0.1, 0.05, 1.0, 0.98)
    im_hist = plt.hist(img.flatten(), 50, histtype = 'stepfilled', normed = True)

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
        img_crop = img.copy()
        img_crop[img_crop > currMax] = 1.0
        img_crop[img_crop < currMin] = 0.0
        print 'Min [%f] - Max [%f]' % (currMin, currMax)
        im.set_data(img_crop)
        im.set_clim(min(currMin, currMax), max(currMin, currMax))
        plt.figure(fig.number)
        fig.canvas.draw()
        plt.figure(fig_hist.number)
        ax_hist.cla()
        ax_hist.hist(img_crop.flatten(), 50, histtype = 'stepfilled', normed = True)
        fig_hist.canvas.draw()
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
    img = dataio.imread32f(imgPath)
    createImagesc(img)
