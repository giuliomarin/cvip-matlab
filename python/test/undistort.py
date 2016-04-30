import numpy as np
import cv2
import glob
import sys

#########################
# Parameters
#########################

imgfilebase = '/Users/giulio/Dropbox (Personal)/Temporary/calibration/0/%s'
imgfile = imgfilebase % 'img0_*.png'
N_CHECKERS = (10, 8)  # (points_per_row,points_per_colum)

# Visualization
H_IMGS = 480  # -1 for original size


#########################
# Functions
#########################

def resizeimgh(img, h):
    if h <= 0:
        return img
    else:
        return cv2.resize(img, (int(float(img.shape[1]) / img.shape[0] * h), h))


def resizeimgw(img, w):
    if w <= 0:
        return img
    else:
        return cv2.resize(img, (w, int(float(img.shape[0]) / img.shape[1] * w)))


def tostr(arr, prec = 3):
    strout = ''
    for l in arr:
        for c in l:
            strout += ('%.' + str(prec) + 'f ') % c
        strout += '\n'
    return strout

##########################
# Load calibration
##########################

try:
    with open(imgfilebase % 'calib.txt', 'r') as calibfile:
        # K
        K = []
        calibstr = calibfile.readline()
        K = np.append(K, [float(x) for x in calibfile.readline().split()])
        K = np.append(K, [float(x) for x in calibfile.readline().split()])
        K = np.append(K, [float(x) for x in calibfile.readline().split()])
        K = K.reshape(3, 3)

        # D
        D = []
        calibstr = calibfile.readline()
        calibstr = calibfile.readline()
        D = np.append(D, [float(x) for x in calibfile.readline().split()])
except:
    sys.exit('Unable to parse calibration file')

##########################
# Undistortion
##########################

print '\nTest undistortion'
images = glob.glob(imgfile)
img = cv2.imread(images[0])
h, w = img.shape[:2]
Knew, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
print 'Knew:\n%s' % tostr(Knew)
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, Knew, (img.shape[1], img.shape[0]), 5)
imgund = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imwrite(imgfilebase % 'imgund.png', imgund)
bothimg = resizeimgw(np.hstack((img, imgund)), 1024)
cv2.imshow('original & undistorted', bothimg)
cv2.waitKey(0)

