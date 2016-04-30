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
# Calibration
##########################

# Load images
images = glob.glob(imgfile)

if len(images) == 0:
    sys.exit('No image matches: %s' % imgfile)

# Prepare object points
objp = np.zeros((np.prod(N_CHECKERS), 3), np.float32)
objp[:, :2] = np.mgrid[0:N_CHECKERS[0], 0:N_CHECKERS[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPoints = []  # 2d points in image plane.

# Info
n_valid = 0

# Convergence criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)

# Loop on images
for fname in images:
    print 'Image: %s' % fname,
    # load image and convert to grayscale
    img = cv2.imread(fname)
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    cv2.imshow('curr img', img)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, N_CHECKERS, None, flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

    # If not found skip impage
    if ret is not True:
        print ' [not found]'
    else:
        print ' [found]'
        n_valid += 1

        # refinine image points using subpixel accuracy
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # add object points and image points
        objPoints.append(objp)
        imgPoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, N_CHECKERS, corners, ret)

        img = resizeimgh(img, H_IMGS)
        cv2.imshow('curr img', img)

    # Draw all the corners found so far
    imgAll = np.zeros(img.shape, dtype = np.uint8)
    for f in imgPoints:
        for p in f:
            cv2.circle(imgAll, tuple(p[0]), 2, (255, 255, 255), -1)
    imgAll = resizeimgh(imgAll, H_IMGS)
    cv2.imshow('all points', imgAll)
    cv2.waitKey(1)

print '\n%d checkerboards detected' % n_valid
print '%d checkerboards not detected' % (len(images) - n_valid)

# Save image points for debug
cv2.imwrite(imgfilebase % 'allpoints.png', imgAll)

# Calibration routine
print '\nCompute calibration'
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

# Print and calibration parameters
calibstr = 'K:\n%s\nD:\n%s' % (tostr(K, 2), tostr(D, 5))
print calibstr
print 'Save calibration'
with open(imgfilebase % 'calib.txt', 'w') as calibfile:
    calibfile.write(calibstr)

# Compute reprojection error
print '\nCompute reprojection error'
mean_error = 0
for i in xrange(len(objPoints)):
    # reproject points
    reprojPoints, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], K, D)
    # compute error
    error = cv2.norm(imgPoints[i], reprojPoints, cv2.NORM_L2) / len(reprojPoints)
    mean_error += error

print 'Reprojection error: %.3f [pixel]' % (mean_error / len(objPoints))
cv2.waitKey(0)
