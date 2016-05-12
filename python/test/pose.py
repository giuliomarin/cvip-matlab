import numpy as np
import cv2
import glob
import sys

#########################
# Parameters
#########################

imgfilebase = '/Users/giulio/Library/Application Support/Aquifi/1/%s'
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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

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
# Pose estimation
##########################

# Load images
images = glob.glob(imgfile)

if len(images) == 0:
    sys.exit('No image matches: %s' % imgfile)

# Convergence criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)

# Prepare object points
objp = np.zeros((np.prod(N_CHECKERS), 3), np.float32)
objp[:, :2] = np.mgrid[0:N_CHECKERS[0], 0:N_CHECKERS[1]].T.reshape(-1, 2)
objp *= 20.0;

axis = 20.0 * np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)

# Loop on images
for fname in images:
    print 'Image: %s' % fname,
    # load image and convert to grayscale
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
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

        # refinine image points using subpixel accuracy
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, K, D)
        
        print 'Distance: %d [mm]' % int(round(np.linalg.norm(tvecs)))

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, D)

        imgToShow = draw(img, corners, imgpts)
        imgToShow = resizeimgh(imgToShow, H_IMGS)
        cv2.imshow('curr img', imgToShow)
        cv2.waitKey(0)