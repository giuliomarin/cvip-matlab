import numpy as np
import cv2
import glob
import sys
import xml.etree.ElementTree as et
from random import uniform, randrange

sys.path.append('/GitHub/cvip/python')
from cvip import *

#########################
# Parameters
#########################

imgfilebase = '/Users/giulio/Library/Application Support/Aquifi/3/%s'
imgfile_l = imgfilebase % 'img2_*.png'
imgfile_r = imgfilebase % 'img0_*.png'
N_CHECKERS = (10, 8)  # (points_per_row,points_per_colum)
SIZE_CHECKERS = 20.0  # mm

# Visualization
H_IMGS = 400  # -1 for original size


#########################
# Data structures
#########################

class CameraParams():
    def __init__(self, K, D, R, T, resolution, camera_id):
        self.K = K
        self.D = D
        self.R = R
        self.T = T
        self.resolution = resolution
        self.camera_id = camera_id


class CalibData():
    def __init__(self, calibDataFilename = None):

        self.numCameras = 0
        self.cameraParsList = []

        if not (calibDataFilename == None):
            tree = et.parse(calibDataFilename)
            root = tree.getroot()

            # Loop on the root
            for child in root:
                if child.tag == "camera_calibrations":
                    for cameras in child:

                        # Search the calibration parameters for each camera
                        K = []
                        D = []
                        R = []
                        T = []
                        resolution = []
                        camera_id = 0
                        for matrix in cameras:
                            if matrix.tag == "K":
                                K = np.array([float(x) for x in matrix[3].text.split()])
                                K = np.reshape(K, (3, 3))
                            if matrix.tag == "D":
                                D = np.array([float(x) for x in matrix[3].text.split()])
                            if matrix.tag == "R":
                                R = np.array([float(x) for x in matrix[3].text.split()])
                                R = np.reshape(R, (3, 3))
                            if matrix.tag == "T":
                                T = np.array([float(x) for x in matrix[3].text.split()])
                            if matrix.tag == "resolution":
                                resolution = (int(matrix.text.split()[0]), int(matrix.text.split()[1]))
                            if matrix.tag == "camera_id":
                                camera_id = int(matrix.text.split()[0])
                        currCam = CameraParams(K, D, R, T, resolution, camera_id)
                        self.numCameras += 1
                        self.cameraParsList.append(currCam)

    def initFromCameraList(self, cameraList):
        self.numCameras = len(cameraList)
        for cam in cameraList:
            self.cameraParsList.append(cam)


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


def getoptimalimg(img):
    img = np.sqrt(img)
    img /= max(img.flatten())
    img *= 255.0
    return img.astype(np.uint8)


def getimage(imgpath):
    print 'Image: %s' % imgpath,
    # load image and convert to grayscale
    img, isfloat = dataio.imread(imgpath)
    if isfloat:
        gray = getoptimalimg(img)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
    return img, gray


def processimg(img, gray, idx):
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, N_CHECKERS, None,
                                             flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

    imgtoshow = img.copy()
    isvalid = False
    # If not found skip impage
    if ret is not True:
        print ' [not found]'
        corners = None
    else:
        print ' [found]'
        isvalid = True
        # refinine image points using subpixel accuracy
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(imgtoshow, N_CHECKERS, corners, True)

    # Draw all the corners found so far
    pointSize = int(round(img.shape[1] / 350.0))
    for f in imgPoints[idx]:
        for p in f:
            cv2.circle(imgtoshow, tuple(p[0]), pointSize, (145, 255, 145), -1)
    return imgtoshow, isvalid, corners


def tostr(arr, prec = 3):
    strout = ''
    for l in arr:
        for c in l:
            strout += ('%.' + str(prec) + 'f ') % c
        strout += '\n'
    return strout

def loadcalib(calibpath):
    try:
        with open(calibpath, 'r') as calibfile:
            # K left
            K_l = []
            calibstr = calibfile.readline()
            K_l = np.append(K_l, [float(x) for x in calibfile.readline().split()])
            K_l = np.append(K_l, [float(x) for x in calibfile.readline().split()])
            K_l = np.append(K_l, [float(x) for x in calibfile.readline().split()])
            K_l = K_l.reshape(3, 3)

            # D left
            D_l = []
            calibstr = calibfile.readline()
            calibstr = calibfile.readline()
            D_l = np.append(D_l, [float(x) for x in calibfile.readline().split()])

            # K right
            K_r = []
            calibstr = calibfile.readline()
            calibstr = calibfile.readline()
            K_r = np.append(K_r, [float(x) for x in calibfile.readline().split()])
            K_r = np.append(K_r, [float(x) for x in calibfile.readline().split()])
            K_r = np.append(K_r, [float(x) for x in calibfile.readline().split()])
            K_r = K_r.reshape(3, 3)

            # D
            D_r = []
            calibstr = calibfile.readline()
            calibstr = calibfile.readline()
            D_r = np.append(D_r, [float(x) for x in calibfile.readline().split()])

            # R
            R = []
            calibstr = calibfile.readline()
            calibstr = calibfile.readline()
            R = np.append(R, [float(x) for x in calibfile.readline().split()])
            R = np.append(R, [float(x) for x in calibfile.readline().split()])
            R = np.append(R, [float(x) for x in calibfile.readline().split()])
            R = R.reshape(3, 3)

            # T
            T = []
            calibstr = calibfile.readline()
            calibstr = calibfile.readline()
            T = np.append(T, [float(calibfile.readline())])
            T = np.append(T, [float(calibfile.readline())])
            T = np.append(T, [float(calibfile.readline())])
            T = T.reshape(3, 1)
    except:
        sys.exit('Unable to parse calibration file')

    return K_l, D_l, K_r, D_r, R, T

##########################
# Calibration
##########################

newsize = (640, 480)
# Load images
images = [glob.glob(imgfile_l), glob.glob(imgfile_r)]

if False:

    if len(images) == 0:
        sys.exit('No image matches: %s' % imgfilebase)

    # Prepare object points
    objp = np.zeros((np.prod(N_CHECKERS), 3), np.float32)
    objp[:, :2] = np.mgrid[0:N_CHECKERS[0], 0:N_CHECKERS[1]].T.reshape(-1, 2)
    objp *= SIZE_CHECKERS

    # Arrays to store object points and image points from all the images.
    isvalid = ([], [])
    imgPoints = ([], [])  # 2d points in image plane.

    # Info
    n_valid = 0

    # Convergence criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)

    # Loop on images
    for i in xrange(len(images[0])):
        # Left camera
        img_l, gray_l = getimage(images[0][i])
        imgtoshow_l, isvalid_l, corners_l = processimg(img_l, gray_l, 0)
        if isvalid_l:
            isvalid[0].append(True)
            imgPoints[0].append(corners_l)
        else:
            isvalid[0].append(False)
            imgPoints[0].append([])

        # Right camera
        img_r, gray_r = getimage(images[1][i])
        imgtoshow_r, isvalid_r, corners_r = processimg(img_r, gray_r, 1)
        if isvalid_r:
            isvalid[1].append(True)
            imgPoints[1].append(corners_r)
        else:
            isvalid[1].append(False)
            imgPoints[1].append([])

        imgtoshow_l = resizeimgh(imgtoshow_l, H_IMGS)
        imgtoshow_r = resizeimgh(imgtoshow_r, H_IMGS)
        imgtoshow = np.hstack((imgtoshow_l, imgtoshow_r))
        if False:
            cv2.imshow('curr img', imgtoshow)
            cv2.waitKey(1)

    # Calibration routine
    print '\nCompute calibration'

    objpoints = [objp for v in isvalid[0] if v]
    currimgpoints = [imgPoints[0][i] for i, v in enumerate(isvalid[0]) if v]
    ret, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, currimgpoints, img_l.shape[0:2], None, None)

    objpoints = [objp for v in isvalid[1] if v]
    currimgpoints = [imgPoints[1][i] for i, v in enumerate(isvalid[1]) if v]
    ret, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, currimgpoints, img_r.shape[0:2], None, None)

    bothvalid = [l_valid and r_valid for l_valid, r_valid in zip(isvalid[0], isvalid[1])]
    objpoints = [objp for v in bothvalid if v]
    currimgpoints_l = [imgPoints[0][i] for i, v in enumerate(bothvalid) if v]
    currimgpoints_r = [imgPoints[1][i] for i, v in enumerate(bothvalid) if v]

    ret, Knew_l, Dnew_l, Knew_r, Dnew_r, R, T, _, _ = cv2.stereoCalibrate(objpoints,
                                                                          currimgpoints_l,
                                                                          currimgpoints_r,
                                                                          K_l.copy(),
                                                                          D_l.copy(),
                                                                          K_r.copy(),
                                                                          D_r.copy(),
                                                                          newsize,
                                                                          flags = cv2.CALIB_FIX_INTRINSIC)

    # Print and calibration parameters
    calibstr = 'K_l:\n%s\nD_l:\n%s\nK_r:\n%s\nD_l:\n%s\nR:\n%s\nT:\n%s' % (tostr(K_l, 2), tostr(D_l, 5), tostr(K_r, 2), tostr(D_r, 5), tostr(R, 3), tostr(T, 3))
    print calibstr
    print 'Save calibration'
    with open(imgfilebase % 'calib.txt', 'w') as calibfile:
        calibfile.write(calibstr)


K_l, D_l, K_r, D_r, R, T = loadcalib(imgfilebase % 'calib.txt')

# Rectification
R_l, R_r, P_l, P_r, _, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, newsize, R, T, flags = cv2.CALIB_ZERO_DISPARITY)

rectMap = [[[], []], [[], []]]
rectMap[0][0], rectMap[0][1] = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, newsize, cv2.CV_16SC2)
rectMap[1][0], rectMap[1][1] = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, newsize, cv2.CV_16SC2)

idx = randrange(len(images[0]))
img_l, gray_l = getimage(images[0][idx])
img_r, gray_r = getimage(images[1][idx])

img_l, gray_l = getimage('/Users/giulio/Library/Application Support/Aquifi/2/img1_1.png')
img_r, gray_r = getimage('/Users/giulio/Library/Application Support/Aquifi/2/img0_1.png')

imgRect_l = cv2.remap(img_l, rectMap[0][0], rectMap[0][1], cv2.INTER_CUBIC)
imgRect_r = cv2.remap(img_r, rectMap[1][0], rectMap[1][1], cv2.INTER_CUBIC)

imgtoshow_l = resizeimgh(imgRect_l, H_IMGS)
imgtoshow_r = resizeimgh(imgRect_r, H_IMGS)
imgtoshow = np.hstack((imgtoshow_l, imgtoshow_r))

lines = [((0, int(r)), (imgtoshow.shape[1], int(r)), (uniform(0,255), uniform(0,255), uniform(0,255))) for r in np.linspace(0, imgtoshow.shape[0], 20)]
for pt in lines:
    cv2.line(imgtoshow, pt[0], pt[1], pt[2], 1)

cv2.imwrite('/Users/giulio/Library/Application Support/Aquifi/2/imgrect1_1.png', imgRect_l)
cv2.imwrite('/Users/giulio/Library/Application Support/Aquifi/2/imgrect0_1.png', imgRect_r)

cv2.imshow('curr img', imgtoshow)
cv2.waitKey(0)