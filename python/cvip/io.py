import cv2
import numpy as np


def imread32f(imgPath):
    """
    Load a float mat stored in a png file
        \param imgPath : path to the .png image
        \return : matrix values
    """
    ## Read the png image
    img = cv2.imread(imgPath, -1)
    if img is None:
        raise IOError('File not found: %s' % imgPath)
    # Convert it to float
    imSize = (img.shape[0],img.shape[1])
    imFloat = np.zeros(imSize, np.float32)
    imFloat.data = img.data

    return imFloat


def imwrite32f(imgPath, img):
    """
    Write a float matrix in a png file
        \param imgPath : path to the .png image
        \img : image to store
    """
    # Save image
    imgToWrite = np.zeros((480, 640, 4), np.uint8)
    imgToWrite.data = img.data
    cv2.imwrite(imgPath, imgToWrite)


if __name__ == '__main__':
    print 'done'