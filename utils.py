import numpy as np
import cv2

def imread32f(imgPath):
    """
    Load a float mat stored in a png file
        \param imgPath : path to the .png image
        \return : matrix values
    """
    ## Read the png image
    img2 = cv2.imread(imgPath, -1)

    # Convert it to float
    imSize = (img2.shape[0],img2.shape[1])
    imFloat = np.zeros(imSize, np.float32)
    imFloat.data = img2.data

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
