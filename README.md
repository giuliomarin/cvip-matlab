==============================================================================
Computer Vision and Image Processing functions and scripts 
==============================================================================



Collect row, column and value of the selected points
--------------------------------------------------------------------------------

Collect row, column and value of the selected points in an image given the handle to the figure.

File: [collectcursorinfo.m](https://github.com/giuliomarin/cvip/blob/master/collectcursorinfo.m)

--------------------------------------------------------------------------------

Create a video from a list of images
--------------------------------------------------------------------------------

Create a video from a list of black and white or colored images, with the option to specify the frame rate and whether to compress the video or not.

File: [createvideofromimages.m](https://github.com/giuliomarin/cvip/blob/master/createvideofromimages.m)

--------------------------------------------------------------------------------

Convert disparity map to depth map and vice versa
--------------------------------------------------------------------------------

Compute a depth map given a disparity map and the product baseline * focal length. It can be used also to compute the opposite, given a depth map, compute the disparity map.

File: [disparity2depth.m](https://github.com/giuliomarin/cvip/blob/master/disparity2depth.m)

--------------------------------------------------------------------------------

Fit a plane
--------------------------------------------------------------------------------

Fit a plane on a 3D point cloud (x,y,z).

File: [fitplate.m](https://github.com/giuliomarin/cvip/blob/master/fitplane.m)

--------------------------------------------------------------------------------

Load data in the xml file stored with OpenCV
--------------------------------------------------------------------------------
Create a structure with all the entries in the xml file. Deal with Mat, string and single numbers.
Each entry in the structure has the name of the variable in the XML file.

File: [loadopencvxmlfile.m](https://github.com/giuliomarin/cvip/tree/master/loadopencvxmlfile.m)

--------------------------------------------------------------------------------

Create ply point cloud
--------------------------------------------------------------------------------

Converts 3D point cloud (x,y,z) to '.ply' format. The point cloud can have additional columns for the color information.

File: [mat2ply.m](https://github.com/giuliomarin/cvip/blob/master/mat2ply.m)

--------------------------------------------------------------------------------

Otsu's segmentation method with 3 gray levels
--------------------------------------------------------------------------------
Example that shows the Otsu's algorithm for segmenting an image whose color distribution presents 3 peaks. The algorithm find the thresholds such that the within-class variance is minimized. This is an extension of the original algorithm for two peaks distribution.

Directory: [otsu3](https://github.com/giuliomarin/cvip/tree/master/otsu3)

--------------------------------------------------------------------------------

Load ply point cloud
--------------------------------------------------------------------------------

Load a '.ply' file and store 3D point cloud in a N x 3 matrix (x,y,z). If the original point cloud has also color information, this will be saved as well.

File: [ply2mat.m](https://github.com/giuliomarin/cvip/blob/master/ply2mat.m)

--------------------------------------------------------------------------------

Compute Block Matching
--------------------------------------------------------------------------------

Class to produce a disparity map by computing stereo block matching given a couple of rectified images.

File: [stereobm.m](https://github.com/giuliomarin/cvip/blob/master/stereobm.m)
Sample: [test_stereobm.m](https://github.com/giuliomarin/cvip/blob/master/samples/stereobm.m)

--------------------------------------------------------------------------------

Back project 2D points to 3D
--------------------------------------------------------------------------------

Project a 2D depth map in the 3D space.

File: [to3d.m](https://github.com/giuliomarin/cvip/blob/master/to3d.m)

--------------------------------------------------------------------------------