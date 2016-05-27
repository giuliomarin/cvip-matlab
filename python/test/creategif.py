from moviepy.editor import *

inPath = '../../samples/images/%s'
imglist = ['left.png', 'right.png']

images = []
for i in imglist:
    images.append(inPath % i)
    
clip = ImageSequenceClip(images, fps=2).crop(x1=400, x2=1100, y1=500, y2 = 1000).resize(0.5)
clip.write_gif(inPath % 'sample.gif')