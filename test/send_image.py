"""
    send_image.py

"""

import cv2
import sys
sys.path.append("./")
from fastapi_frame_stream import FrameStreamer
from glob import glob
import time
import asyncio

fs2 = FrameStreamer()

async def send_image():
    # iname = 'C:/PycharmProjects/fastapi-frame-stream/test/images/chessboard_01.png'
    # img = cv2.imread(iname, cv2.IMREAD_GRAYSCALE)
    img_mask = 'C:\\PycharmProjects\\fastapi-frame-stream\\test\\images\\chessboard_??.png'  # default
    img_names = glob(img_mask)
    for x in range(0,200):
        print(' X: ', x)
        for fn in img_names:
            await fs2.send_frame('img_stream', fn)
            time.sleep(2)
    print('END')

print('START......')
time.sleep(10)
asyncio.run(send_image())



iname = 'C:/PycharmProjects/fastapi-frame-stream/test/images/chessboard_01.png'
img = cv2.imread(iname, cv2.IMREAD_GRAYSCALE)

# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)
height, width = img.shape
# cv.resizeWindow("image", int(width*0.5), int(height*0.5))

# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)

# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()

