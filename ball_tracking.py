from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# Parse arguemens
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Upper and lower bounderies for the green colour.
upper_boundary = (29, 86, 6)
lower_boundary = (64, 255, 255)

pts = deque(maxlen=args["buffer"])

# If cam path not provided, find the webcam.
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

# Allow some time for the cam
time.sleep(2.0)

while True:
    # Get current frame
    frame = vs.read() #ret frame tuple. idx 1 = bool, and idx 2 = frame capture

    frame = frame[1] if args.get("video", False) else frame


    if frame is None:
        break




