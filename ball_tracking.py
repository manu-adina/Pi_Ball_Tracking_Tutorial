from collections import deque
from imutils.video import VideoStream # I don't like this.
import imutils
import numpy as np
import argparse
import cv2
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
time.sleep(1.0)

while True:
    # Get current frame
    frame = vs.read() #ret frame tuple. idx 1 = bool, and idx 2 = frame capture

    frame = frame[1] if args.get("video", False) else frame

    # Stop if frame is empty. Could mean that the cam was stopped.
    if frame is None:
        break

    # Resizing the frame to make it less resource internsive.
    frame = imutils.resize(frame, width=600) # Want to make it use normal resizing methods.
    # Blur to reduce noise.
    blurred = cv2.GaussianBlur(frame, (11,11), 0)

    # Converting to HSV space. Hue is useful to differentiate colours.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Returns a binary mask of the image. Puts '1' for indexes that are within range.
    mask = cv2.inRange(hsv, lower_boundary, upper_boundary)

    # Removes any small particles that are left by inRange.
    mask = cv2.erode(mask, None, iterations = 2)
    mark = cv2.dilate(mask, None, iterations = 2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # I don't want to use imutils.
    center = None


    if len(cnts) > 0:
        # Find the largest contour.
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) # Thats how you find the center

        # Only proceed if the radius is over the minimum size.
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0,0,255), thickness)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()


