#this project aim to showcase some basic yet useful functions and methods used to detect movement
#as well as using some of the captured data to put a target on the moving object
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

argp = argparse.ArgumentParser()
argp.add_argument("-v", "--video", help="Path to the video file")
argp.add_argument("-a", "--min-area", type=int, default=500, help="Minimum area size")
args = vars(argp.parse_args())


if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])

firstFrame = None

while True:
    # grab a frame
    frame0 = vs.read()
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "No Motion "

    if frame is None:
        break
    frame0 = imutils.resize(frame0, width=500)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue
#background_subtraction
    frameDelta = cv2.absdiff(firstFrame, gray)
# applying a morphological operation (binary thresholding)  : creating binary frames out of "gray" in order to make the object more prominent
#dilating the frame (adding pixels to the boundaries of the detected objects)
#extracting the outer contours
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    x = np.ones((2,2),np.uint8)
    thresh = cv2.erode(thresh,x,iterations=4)
    thresh= cv2.dilate(thresh, x, iterations=8)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue

        (x, y, w, z) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+z), (0, 255, 0), 2)
    text = "Motion Detected"
    
    cv2.putText(frame, "Motion status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), (10, frame.shape[0]-10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # make coutour frame
    contr = frame0.copy()
    # target contours
    targets = []
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # contour data
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(c)
        rx = x+int(w/2)
        ry = y+int(h/2)
        ca = cv2.contourArea(c)
        # plot contours
        cv2.drawContours(contr,[c],0,(0,0,255),2)
        cv2.rectangle(contr,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(contr,(cx,cy),2,(0,0,255),2)
        cv2.circle(contr,(rx,ry),2,(0,255,0),2)
        cv2.putText(contr, "contours:".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(contr, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), (10, contr.shape[0]-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # save target contours
        targets.append((cx,cy,ca))
    # make target
    mx = 0
    my = 0
    if targets:
    #average centroid adjusted for contour size
         area = 0
         for x,y,a in targets:
            mx += x*a
            my += y*a
            area += a
         mx = int(round(mx/area,0))
         my = int(round(my/area,0))
        # centroid of largest contour
         area = 0
         for x,y,a in targets:
            if a > area:
                mx = x
                my = y
                area = a
    # plot target
    tr = 50
    targ = frame0.copy()
    if targets:
        cv2.circle(targ,(mx,my),tr,(0,0,255,0),2)
        cv2.line(targ,(mx-tr,my),(mx+tr,my),(0,0,255,0),2)
        cv2.line(targ,(mx,my-tr),(mx,my+tr),(0,0,255,0),2)
        cv2.putText(targ, "Target:".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(targ, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), (10, targ.shape[0]-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    cv2.imshow("normal video feed", frame0)
    cv2.imshow("Video feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow('Frame delta', frameDelta)
    cv2.imshow("Contours",contr)
    cv2.imshow("Target",targ)

    key = cv2.waitKey(1) & 0xFF
#press 'q' to quit
    if key == ord('q'):
        break
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
