# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#       --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#       --output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#       --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#       --output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
from scipy.spatial import distance as disFuc
import pyrealsense2 as rs
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import bcc
import group
import time
import itertools


def merge_recs(rects):
    result = (640, 480, 0, 0)
    for rect in rects:
        result = union(result, rect)
    return [result]


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[2], b[2])
    h = max(a[3], b[3])
    return (x, y, w, h)

def isGrouped(groupList, objID):
    for g in groupList:
        if objID in g.idGroup and len(g.idGroup) > 1:
             return True
    
    return False


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
ap.add_argument("-pd", "--pixel-distance", type=int, default=150,
                help="pixel threshold for contact tracking")
ap.add_argument("-md", "--meter-distance", type=float, default=1,
                help="meter threshold for contact tracking")
ap.add_argument("-se", "--contact-time", type=int, default=3,
                help="minimum seconds for close contact")
args = vars(ap.parse_args())

# load our serialized model
print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# grab a reference to the webcam
print("[INFO] starting video stream...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
time.sleep(2.0)

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# initialize group list for tracking
groupList = []

# loop over frames from the video stream
while True:
        # grab the next frame and handle
    frame = pipeline.wait_for_frames()
    depth = frame.get_depth_frame()
    frame = np.asanyarray(frame.get_color_frame().get_data())

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    # frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    # depth = imutils.resize(depth, width=min(400, frame.shape[1]))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        # Detect people in the frame
        (rectangles, weights) = hog.detectMultiScale(
            frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        rectangles = np.array([[x, y, x + w, y + h]
                               for (x, y, w, h) in rectangles])
        picks = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)

        # loop over the detections
        for pick, weight in zip(picks, weights):
            # filter out weak detections by requiring a minimum
            # confidence
            if weight > args["confidence"]:
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                (startX, startY, endX, endY) = pick
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    # set the status of our system to be 'tracking' rather
    # than 'waiting' or 'detecting'
    status = "Tracking"
    # loop over the trackers
    for tracker in trackers:
        # update the tracker and grab the updated position
        tracker.update(rgb)
        pos = tracker.get_position()
        # unpack the position object
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        # add the bounding box coordinates to the rectangles list
        rects.append((startX, startY, endX, endY))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # record close contact of people in the frame
    if totalFrames % args["skip_frames"] == 0:
        objectList = list(
            map(lambda x: (x[0], x[1][0]), list(objects.items())))
        g = bcc.Graph(len(objectList))

        def discoverEdge(obList, g):
            if len(obList) == 0:
                return
            else:
                originVertex = obList[0]
                destinationVertices = obList[1:]
                for destinationVertex in destinationVertices:
                    (oriObjectID, oriCentroid) = originVertex
                    (desObjectID, desCentroid) = destinationVertex

                    distanceBtwObjs = disFuc.euclidean(
                        oriCentroid, desCentroid)
                    print(oriCentroid)
                    print(desCentroid)
                    if 0 < oriCentroid[0] < 640 and 0 < oriCentroid[1] < 480 \
                        and 0 < desCentroid[0] < 640 and 0 < desCentroid[1] < 480:
                        oriDistanceFromCam = depth.get_distance(
                            oriCentroid[0], oriCentroid[1])
                        desDistanceFromCam = depth.get_distance(
                            desCentroid[0], desCentroid[1])
                    else: # fail to get a depth from the pixel
                        oriDistanceFromCam = 0
                        desDistanceFromCam = 9999999999999

                    print(distanceBtwObjs)
                    if distanceBtwObjs < args["pixel_distance"] and abs(oriDistanceFromCam - desDistanceFromCam) < args["meter_distance"]:
                        g.addEdge(oriObjectID, desObjectID)
                discoverEdge(destinationVertices, g)
        discoverEdge(objectList, g)

        # get update group list for tracking their remaining time
        newGroupList = g.BCC()
        updatedGroupList = group.updateGroupList(groupList, newGroupList)

        # capture long-lasting group and renew group list
        groupList = []
        for g in updatedGroupList:
            if abs(time.time() - g.timestamp) > args["contact_time"]:
                capturedRects = []
                for idx in g.idGroup:
                    (centroid, rect) = objects[idx]
                    capturedRects.append(rect)

                (startX, startY, endX, endY) = merge_recs(capturedRects)[0]
                cropImg = frame[startY:endY, startX:endX]
                timestr = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite("capture/" + timestr + ".png", cropImg)
                print(g)
                print(
                    "-----------------------------Capured!-----------------------------")

                g.captured = True
                groupList.append(g)
            else:
                groupList.append(g)
                
    # draw line between members of a detected group
    for g in updatedGroupList:
        if len(g.idGroup) != 1:
            for (oriID, desID) in itertools.combinations(g.idGroup, 2):
                try:
                    oriCentroid = tuple(objects[oriID][0])
                    desCentroid = tuple(objects[desID][0])
                    cv2.line(frame, oriCentroid, desCentroid, (255, 0, 0), 2)
                except:
                    pass

    # loop over the tracked objects
    for (objectID, (centroid, rect)) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object, the distance of the object,
        # and the centroid of the object on the output frame
        if 0 < centroid[0] < 640 and 0 < centroid[1] < 480:
            distance = depth.get_distance(centroid[0], centroid[1])
        else:
            distance = 0
        text = "Person {} - {}m".format(objectID, round(distance, 2))
        (startX, startY, endX, endY) = rect
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # draw the green bounding box if the object is not included in any group,
        # otherwise red bounding box
        if isGrouped(updatedGroupList, objectID):
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
        else:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # cv2.imwrite("capture/full.png", frame)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    frame = imutils.resize(frame, width=1024)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# stop the camera video stream
pipeline.stop()

# close any open windows
cv2.destroyAllWindows()
