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
import numpy as np

import imutils
import time
import dlib
import cv2
import bcc
import group
import time
import itertools

class Recorder:
    def __init__(self, argConf, argSkip, argPd, argMd, argTime):
        self.argConf = argConf
        self.argSkip = argSkip
        self.argPd = argPd
        self.argMd = argMd
        self.argTime = argTime
        
        # load our serialized model
        print("[INFO] loading model...")
        # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        # Initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0

        # start the frames per second throughput estimator
        self.fps = FPS().start()

        # initialize group list for tracking
        self.groupList = []
        self.updatedGroupList = []
    
    # loop over frames from the video stream
    def forward(self, frame, depth):
        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        # frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        # depth = imutils.resize(depth, width=min(400, frame.shape[1]))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if self.totalFrames % self.argSkip == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            self.trackers = []

            # Detect people in the frame
            (rectangles, weights) = self.hog.detectMultiScale(
                frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

            rectangles = np.array([[x, y, x + w, y + h]
                                   for (x, y, w, h) in rectangles])
            picks = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)

            # loop over the detections
            for pick, weight in zip(picks, weights):
                # filter out weak detections by requiring a minimum
                # confidence
                if weight > self.argConf:
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    (startX, startY, endX, endY) = pick
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        # set the status of our system to be 'tracking' rather
        # than 'waiting' or 'detecting'
        status = "Tracking"
        # loop over the trackers
        for tracker in self.trackers:
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
        objects = self.ct.update(rects)

        # record close contact of people in the frame
        if self.totalFrames % self.argSkip == 0:
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
                        if 0 < oriCentroid[0] < 640 and 0 < oriCentroid[1] < 480 \
                            and 0 < desCentroid[0] < 640 and 0 < desCentroid[1] < 480:
                            oriDistanceFromCam = depth.get_distance(
                                oriCentroid[0], oriCentroid[1])
                            desDistanceFromCam = depth.get_distance(
                                desCentroid[0], desCentroid[1])
                        else: # fail to get a depth from the pixel
                            oriDistanceFromCam = 0
                            desDistanceFromCam = 9999999999999

                        if distanceBtwObjs < self.argPd and abs(oriDistanceFromCam - desDistanceFromCam) < self.argMd:
                            g.addEdge(oriObjectID, desObjectID)
                    discoverEdge(destinationVertices, g)
            discoverEdge(objectList, g)

            # get update group list for tracking their remaining time
            newGroupList = g.BCC()
            self.updatedGroupList = group.updateGroupList(self.groupList, newGroupList)

            # capture long-lasting group and renew group list
            groupList = []
            for g in self.updatedGroupList:
                if abs(time.time() - g.timestamp) > self.argTime:
                    capturedRects = []
                    for idx in g.idGroup:
                        (centroid, rect) = objects[idx]
                        capturedRects.append(rect)

                    (startX, startY, endX, endY) = merge_recs(capturedRects)[0]
                    cropImg = frame[startY:endY, startX:endX]
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite("capture/" + timestr + ".png", cropImg)
                    print(
                        "-----------------------------Capured!-----------------------------")

                    g.captured = True
                    self.groupList.append(g)
                else:
                    self.groupList.append(g)
                
        # draw line between members of a detected group
        for g in self.updatedGroupList:
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
            to = self.trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to

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
            if isGrouped(self.updatedGroupList, objectID):
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
            cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    

    
        # increment the total number of frames processed thus far and
        # then update the FPS counter
        self.totalFrames += 1
        
        return frame

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
