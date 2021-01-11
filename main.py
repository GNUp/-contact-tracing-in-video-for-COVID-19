# main module
# bring frames from the depth camera and insert them to the tracer
# after getting the result framem, display it to the screen

import pyrealsense2 as rs
import numpy as np
import argparse
import cv2
import time
import imutils

import recorder

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
                help="path to Caffe pre-trained model")
ap.add_argument("-s", "--skip-frames", type=int, default=20,
                help="# of skip frames between detections")
ap.add_argument("-pd", "--pixel-distance", type=int, default=100,
                help="pixel threshold for contact tracking")
ap.add_argument("-md", "--meter-distance", type=float, default=1,
                help="meter threshold for contact tracking")
ap.add_argument("-se", "--contact-time", type=int, default=3,
                help="minimum seconds for close contact")
ap.add_argument("-c", "--conf", default="config/config.json",
                help="Path to the input configuration file")
args = vars(ap.parse_args())

# grab a reference to the webcam
print("[INFO] starting video stream...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
time.sleep(2.0)

recorderHandler = recorder.Recorder(args["skip_frames"], args["pixel_distance"],
                                    args["meter_distance"], args["contact_time"], args["prototxt"], args["model"], args["conf"])

while(True):
    # grab the next frame and handle
    frame = pipeline.wait_for_frames()
    depth = frame.get_depth_frame()
    frame = np.asanyarray(frame.get_color_frame().get_data())
    
    frame = recorderHandler.forward(frame, depth)
    
    # show the output frame
    frame = imutils.resize(frame, width=1024)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    

# close any open windows
cv2.destroyAllWindows()

# stop the camera video stream
pipeline.stop()
    

