# Lint as: python
#
# Authors: Vittorio Mazzia
# Location: Turin
#!/usr/bin/env python
"""Launch a simple server to show, on a web page, live predictions made by the network
 loaded in the detector objtect"""
import os
from flask import Flask, render_template, Response
from utils.detect_camera import Detector
import argparse

parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--tpu', default=0, type=int, 
                    help='CPU: 0 | TPU: 1', required=True)
parser.add_argument('--threshold', default=0.5, type=float, 
                    help='SSD Threshold level', required=False)
parser.add_argument('--camera', default=0, type=int, 
                    help='Camera OpenCV index', required=False)

args = parser.parse_args()

# create an detector object (put wathever network you like)
# in this example I used a SSD trained on COCO for object detection
detector = Detector(args.tpu, args.threshold, args.camera)
detector.start()
app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page"""
    return render_template('index.html')


def gen(detector):
    """Video streaming generator function"""
    while True:
        frame = detector.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag"""
    return Response(gen(detector),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
