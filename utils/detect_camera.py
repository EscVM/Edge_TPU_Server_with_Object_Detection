# Lint as: python
#
# Authors: Vittorio Mazzia
# Location: Turin
"""Detector class to detect objects in a video stream with a dedicated thread"""
import argparse
import time
from PIL import Image
from PIL import ImageDraw
import utils.detect as detect
import tflite_runtime.interpreter as tflite
import platform
import cv2
from threading import Thread


class Detector(Thread):
  """
  Detector uses TensorFlow Lite and OpenCV to detect objects in a video stream
  provided by a camera attached to the host computer
  ...

  Attributes
  ----------
  tpu: bool
    make use of TPU connected to the host device
  threshold: float
    threshold for the SSD network
  camera: int
    index of the OpenCV camera


  Methods
  -------
  oad_labels(path, encoding='utf-8')
    Load labels for the network
  make_interpreter(model_file)
    Create a TensorFlow Lite Interpreter
  draw_objects(frame, objs, labels, scale, fps)
    draw detections on a given frame
  run()
    run the thread
  get_frame()
    get a processed frame
  """
  def __init__(self, tpu=False, threshold=0.5, camera=0):
    Thread.__init__(self)
    if tpu:
      self.MODEL_PATH = 'models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    else:
      self.MODEL_PATH = 'models/mobilenet_ssd_v2_coco_quant_postprocess.tflite'
    self.LABEL_PATH = 'models/coco_labels.txt'
    self.threshold = threshold
    self.frame_bytes = None
    self.tpu=tpu
    self.camera = camera


  def load_labels(self, path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers)"""
    with open(path, 'r', encoding=encoding) as f:
      lines = f.readlines()
      if not lines:
        return {}

      if lines[0].split(' ', maxsplit=1)[0].isdigit():
        pairs = [line.split(' ', maxsplit=1) for line in lines]
        return {int(index): label.strip() for index, label in pairs}
      else:
        return {index: line.strip() for index, line in enumerate(lines)}
   

  def make_interpreter(self, model_file):
    """Make a TensorFlow Lite interpreter"""
    EDGETPU_SHARED_LIB = {
      'Linux': 'libedgetpu.so.1',
      'Darwin': 'libedgetpu.1.dylib',
      'Windows': 'edgetpu.dll'
    }[platform.system()]

    model_file, *device = model_file.split('@')

    if self.tpu == False:
      return tflite.Interpreter(
        model_path=model_file)

    else:
      return tflite.Interpreter(
          model_path=model_file,
          experimental_delegates=[
              tflite.load_delegate(EDGETPU_SHARED_LIB,
                                   {'device': device[0]} if device else {})])

  def draw_objects(self, frame, objs, labels, scale, fps):
    """Draws the bounding box and label for each object"""
    for obj in objs:

      bbox = obj.bbox
      cv2.rectangle(frame, 
                  (int(bbox.xmin), int(bbox.ymin)), 
                  (int(bbox.xmax), int(bbox.ymax)),
                 (0,0,255), 
                 3)
      cv2.putText(frame,
            '{} {:.2}'.format(labels.get(obj.id, obj.id), obj.score),
            (int(bbox.xmin - 10), int(bbox.ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            1,
            cv2.LINE_AA)

    cv2.putText(frame, 'FPS:{:.4}'.format(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)


  def run(self):
    """Run a thread"""
    # initialize coral accelerator
    labels = self.load_labels(self.LABEL_PATH)
    interpreter = self.make_interpreter(self.MODEL_PATH)
    interpreter.allocate_tensors()

    # initialize camera
    camera = cv2.VideoCapture(self.camera)

    # start loop
    while(True):
      # get a frame from the camera
      ret, frame = camera.read()

      # ------- 
      t0 = time.time()
      frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
      scale = detect.set_input_video(interpreter, frame.shape[:2], frame_rgb)
      
      interpreter.invoke()

      objs = detect.get_output(interpreter, self.threshold, scale)
      t1 = time.time()
      # -------   
      frame_bgr = frame.copy()
      fps = (1/(t1-t0))
      self.draw_objects(frame_bgr, objs, labels, scale, fps)
      
      self.frame_bytes = cv2.imencode('.jpg', frame_bgr)[1].tobytes()

  def get_frame(self):
    """get a processed frame"""
    return self.frame_bytes

