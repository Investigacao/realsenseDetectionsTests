"""BW: this file is all to speed up the inference step, including measures:
    + with smaller image size, i.e. (1280x720)-(640,480)-(640-360).
    + with less displayed sentences, only centers and distances are kept shown.
    + with larger softmax threshold, i.e. 75%.


In Windows10 BW vip notes:
1. Regarding "cannot import name '_C' #157" error.
  Backgraound: **Build Detectron2 from Source**

    [Windows] Install Visual C++ Build tools form this link: https://answers.microsoft.com/en-us/windows/forum/windows_10-windows_install/microsoft-visual-c-140-is-required-in-windows-10/f0445e6b-d461-4e40-b44f-962622628de7.  Then restart your PC, then you also need to upgrade Python setup tools, by running this command: `pip install --upgrade setuptools`.

    After having the above dependencies you can install detectron2 from source by running:
    ~~~~~bash
    [Note-Works in Windows10!] pip install git+https://github.com/facebookresearch/detectron2.git
    # (add --user if you don't have permission)

    # Or, to install it from a local clone:
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2 && pip install -e .

    # Or if you are on macOS
    # CC=clang CXX=clang++ pip install -e .
    ~~~~~
  A: So I install by pip, and run demo or example in git repo detectron in root directory,
      import detectron may import lib from your git root directory (not pip installation).
      This won't work (you want to use pip installation).
    (VIP-BW) You may remove detectron directory or change the name(as I do here), so python will look in pip packages.
"""

from ast import If
from calendar import c
from math import *

import numpy as np
import time
import cv2
import pyrealsense2 as rs 
import random
import math
import argparse

from threading import Thread
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

from darwin.torch.utils import detectron2_register_dataset

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes, RotatedBoxes
from detectron2 import model_zoo

from detectron2.data import MetadataCatalog

import torch, torchvision

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pkg_resources
import sys

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import asyncio
import websocket
import json

# >>---------------------- load predefined model -------------------
class _ModelZooUrls(object):
    """
    Mapping from names to officially released Detectron2 pre-trained models.
    """

    S3_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    # format: {config_path.yaml} -> model_id/model_final_{commit}.pkl
    CONFIG_PATH_TO_URL_SUFFIX = {
        # COCO Detection with Faster R-CNN
        "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml": "137257644/model_final_721ade.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml": "137847829/model_final_51d356.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml": "137257794/model_final_b275ba.pkl",
        "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml": "137849393/model_final_f97cb7.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml": "137849425/model_final_68d202.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml": "137849458/model_final_280758.pkl",
        "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml": "138204752/model_final_298dad.pkl",
        "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml": "138204841/model_final_3e0943.pkl",
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml": "137851257/model_final_f6e8b1.pkl",
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml": "139173657/model_final_68b088.pkl",
        # COCO Detection with RPN and Fast R-CNN
        "COCO-Detection/rpn_R_50_C4_1x.yaml": "137258005/model_final_450694.pkl",
        "COCO-Detection/rpn_R_50_FPN_1x.yaml": "137258492/model_final_02ce48.pkl",
        "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml": "137635226/model_final_e5f7ce.pkl",
        # COCO Instance Segmentation Baselines with Mask R-CNN
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml": "137259246/model_final_9243eb.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml": "137260150/model_final_4f86c3.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml": "137260431/model_final_a54504.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml": "137849525/model_final_4ce675.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml": "137849551/model_final_84107b.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml": "137849600/model_final_f10217.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml": "138363239/model_final_a2914c.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml": "138363294/model_final_0464b7.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml": "138205316/model_final_a3ec72.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml": "139653917/model_final_2d9806.pkl",  # noqa

    }
def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: a URL to the model
    """
    name = config_path.replace(".yaml", "")
    if config_path in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
        suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
        return _ModelZooUrls.S3_PREFIX + name + "/" + suffix
    raise RuntimeError("{} not available in Model Zoo!".format(name))
def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "detectron2.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


# Resolution of camera streams
# RESOLUTION_X = 640  #640, 1280
# RESOLUTION_Y = 480   #360(BW:cannot work in this PC, min:480)  #480, 720

# Configuration for histogram for depth image
NUM_BINS = 500    #500 x depth_scale = e.g. 500x0.001m=50cm
MAX_RANGE = 10000  #10000xdepth_scale = e.g. 10000x0.001m=10m

AXES_SIZE = 10

# Set test score threshold
SCORE_THRESHOLD = 0.65  #vip-The smaller, the faster.

# TRESHOLD para a frente do robo
THRESHOLD_FRENTE = 0.035
#TRHESHOLD para a altura do robo
THRESHOLD_ALTURA = 0.05


class VideoStreamer:
    """
    Video streamer that takes advantage of multi-threading, and continuously is reading frames.
    Frames are then ready to read when program requires.
    """
    def __init__(self, video_file=None):
        """
        When initialised, VideoStreamer object should be reading frames
        """
        self.setup_image_config(video_file)
        self.configure_streams()
        self.stopped = False

    def start(self):
        """
        Initialise thread, update method will run under thread
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Constantly read frames until stop() method is introduced
        """
        # colorizer = rs.colorizer()
        while True:

            if self.stopped:
                return

            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # Convert image to numpy array and initialise images
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())
            # self.depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    def stop(self):
        self.pipeline.stop()
        self.stopped = True

    def read(self):
        return (self.color_image, self.depth_image)
        # return (self.color_image, self.depth_image, self.depth_colormap)

    def setup_image_config(self, video_file=None):
        """
        Setup config and video steams. If --file is specified as an argument, setup
        stream from file. The input of --file is a .bag file in the bag_files folder.
        .bag files can be created using d435_to_file in the tools folder.
        video_file is by default None, and thus will by default stream from the 
        device connected to the USB.
        """
        config = rs.config()

        if video_file is None:
            
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
        else:
            try:
                config.enable_device_from_file("{}".format(video_file))
                
            except:
                print("Cannot enable device from: '{}'".format(video_file))

        self.config = config

    def configure_streams(self):
        # Configure video streams
        self.pipeline = rs.pipeline()
    
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def get_depth_scale(self):
        return self.profile.get_device().first_depth_sensor().get_depth_scale()



class Predictor(DefaultPredictor):
    def __init__(self):
        self.config = self.setup_predictor_config()
        super().__init__(self.config)

    def create_outputs(self, color_image):
        self.outputs = self(color_image)

    def setup_predictor_config(self):
        """
        Setup config and return predictor. See config/defaults.py for more options
        """

        # cfg_file = get_config_file(config_path)

        # cfg = get_cfg()
        # config_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        # pretrained = True
        # if pretrained:
        #     cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

        #? O CODIGO SEGUINTE QUE ESTA COMENTADO ESTA A FUNCIONAR
        #!
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

        
        dataset_id = "pedro2team/oranges-apples-vases:oranges-apples-vases1.0"
        dataset_train = detectron2_register_dataset(dataset_id, partition='train', split_type='stratified')
        cfg.DATASETS.TRAIN = (dataset_train)


        # cfg.merge_from_file("config.yml")

        # config_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        # cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD






        # # Mask R-CNN ResNet101 FPN weights
        # cfg.DATALOADER.NUM_WORKERS = 2
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # cfg.SOLVER.IMS_PER_BATCH = 8  # batch size
        # cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
        # cfg.SOLVER.MAX_ITER = 5000  # and a good number of iterations
        # cfg.SOLVER.STEPS = (4000, 4500)
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        # cfg.MODEL.WEIGHTS = "model_final.pth"
        # print(f"cfg.MODEL.WEIGHTS_3: {cfg.MODEL.WEIGHTS}")
        
        # cfg = get_cfg()
        # config_path = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
        # cfg.merge_from_file("configs/" + config_path)
        # pretrained = False
        # if pretrained:
        #     cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)

        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
        # # Mask R-CNN ResNet101 FPN weights
        # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  #Load local model
        # # This determines the resizing of the image. At 0, resizing is disabled.
        cfg.INPUT.MIN_SIZE_TEST = 0


        return cfg

    def format_results(self, class_names):
        """
        Format results so they can be used by overlay_instances function
        """
        predictions = self.outputs['instances']
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        labels = None 
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        masks = predictions.pred_masks.cpu().numpy()
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
        boxes_list = boxes.tensor.tolist()
        scores_list = scores.tolist()
        class_list = classes.tolist()

        for i in range(len(scores_list)):
            boxes_list[i].append(scores_list[i])
            boxes_list[i].append(class_list[i])
        

        boxes_list = np.array(boxes_list)

        return (masks, boxes, boxes_list, labels, scores_list, class_list)    



class OptimizedVisualizer(Visualizer):
    """
    Detectron2's altered Visualizer class which converts boxes tensor to cpu
    """
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)
    
    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.cpu().numpy()
        else:
            return np.asarray(boxes)



class DetectedObject:
    """
    Each object corresponds to all objects detected during the instance segmentation
    phase. Associated trackers, distance, position and velocity are stored as attributes
    of the object.
    masks[i], boxes[i], labels[i], scores_list[i], class_list[i]
    """
    def __init__(self, mask, box, label, score, class_name):
        self.mask = mask
        self.box = box
        self.label = label
        self.score = score
        self.class_name = class_name

    #BW: comment below for speed-up! ~5sec/frame faster.
    # def __str__(self):
    #     ret_str = "The pixel mask of {} represents a {} and is {}m away from the camera.\n".format(self.mask, self.class_name, self.distance)
    #     if hasattr(self, 'track'):
    #         if hasattr(self.track, 'speed'):
    #             if self.track.speed >= 0:
    #                 ret_str += "The {} is travelling {}m/s towards the camera\n".format(self.class_name, self.track.speed)
    #             else:
    #                 ret_str += "The {} is travelling {}m/s away from the camera\n".format(self.class_name, abs(self.track.speed))
    #         if hasattr(self.track, 'impact_time'):
    #             ret_str += "The {} will collide in {} seconds\n".format(self.class_name, self.track.impact_time)
    #         if hasattr(self.track, 'velocity'):
    #             ret_str += "The {} is located at {} and travelling at {}m/s\n".format(self.class_name, self.track.position, self.track.velocity)
    #     return ret_str

    def create_vector_arrow(self):
        """
        Creates direction arrow which will use Arrow3D object. Converts vector to a suitable size so that the direction is clear.
        NOTE: The magnitude of the velocity is not represented through this arrow. The arrow lengths are almost all identical
        """
        arrow_ratio = AXES_SIZE / max(abs(self.track.velocity_vector[0]), abs(self.track.velocity_vector[1]), abs(self.track.velocity_vector[2]))
        self.track.v_points = [x * arrow_ratio for x in self.track.velocity_vector]

    

class Arrow3D(FancyArrowPatch):
    """
    Arrow used to demonstrate direction of travel for each object
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



def find_mask_centre(mask, color_image):
    """
    Finding centre of mask using moments
    """
    moments = cv2.moments(np.float32(mask))

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    return cX, cY


def find_median_depth(mask_area, num_median, histg):
    """
    Iterate through all histogram bins and stop at the median value. This is the
    median depth of the mask.
    """
    
    median_counter = 0
    centre_depth = "0.00"
    for x in range(0, len(histg)):
        median_counter += histg[x][0]
        if median_counter >= num_median:
            # Half of histogram is iterated through,
            # Therefore this bin contains the median
            centre_depth = x / 50
            break 

    return float(centre_depth)

def debug_plots(color_image, depth_image, mask, histg, depth_colormap):
    """
    This function is used for debugging purposes. This plots the depth color-
    map, mask, mask and depth color-map bitwise_and, and histogram distrobutions
    of the full image and the masked image.
    """
    full_hist = cv2.calcHist([depth_image], [0], None, [NUM_BINS], [0, MAX_RANGE])
    masked_depth_image = cv2.bitwise_and(depth_colormap, depth_colormap, mask= mask)

    plt.figure()
            
    plt.subplot(2, 2, 1)
    plt.imshow(depth_colormap)

    plt.subplot(2, 2, 2)
    plt.imshow(masks[i].mask)

    plt.subplot(2, 2, 3).set_title(labels[i])
    plt.imshow(masked_depth_image)

    plt.subplot(2, 2, 4)
    plt.plot(full_hist)
    plt.plot(histg)
    plt.xlim([0, 600])
    plt.show()

def on_open(ws):
    print("Connected to WebSocket server.")
    for topic in topics:
        subscribe_message = {"topic": topic, "action": "subscribe"}
        ws.send(json.dumps(subscribe_message))

def on_message(ws, message):
    print(f"Received message: {message}")

def on_close(ws):
    print("Disconnected from WebSocket server.")

def send_message(ws, topics):
    for topic in topics:
        message = {"topic": topic, "data": f"Hello, world! {topic}"}
        ws.send(json.dumps(message))
    # ws.close()

if __name__ == "__main__":

    #? Define the WebSocket URL and topics to subscribe to
    ws_url = "ws://localhost:3306/"
    topics = ["ruben", "mario"]
    
    #? Create a WebSocket instance and attach the handlers
    # ws = websocket.WebSocketApp(ws_url, on_open=on_open)
    ws = websocket.WebSocket()
    # ws.connect("ws://localhost:3306/")
    # send_message(ws, topics)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='type --file=file-name.bag to stream using file instead of webcam')
    args = parser.parse_args()

    # Initialise Detectron2 predictor
    predictor = Predictor()

    # Initialise video streams from D435
    video_streamer = VideoStreamer(video_file=args.file)

    depth_scale = video_streamer.get_depth_scale()
    print("Depth Scale is: {:.4f}m".format(depth_scale))

    speed_time_start = time.time()

    video_streamer.start()
    time.sleep(1)

    result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))


    while True:
        message_to_send = {"fruits":{}, "trees":[]}
        
        time_start = time.time()
        color_image, depth_image = video_streamer.read()

        # color_image, depth_image, depth_colormap = video_streamer.read()
        detected_objects = []

        t1 = time.time()

        camera_time = t1 - time_start

        video_file = False

        if args.file != None:
            video_file = True
        
        if video_file:
            predictor.create_outputs(color_image[:, :, ::-1])
            RESOLUTION_X = 1280
            RESOLUTION_Y = 720
        else:
            predictor.create_outputs(color_image)
            RESOLUTION_X = 640
            RESOLUTION_Y = 480
            
        outputs = predictor.outputs

        t2 = time.time()
        model_time = t2 - t1
        # print("Model took {:.2f} time".format(model_time))

        predictions = outputs['instances']
        

        if outputs['instances'].has('pred_masks'):
            num_masks = len(predictions.pred_masks)
        
        detectron_time = time.time()

        # metadata = MetadataCatalog.get("darwin_oranges-apples-vases_train")

        # dataset_val = detectron2_register_dataset(dataset_id, partition='val', split_type='stratified')

        # Create a new Visualizer object from Detectron2 
        dataset_metadata = MetadataCatalog.get(predictor.config.DATASETS.TRAIN)

        v = OptimizedVisualizer(color_image[:, :, ::-1], metadata=dataset_metadata)
        
        masks, boxes, boxes_list, labels, scores_list, class_list = predictor.format_results(v.metadata.get("thing_classes"))

        # for i in range(len(labels)):
        #     percentages = labels[i].split()[1]
        #     labels[i] = f"apple {percentages}"

        for i in range(num_masks):
            try:
                detected_obj = DetectedObject(masks[i], boxes[i], labels[i], scores_list[i], class_list[i])
            except:
                print("Object doesn't meet all parameters")
            
            detected_objects.append(detected_obj)

        
        v.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=None,
            assigned_colors=None,
            alpha=0.3
        )
        
        speed_time_end = time.time()
        total_speed_time = speed_time_end - speed_time_start
        speed_time_start = time.time()

        R = 6378.1 
        # These values should be replaced with real coordinates
        latDrone = radians(39.73389)
        lonDrone = radians(-8.821944)

        for i in range(num_masks):
            """
            Converting depth image to a histogram with num bins of NUM_BINS 
            and depth range of (0 - MAX_RANGE millimeters)
            """
        
            mask_area = detected_objects[i].mask.area()
            num_median = math.floor(mask_area / 2)
            
            histg = cv2.calcHist([depth_image], [0], detected_objects[i].mask.mask, [NUM_BINS], [0, MAX_RANGE])

            # Uncomment this to use the debugging function
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # debug_plots(color_image, depth_image, masks[i].mask, histg, depth_colormap)
            centre_depth = find_median_depth(mask_area, num_median, histg)
            detected_objects[i].distance = centre_depth
            cX, cY = find_mask_centre(detected_objects[i].mask._mask, v.output)


            # TODO
            #? CONTINUAR A FAZER
            # Calculo dos angulos (d?? a diferenca, ou seja, pode ser um valor negativo, para a esquerda do centro e negativo,
            # vou ter de subtrair ou somar no angulo do drone)

            #? _DEPTH_VGA
            # HFOV = 75
            # VFOV_DEPTH_VGA = 62

            #? _DEPTH_HD
            # HFOV_DEPTH_HD = 78
            # VFOV_DEPTH_HD = 58

            #? _Color_Camera
            HFOV = 69
            VFOV = 42

            CENTER_POINT_X = RESOLUTION_X / 2
            CENTER_POINT_Y = RESOLUTION_Y / 2

            # cx, cy -> mask center point

            #? Angulos da relacao ao centro da camera com o centro da mascara
            H_Angle = ((cX- CENTER_POINT_X)/CENTER_POINT_X)*(HFOV/2)
            V_Angle = ((cY - CENTER_POINT_Y)/CENTER_POINT_Y)*(VFOV/2)

            v.draw_circle((cX, cY), (0, 0, 0))


            #? detected_objects[i].distance = centre_depth -> profundidade media da mascara a camera

            # Calculate Coords of objects
            # 3D coords of object in camera frame (x, y, z) in meters (m)
            # x = detected_objects[i].distance * math.cos(math.radians(V_Angle)) * math.cos(math.radians(H_Angle))
            # y = detected_objects[i].distance * math.cos(math.radians(V_Angle)) * math.sin(math.radians(H_Angle))
            # z = detected_objects[i].distance * math.sin(math.radians(V_Angle))
            
            
            #convert degrees to radians - em vez do 45 deve tar a direcao do drone
            direction = 45 + H_Angle
            if direction > 360:
                direction = direction - 360
            elif direction < 0:
                direction = direction + 360
            brng = radians(direction)


            #? Distancia em linha reta da camera para o objeto
            distanceToFruit = ((centre_depth/cos(radians(H_Angle)))**2 + (centre_depth*tan(radians(V_Angle)))**2)**0.5

                #? Calculos de teste para a distancia da camera ao objecto detetado. Ainda nao testados
                # angle_camera_to_point = (H_Angle**2+V_Angle**2)**0.5
                # new_Distance_2 = centre_depth/cos(radians(angle_camera_to_point))
                # print("new_Distance_2: ", new_Distance_2)

            #? Distancia em linha reta da camera para o objeto com threshold da garra
            depthFromObjectToClawThreshold = round(centre_depth - THRESHOLD_FRENTE, 3)


            #* Calculo do Y (o quanto o braco tem de andar para a esquerda ou direita)
            #* (ap??s multiplicar por -1) -> se objeto estiver a esquerda do centro da camera, o valor ?? positivo
            distancia_lateral = tan(radians(H_Angle)) * centre_depth * -1

            #*!Calculos para acertar o Y consoante a distancia do objeto ao centro da camera lateralmente
            # diferenca = abs(int(distancia_lateral /0.01))
            # if distancia_lateral < 0:
            #     distancia_lateral = distancia_lateral + diferenca * 0.0035
            # if distancia_lateral > 0:
            #     distancia_lateral = distancia_lateral + diferenca * 0.0025
            
            # distancia_lateral = round(distancia_lateral, 3)

            #* Calculo do Z (o quanto o braco tem de andar para cima ou para baixo)
            #* (ap??s multiplicar por -1) -> se objeto estiver acima do centro da camera, o valor ?? positivo
            distancia_vertical = (tan(radians(V_Angle)) * centre_depth * -1)
            #! Calculos para acertar o Z consoante a distancia do objeto ao centro da camera verticalmente
            # distancia_vertical = round((((tan(radians(V_Angle)) * centre_depth) * -1) + THRESHOLD_ALTURA + (sin(radians(5)) * distanceToFruit)), 3)

                
            claw_origin = (THRESHOLD_FRENTE, 0, -THRESHOLD_ALTURA)
            fruit_location = (depthFromObjectToClawThreshold, distancia_lateral, distancia_vertical)
            distance_claw_to_fruit = ((fruit_location[0] - claw_origin[0])**2 + (fruit_location[1] - claw_origin[1])**2 + (fruit_location[2] - claw_origin[2])**2)**0.5

            #? Calculus of the fruit width and height considering the depth to the object
            fruit_width_pixels = detected_objects[i].box.tensor[0][2] - detected_objects[i].box.tensor[0][0]
            fruit_height_pixels = detected_objects[i].box.tensor[0][3] - detected_objects[i].box.tensor[0][1]
            fruit_width = (fruit_width_pixels * centre_depth) / RESOLUTION_X
            fruit_height = (fruit_height_pixels * centre_depth) / RESOLUTION_Y

            #! Calculus das coordenadadas Globais
            # TODO -> ver qual o calculo que est?? correto
            new_Distance_km = distanceToFruit/1000
            
            latFruit = asin(sin(latDrone) * cos(new_Distance_km/R) + cos(latDrone) * sin(new_Distance_km/R) * cos(brng))
            lonFruit = lonDrone + atan2(sin(brng) * sin(new_Distance_km/R) * cos(latDrone), cos(new_Distance_km/R) - sin(latDrone) * sin(latFruit))

            lateral_distance_with_FOV = tan(radians(HFOV)) * centre_depth * 2
            vertical_distance_with_FOV = tan(radians(VFOV)) * centre_depth * 2
            # print(f"Lateral Distance with FOV: {lateral_distance_with_FOV}")
            # print(f"Vertical Distance with FOV: {vertical_distance_with_FOV}")
            #? Heights are to water level
            gimbal_inclination = 0
            drone_height = 100 # Replace 100 with real drone height
            # (talvez seja - em vez de +)
            new_Angle = gimbal_inclination + V_Angle 
            fruit_altitude = drone_height - distanceToFruit * sin(radians(new_Angle))
            
    
            # v.draw_text("{:.2f}m".format(centre_depth), (cX, cY + 20))

            # v.draw_text("{:.2f}m".format(distanceToFruit), (cX, cY + 70))
            # v.draw_text(f"claw_to_fruit: {distance_claw_to_fruit:.2f}", (cX, cY + 20))
            # v.draw_text(f"latFruit:{degrees(latFruit):.8f}\nLon_B:{degrees(lonFruit):.8f}\fruit_altitude:{fruit_altitude:.2f}", (cX, cY + 20))
            # v.draw_text(f"Fruit Width: {fruit_width:.2f}\nFruit Height: {fruit_height:.2f}", (cX, cY + 20))
            # v.draw_text(f"Fruit Width in Meters: {fruit_width:.2f}\nFruit Height in Meters: {fruit_height:.2f}", (cX, cY + 20))
            
            
            
            
            # v.draw_text(f"Lat:{degrees(latFruit):.8f}\nLon:{degrees(lonFruit):.8f}\naltitude:{round(fruit_altitude,3):.2f}", (cX, cY + 20))
            # v.draw_text(f"X:{depthFromObjectToClawThreshold:.3f}m\nY:{distancia_lateral:.3f}m\nZ:{distancia_vertical:3f}m", (cX, cY + 20))
            
            # v.draw_text(f"W: {fruit_width:.2f}\nH: {fruit_height:.2f}", (cX, cY + 20))
            # v.draw_text(f"X: {centre_depth:.3f}m\nY: {distancia_lateral:.3f}m\nZ: {distancia_vertical:3f}m\nD: {distanceToFruit:3f}m", (cX, cY + 20))
            # v.draw_text(f"X:{x:.3f}m\nY:{y:.3f}m\nZ:{z:3f}m", (cX, cY + 20))

            #? PARA O RUBEN
            # v.draw_text(f"H_Angle:{H_Angle:.2f}\nV_Angle:{V_Angle:.2f}", (cX, cY + 35))
            # v.draw_text(f"fruit_altitude:{fruit_altitude:.2f}", (cX, cY + 52))


            v.draw_circle((CENTER_POINT_X, CENTER_POINT_Y), '#eeefff')
            # v.draw_circle((0, 10), '#eeefff')
            # v.draw_circle((boxes[i].tensor[0][0], boxes[i].tensor[0][1]), '#eeefff')
            # v.draw_circle((boxes[i].tensor[0][2], boxes[i].tensor[0][3]), '#eeefff')            


        #     short_label = labels[i].split()[0]
        #     print(f"labels: {labels[i]}")
        #     if short_label == 'apple' or short_label == 'orange':
        #         if short_label in message_to_send["fruits"]:
        #             message_to_send["fruits"][short_label].append({"lat": degrees(latFruit), "lon": degrees(lonFruit), "altitude": fruit_altitude, "fruit_width": fruit_width, "fruit_height": fruit_height})
                    
        #         else:
        #             message_to_send["fruits"][short_label] = [{"lat": degrees(latFruit), "lon": degrees(lonFruit), "altitude": fruit_altitude, "fruit_width": fruit_width, "fruit_height": fruit_height}]

        #     if "potted plant" in labels[i]:
        #         message_to_send["trees"].append({"lat": degrees(latFruit), "lon": degrees(lonFruit), "altitude": fruit_altitude})

        # print(f"{message_to_send}\n\n\n")



        #for i in detected_objects:
            #print(i)

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow('Segmented Image', color_image)

        if video_file:
            cv2.imshow('Segmented Image', v.output.get_image())
            # print(v.output.get_image().shape)
            result.write(v.output.get_image())
        else:
            cv2.imshow('Segmented Image', v.output.get_image()[:,:,::-1])
            result.write(v.output.get_image()[:,:,::-1])
        
        # cv2.imshow('Depth Image', depth_colormap)
        # cv2.imshow('Color Image', color_image)

        #cv2.imshow('Depth', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            result.release()
            break
        
        time_end = time.time()
        total_time = time_end - time_start


        print("Time to process frame: {:.2f}".format(total_time))
        print("FPS: {:.2f}\n".format(1/total_time))

    # cv2.imwrite('output.png', v.output.get_image())
    video_streamer.stop()
    cv2.destroyAllWindows()
