import socket
import struct
import numpy as np
import cv2
from math import *
import time
import argparse
import os
import pkg_resources

from threading import Thread
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo

import torch, torchvision

import pyrealsense2 as rs
from darwin.torch.utils import detectron2_register_dataset
import subprocess

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

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

        dataset_id = "pedro2team/oranges-apples-vases:oranges-apples-vases1.0"
        dataset_train = detectron2_register_dataset(dataset_id, partition='train', split_type='stratified')
        cfg.DATASETS.TRAIN = (dataset_train)

        # This determines the resizing of the image. At 0, resizing is disabled.
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

        #masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
        masks = [GenericMask(x, 480, 640) for x in masks]
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


if __name__ == "__main__":

    # Set up socket connection
    HOST = '192.168.1.20'
    PORT = 8880
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    # Create windows to display the depth and color streams on the client side
    # cv2.namedWindow("Depth Stream (Client)", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Depth Stream (Client)", 640, 480)
    # cv2.namedWindow("Color Stream (Client)", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Color Stream (Client)", 640, 480)
    cv2.namedWindow("Segmented Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmented Image", 960, 900)

    colorizer = rs.colorizer()

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='type --file=file-name.bag to stream using file instead of webcam')
    args = parser.parse_args()

    # Initialise Detectron2 predictor
    predictor = Predictor()

    # Initialise video streams from D435

    # depth_scale = video_streamer.get_depth_scale()
    # print("Depth Scale is: {:.4f}m".format(depth_scale))

    speed_time_start = time.time()

    # command = ['ffmpeg',
    # '-y',
    # '-f', 'rawvideo',
    # '-vcodec', 'rawvideo',
    # '-pix_fmt', 'rgb24',
    # '-s', "{}x{}".format(640,480),
    # '-r', str(5),
    # '-i', '-',
    # '-c:v', 'libx264',
    # '-pix_fmt', 'yuv420p',
    # '-f', 'flv',
    # '-flvflags', 'no_duration_filesize',
    # 'rtmp://127.0.0.1/live/test']

    # p = subprocess.Popen(command, stdin=subprocess.PIPE)


    # result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))

    # Receive frames
    try:
        while True:
            time_start = time.time()
            # Receive depth frame size and data
            depth_size_data = sock.recv(struct.calcsize('I'))
            depth_size = struct.unpack('I', depth_size_data)[0]
            depth_data = b''
            while len(depth_data) < depth_size:
                depth_data += sock.recv(depth_size - len(depth_data))
            depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

            print(f"depth_image: {depth_image}")

            # Receive color frame size and data
            color_size_data = sock.recv(struct.calcsize('I'))
            color_size = struct.unpack('I', color_size_data)[0]
            color_data = b''
            while len(color_data) < color_size:
                color_data += sock.recv(color_size - len(color_data))
            color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            detected_objects = []

            t1 = time.time()
            camera_time = t1 - time_start

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

            dataset_metadata = MetadataCatalog.get(predictor.config.DATASETS.TRAIN)

            v = OptimizedVisualizer(color_image[:, :, ::-1], metadata = dataset_metadata)

        
            masks, boxes, boxes_list, labels, scores_list, class_list = predictor.format_results(v.metadata.get("thing_classes"))

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
                num_median = floor(mask_area / 2)
                histg = cv2.calcHist([depth_image], [0], detected_objects[i].mask.mask, [NUM_BINS], [0, MAX_RANGE])

                # Uncomment this to use the debugging function
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # debug_plots(color_image, depth_image, masks[i].mask, histg, depth_colormap)
                centre_depth = find_median_depth(mask_area, num_median, histg)
                detected_objects[i].distance = centre_depth
                cX, cY = find_mask_centre(detected_objects[i].mask._mask, v.output)

                #? _Color_Camera
                HFOV = 69
                VFOV = 42

                CENTER_POINT_X = RESOLUTION_X / 2
                CENTER_POINT_Y = RESOLUTION_Y / 2

                #? Angulos da relacao ao centro da camera com o centro da mascara
                H_Angle = ((cX- CENTER_POINT_X)/CENTER_POINT_X)*(HFOV/2)
                V_Angle = ((cY - CENTER_POINT_Y)/CENTER_POINT_Y)*(VFOV/2)

                v.draw_circle((cX, cY), (0, 0, 0))

                #? detected_objects[i].distance = centre_depth -> profundidade media da mascara a camera

                #convert degrees to radians - em vez do 45 deve tar a direcao do drone
                direction = 45 + H_Angle
                if direction > 360:
                    direction = direction - 360
                elif direction < 0:
                    direction = direction + 360
                brng = radians(direction)

                #? Distancia em linha reta da camera para o objeto
                distanceToFruit = ((centre_depth/cos(radians(H_Angle)))**2 + (centre_depth*tan(radians(V_Angle)))**2)**0.5

                #? Distancia em linha reta da camera para o objeto com threshold da garra
                depthFromObjectToClawThreshold = round(centre_depth - THRESHOLD_FRENTE, 3)

                new_Distance_to_Claw = (((centre_depth - 3.5)/cos(radians(H_Angle)))**2 + (((centre_depth-3.5)*tan(radians(V_Angle)))+5)**2)**0.5

                #? Relative Coordinates calculation
                #* Calculo do Y (o quanto o braco tem de andar para a esquerda ou direita)
                #* (após multiplicar por -1) -> se objeto estiver a esquerda do centro da camera, o valor é positivo
                distancia_lateral = (tan(radians(H_Angle)) * centre_depth * -1 )
                distancia_lateral = 0.046*(distancia_lateral)**2 + 0.863*(distancia_lateral) + 0.038
                distancia_lateral = round(distancia_lateral, 3)

                #* Calculo do Z (o quanto o braco tem de andar para cima ou para baixo)
                #* (após multiplicar por -1) -> se objeto estiver acima do centro da camera, o valor é positivo
                distancia_vertical = (tan(radians(V_Angle)) * centre_depth * -1) + THRESHOLD_ALTURA
                #! Calculos para acertar o Z consoante a distancia do objeto ao centro da camera verticalmente
                if distancia_vertical < -0.02:
                    distancia_vertical += 0.025
                elif distancia_vertical < 0:
                    distancia_vertical += 0.032
                elif distancia_vertical < 0.025:
                    distancia_vertical += 0.035
                elif distancia_vertical < 0.05:
                    distancia_vertical += 0.043
                elif distancia_vertical < 0.075:
                    depthFromObjectToClawThreshold += 0.01
                    distancia_vertical += 0.05
                elif distancia_vertical < 0.1:
                    depthFromObjectToClawThreshold += 0.015
                    distancia_vertical += 0.057
                elif distancia_vertical < 0.125:
                    depthFromObjectToClawThreshold += 0.02
                    distancia_vertical += 0.064
                
                #? Calculus of the fruit width and height considering the depth to the object
                fruit_width_pixels = (detected_objects[i].box.tensor[0][2] - detected_objects[i].box.tensor[0][0]).item()
                fruit_height_pixels = (detected_objects[i].box.tensor[0][3] - detected_objects[i].box.tensor[0][1]).item()
                fruit_width = ((fruit_width_pixels * distanceToFruit) / RESOLUTION_X)
                fruit_height = ((fruit_height_pixels * distanceToFruit) / RESOLUTION_Y)

                claw_origin = (0.035, 0, -0.05)
                fruit_location = (depthFromObjectToClawThreshold, distancia_lateral, distancia_vertical)
                distance_claw_to_fruit = ((fruit_location[0] - claw_origin[0])**2 + (fruit_location[1] - claw_origin[1])**2 + (fruit_location[2] - claw_origin[2])**2)**0.5
                
                v.draw_text(f"X: {centre_depth:.3f}m\nY: {distancia_lateral:.3f}m\nZ: {distancia_vertical:3f}m\nD: {distance_claw_to_fruit:3f}m", (cX, cY + 20))
                
                v.draw_circle((CENTER_POINT_X, CENTER_POINT_Y), '#eeefff')


            cv2.imshow('Segmented Image', v.output.get_image()[:,:,::-1])
            # Write the color image to the RTMP output stream




            #? RTMP
            # rtmp_frames = v.output.get_image()
            # p.stdin.write(rtmp_frames.tobytes())


            

            # cv2.imshow('Segmented Image', cv2.resize(v.output.get_image()[:,:,::-1], (960, 900)))

            # Display the depth and color frames on the client side
            # cv2.imshow("Color Stream (Client)", color_image)
            cv2.imshow("Depth Stream (Client)", depth_image.astype(np.float32) / 65535.0)
            cv2.imshow("Depth ColorMap Stream (Client)", depth_colormap)

            # Check for key press to exit
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break

    finally:
        # Clean up the connection and windows on client exit
        cv2.destroyAllWindows()
        sock.close()



























# import socket
# import struct
# import numpy as np
# import cv2
# import time

# # Set up socket connection
# HOST = 'localhost'
# PORT = 8880
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((HOST, PORT))

# # Receive frames
# try:
#     while True:
#         # Receive depth frame size and data
#         depth_size_data = sock.recv(struct.calcsize('I'))
#         depth_size = struct.unpack('I', depth_size_data)[0]
#         depth_data = b''
#         while len(depth_data) < depth_size:
#             depth_data += sock.recv(depth_size - len(depth_data))
#         depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

#         # Receive color frame size and data
#         color_size_data = sock.recv(struct.calcsize('I'))
#         color_size = struct.unpack('I', color_size_data)[0]
#         color_data = b''
#         while len(color_data) < color_size:
#             color_data += sock.recv(color_size - len(color_data))
#         color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)

#         print(f"depth image shape: {depth_image.shape}")
#         print(f"depth image: {depth_image}")

#         # Display frames
#         cv2.imshow('Color Stream', color_image)
#         # cv2.imshow('Depth Stream', depth_image.astype(np.float32) / 65535.0)
#         cv2.imshow('Depth Stream', depth_image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     sock.close()
#     cv2.destroyAllWindows()
