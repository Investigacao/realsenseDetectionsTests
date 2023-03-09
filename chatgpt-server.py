import socket
import struct
import pyrealsense2 as rs
import cv2
import numpy as np

# Set up socket connection
HOST = 'localhost'
PORT = 8880
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"Waiting for client to connect on port {PORT}...")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Main loop
try:
    while True:
        # Wait for a client to connect
        conn, addr = sock.accept()
        print(f"Client connected from {addr}")
        
        # Send depth and color frames
        while True:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Send depth frame size and data
            depth_image = depth_frame.get_data()
            depth_data = bytearray(depth_image)
            depth_size = struct.pack('I', len(depth_data))
            conn.send(depth_size)
            conn.send(depth_data)

            # Send color frame size and data
            color_image = color_frame.get_data()
            color_data = bytearray(color_image)
            color_size = struct.pack('I', len(color_data))
            conn.send(color_size)
            conn.send(color_data)

            # cv2.imshow('Depth Stream', depth_image)
            # print(f"depth image: {depth_image}")

            color_image_cv = np.asanyarray(color_frame.get_data())
            depth_image_cv = np.asanyarray(depth_frame.get_data())

            cv2.imshow('Color Stream', color_image_cv)
            cv2.imshow('Depth Stream', depth_image_cv)

finally:
    pipeline.stop()
    sock.close()

























# import socket
# import cv2
# import numpy as np
# import pyrealsense2 as rs

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# # Create socket
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind(('127.0.0.1', 8000))
# server_socket.listen(0)
# client_socket, addr = server_socket.accept()

# # Send depth and color frames to client
# try:
#     while True:
#         # Wait for a coherent pair of frames
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()

#         # Convert depth frame to numpy array
#         depth_image = np.asanyarray(depth_frame.get_data())

#         # Convert color frame to OpenCV format
#         color_image = np.asanyarray(color_frame.get_data())
#         color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

#         # Serialize depth and color frames as bytes and send to client
#         depth_bytes = depth_image.tobytes()
#         color_bytes = color_image.tobytes()
#         client_socket.sendall(depth_bytes)
#         client_socket.sendall(color_bytes)
# finally:
#     # Stop streaming and close socket
#     pipeline.stop()
#     client_socket.close()
#     server_socket.close()


















# # import socket
# # import struct
# # import cv2
# # import numpy as np
# # import pyrealsense2 as rs

# # # Set up TCP socket
# # TCP_IP = '127.0.0.1'
# # TCP_PORT = 8000
# # BUFFER_SIZE = 1024

# # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # sock.connect((TCP_IP, TCP_PORT))

# # # Set up RealSense pipeline and config
# # pipeline = rs.pipeline()
# # config = rs.config()
# # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # # Start pipeline and create colorizer
# # pipeline.start(config)
# # colorizer = rs.colorizer()

# # try:
# #     while True:
# #         # Wait for a coherent pair of frames: depth and color
# #         frames = pipeline.wait_for_frames()
# #         depth_frame = frames.get_depth_frame()
# #         color_frame = frames.get_color_frame()

# #         if not depth_frame or not color_frame:
# #             continue

# #         # Convert depth frame to meters and colorize it
# #         depth_frame = depth_frame.apply_filter(rs.decimation_filter())
# #         depth_frame = depth_frame.apply_filter(rs.spatial_filter())
# #         depth_frame = depth_frame.apply_filter(rs.temporal_filter())
# #         depth_frame = depth_frame.apply_filter(rs.hole_filling_filter())
# #         depth_frame_meters = depth_frame * depth_frame.get_units()
# #         depth_frame_colored = colorizer.colorize(depth_frame)

# #         # Convert color and depth frames to numpy arrays
# #         color_image = np.asanyarray(color_frame.get_data())
# #         depth_image_colored = np.asanyarray(depth_frame_colored.get_data())

# #         # Pack and send color and depth data over TCP
# #         packed_color_data = cv2.imencode('.jpg', color_image)[1].tobytes()
# #         packed_depth_data = b''
# #         for depth_value in depth_frame_meters.flatten():
# #             packed_depth_data += struct.pack('<f', depth_value)
# #         packed_data = packed_color_data + packed_depth_data
# #         sock.sendall(packed_data)

# # except KeyboardInterrupt:
# #     pass

# # finally:
# #     # Stop pipeline and close socket
# #     pipeline.stop()
# #     sock.close()