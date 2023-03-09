import socket
import struct
import pyrealsense2 as rs
import numpy as np
import cv2

# Set up socket connection
HOST = 'localhost'
PORT = 8889
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
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print("Depth Scale is: {:.4f}m".format(depth_scale))

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

            # Convert depth frame to a numpy array for visualization
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert color frame to a numpy array for visualization
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


            # Display the depth and color frames on the server side
            # cv2.imshow("Depth Stream (Server)", depth_image)
            # cv2.imshow("Color Stream (Server)", color_image)
            # cv2.imshow("Depth ColorMap Stream (Server)", depth_colormap)

            # Send depth frame size and data
            depth_data = depth_image.tobytes()
            depth_size = struct.pack('I', len(depth_data))
            conn.send(depth_size)
            conn.send(depth_data)

            # Send color frame size and data
            color_data = color_image.tobytes()
            color_size = struct.pack('I', len(color_data))
            conn.send(color_size)
            conn.send(color_data)

            # Check for key press to exit
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break

        # Clean up the connection and window on server exit
        cv2.destroyAllWindows()
        conn.close()

finally:
    pipeline.stop()
    sock.close()




























# import socket
# import struct
# import pyrealsense2 as rs
# import cv2
# import numpy as np

# # Set up socket connection
# HOST = 'localhost'
# PORT = 8880
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.bind((HOST, PORT))
# sock.listen(1)
# print(f"Waiting for client to connect on port {PORT}...")

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# # Main loop
# try:
#     while True:
#         # Wait for a client to connect
#         conn, addr = sock.accept()
#         print(f"Client connected from {addr}")
        
#         # Send depth and color frames
#         while True:
#             # Wait for a new frame
#             frames = pipeline.wait_for_frames()
#             depth_frame = frames.get_depth_frame()
#             color_frame = frames.get_color_frame()

#             # Send depth frame size and data
#             depth_image = depth_frame.get_data()
#             depth_data = bytearray(depth_image)
#             depth_size = struct.pack('I', len(depth_data))
#             conn.send(depth_size)
#             conn.send(depth_data)

#             # Send color frame size and data
#             color_image = color_frame.get_data()
#             color_data = bytearray(color_image)
#             color_size = struct.pack('I', len(color_data))
#             conn.send(color_size)
#             conn.send(color_data)

#             # cv2.imshow('Depth Stream', depth_image)
#             # print(f"depth image: {depth_image}")

#             color_image_cv = np.asanyarray(color_frame.get_data())
#             depth_image_cv = np.asanyarray(depth_frame.get_data())

#             cv2.imshow('Color Stream', color_image_cv)
#             cv2.imshow('Depth Stream', depth_image_cv)

# finally:
#     pipeline.stop()
#     sock.close()