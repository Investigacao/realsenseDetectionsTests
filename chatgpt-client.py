import socket
import struct
import numpy as np
import cv2
import time

# Set up socket connection
HOST = 'localhost'
PORT = 8880
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# Receive frames
try:
    while True:
        # Receive depth frame size and data
        depth_size_data = sock.recv(struct.calcsize('I'))
        depth_size = struct.unpack('I', depth_size_data)[0]
        depth_data = b''
        while len(depth_data) < depth_size:
            depth_data += sock.recv(depth_size - len(depth_data))
        depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

        # Receive color frame size and data
        color_size_data = sock.recv(struct.calcsize('I'))
        color_size = struct.unpack('I', color_size_data)[0]
        color_data = b''
        while len(color_data) < color_size:
            color_data += sock.recv(color_size - len(color_data))
        color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)

        print(f"depth image shape: {depth_image.shape}")
        print(f"depth image: {depth_image}")

        # Display frames
        cv2.imshow('Color Stream', color_image)
        # cv2.imshow('Depth Stream', depth_image.astype(np.float32) / 65535.0)
        cv2.imshow('Depth Stream', depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    sock.close()
    cv2.destroyAllWindows()




















# import socket
# import cv2
# import numpy as np

# # Create socket
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('127.0.0.1', 8000))

# # Receive depth and color frames from server and display
# try:
#     while True:
#         # Receive depth bytes and convert to numpy array
#         depth_bytes = b''
#         while len(depth_bytes) < 921600:
#             depth_bytes += client_socket.recv(921600 - len(depth_bytes))
#         depth_image = np.frombuffer(depth_bytes, dtype=np.uint16)
#         depth_image = depth_image.reshape((480, 640))

#         # Receive color bytes and convert to numpy array
#         color_bytes = b''
#         while len(color_bytes) < 921600:
#             color_bytes += client_socket.recv(921600 - len(color_bytes))
#         color_image = np.frombuffer(color_bytes, dtype=np.uint8)
#         color_image = color_image.reshape((480, 640, 3))

#         # Display depth and color frames
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#         cv2.imshow('Depth', depth_colormap)
#         cv2.imshow('Color', color_image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     # Close socket and destroy windows
#     client_socket.close()
#     cv2.destroyAllWindows()















# import socket
# import struct
# import cv2
# import numpy as np

# # Set up TCP socket
# TCP_IP = '127.0.0.1'
# TCP_PORT = 5005
# BUFFER_SIZE = 1024

# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((TCP_IP, TCP_PORT))

# # Set up OpenCV windows
# cv2.namedWindow('Color Stream', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Depth Stream', cv2.WINDOW_NORMAL)

# try:
#     while True:
#         # Receive packed data from server
#         packed_data = b''
#         while True:
#             data = sock.recv(BUFFER_SIZE)
#             if not data:
#                 break
#             packed_data += data

#         # Unpack color and depth data from packed data
#         color_size = len(packed_data) // 2
#         packed_color_data = packed_data[:color_size]
#         packed_depth_data = packed_data[color_size:]
#         color_data = cv2.imdecode(np.frombuffer(packed_color_data, dtype=np.uint8), cv2.IMREAD_COLOR)
#         depth_data = np.frombuffer(packed_depth_data, dtype=np.float32).reshape((480, 640))

#         # Normalize depth data and convert to 8-bit grayscale
#         depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#         # Display color and depth streams
#         cv2.imshow('Color Stream', color_data)
#         cv2.imshow('Depth Stream', depth_data_normalized)

#         if cv2.waitKey(1) == 27:  # Press Esc to exit
#             break

# finally:
#     # Close socket and destroy OpenCV windows
#     sock.close()
#     cv2.destroyAllWindows()