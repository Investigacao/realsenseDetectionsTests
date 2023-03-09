import socket
import struct

# Set up TCP socket
TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)

print(f'Server running on {TCP_IP}:{TCP_PORT}')

# Accept connection from client
conn, addr = sock.accept()
print(f'Connected by {addr}')

# Connect to server
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.connect(('localhost', 8000))
print('Connected to server')

try:
    while True:
        # Receive packed data from client
        packed_data = b''
        while True:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                break
            packed_data += data

        # Send packed data to server
        server_sock.sendall(packed_data)

        # Receive packed data from server
        packed_data = b''
        while True:
            data = server_sock.recv(BUFFER_SIZE)
            if not data:
                break
            packed_data += data

        # Send packed data to client
        conn.sendall(packed_data)

finally:
    # Close sockets
    conn.close()
    sock.close()
    server_sock.close()