import websocket
import json

def send_message(ws, topic):
    message = {"topic": topic, "data": "connected to stream"}
    ws.send(json.dumps(message))

ws = websocket.WebSocket()
ws.connect("ws://127.0.0.1:3306/")
send_message(ws, "stream")
print(ws.recv())
print(ws.recv())