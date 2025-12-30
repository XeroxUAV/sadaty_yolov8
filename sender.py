import socket
from picamera2 import Picamera2
import cv2
HOST = '192.168.225.105'  # Replace with your laptop's IP
PORT = 5000

# TCP connection
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

# Initialize PiCamera
picam2 = Picamera2()
picam2.start()

FRAME_INTERVAL = 1 / 20  # 20 FPS
last_time = time.perf_counter()

while True:
	now = time.perf_counter()
	sleep_time = FRAME_INTERVAL - (now - last_time)
	if sleep_time > 0:
		time.sleep(sleep_time)
	last_time = time.perf_counter()

    frame = picam2.capture_array()  # numpy array (H, W, 3)
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_size = len(frame_bytes)

    # Send size + frame
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.sendall(frame_size.to_bytes(4, 'big') + frame_bytes)
