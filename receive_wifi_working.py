import socket
import cv2
import numpy as np
import time


HOST = '0.0.0.0'
PORT = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print("Waiting for connection...")
conn, addr = s.accept()
print("Connected by", addr)

data = b""
payload_size = 4  # We'll send frame length first

fps = 0
fps_count = 0
fps_start = time.perf_counter()
while True:
    t1 = time.time()
    # Receive frame size
    while len(data) < payload_size:
        data += conn.recv(4096)
    frame_size = int.from_bytes(data[:payload_size], 'big')
    data = data[payload_size:]

    # Receive frame data
    while len(data) < frame_size:
        data += conn.recv(4096)
    frame_data = data[:frame_size]
    data = data[frame_size:]

    # Decode and show
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    fps_count += 1
    elapsed = time.perf_counter() - fps_start

    if elapsed >= 1.0:
        fps = fps_count / elapsed
        fps_count = 0
        fps_start = time.perf_counter()

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow('Raspberry Pi Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
cv2.destroyAllWindows()
