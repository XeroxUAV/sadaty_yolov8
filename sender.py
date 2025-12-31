import socket
import time
import threading
from picamera2 import Picamera2
import cv2

HOST = '192.168.113.105'
PORT = 5000
	
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

picam2 = Picamera2()
picam2.start()

FRAME_INTERVAL = 1 / 20
last_time = time.perf_counter()


def receive_text(sock):
    while True:
        try:
            length_bytes = sock.recv(4)
            if not length_bytes:
                break

            msg_len = int.from_bytes(length_bytes, 'big')

            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(msg_len - len(data))
                if not chunk:
                    break
                data += chunk

            print("FROM LAPTOP:", data.decode('utf-8'))

        except Exception as e:
            print("Text receive error:", e)
            break


recv_thread = threading.Thread(
    target=receive_text,
    args=(s,),
    daemon=True
)
recv_thread.start()

while True:
    now = time.perf_counter()
    sleep_time = FRAME_INTERVAL - (now - last_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    last_time = time.perf_counter()

    frame = picam2.capture_array()
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_size = len(frame_bytes)

    try:
        s.sendall(frame_size.to_bytes(4, 'big') + frame_bytes)
    except (BrokenPipeError, ConnectionResetError):
        print("[ERROR] Receiver disconnected. Exiting loop.")
        break
