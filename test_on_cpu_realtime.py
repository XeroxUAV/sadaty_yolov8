import cv2
import numpy as np
import time
from djitellopy import Tello
import onnxruntime as ort


class FPSTracker:
    def __init__(self, smoothing=0.9):
        self.prev_time = time.time()
        self.fps = 0
        self.smoothing = smoothing  # exponential moving average smoothing

    def update(self):
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        current_fps = 1.0 / (dt + 1e-6)
        # Smooth FPS to avoid jitter
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * current_fps
        return self.fps

# ---------------- Helper functions ----------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_mask(row, box, img_width, img_height):
    mask_size = int(np.sqrt(row.size))   # compute mask height/width
    # print(mask_size)
    mask = np.zeros((img_height, img_width), np.uint8)
    mask = row.reshape(mask_size, mask_size)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype("uint8") * 255

    x1, y1, x2, y2 = box
    mask_x1 = round(x1 / img_width * mask_size)
    mask_y1 = round(y1 / img_height * mask_size)
    mask_x2 = round(x2 / img_width * mask_size)
    mask_y2 = round(y2 / img_height * mask_size)
    # --- Safety check before slicing ---
    if (
        mask_x1 >= mask_x2
        or mask_y1 >= mask_y2
        or mask_x1 < 0
        or mask_y1 < 0
        or mask_x2 > mask_size
        or mask_y2 > mask_size):
        return None

    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    # --- Another safety check before resize ---
    if mask.size == 0:
        return None

    mask = cv2.resize(mask, (max(1, round(x2 - x1)), max(1, round(y2 - y1))))
    return mask.astype("uint8")
def get_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    return [[pt[0][0], pt[0][1]] for pt in contours[0]]

def draw_polygon(img, polygon, color=(0, 255, 0), alpha=0.5):
    if len(polygon) == 0:
        return img
    overlay = img.copy()
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

# ---------------- Load ONNX model ----------------
model_path = r"C:\softwares\yolov8_env64\runs\segment\window_segmentation_model8\weights\best.onnx"
# Enable CPU optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4  # or the number of CPU cores you want to use
session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

# ---------------- Connect Tello ----------------
tello = Tello()
tello.connect()
print("[INFO] Battery:", tello.get_battery())
tello.streamon()
time.sleep(2)  # allow stream to start

yolo_classes = ["window"]

# ---------------- Real-time loop ----------------
fps_tracker = FPSTracker()
input_size = 480
mask_res = 120
try:
    while True:
        total_start = time.time()
        frame = tello.get_frame_read().frame
        img_height, img_width = frame.shape[:2]

        # ---------------- Preprocess ----------------
        t0 = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img_rgb, (input_size, input_size)).astype("float32") / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))[None, ...]
        t1 = time.time()
        # ---------------- Run ONNX ----------------
        outputs = session.run(output_names, {input_name: input_img})
        output0, output1 = outputs[0], outputs[1]
        output0 = output0[0].transpose()
        output1 = output1[0].reshape(32, mask_res*mask_res)
        t2 = time.time()

        conf_mask = output0[:, 4] > 0.3
        output0 = output0[conf_mask]
        output0 = output0[:200]
        boxes = output0[:, :5]
        masks = np.matmul(output0[:, 5:], output1, out=np.empty((output0.shape[0], output1.shape[1]), dtype=np.float32))
        boxes = np.hstack([boxes, masks])
        t3 = time.time()

        # ---------------- Process boxes ----------------
        objects = []
        for row in boxes:
            xc, yc, w, h = row[:4]
            x1 = (xc - w/2)/input_size*img_width
            y1 = (yc - h/2)/input_size*img_height
            x2 = (xc + w/2)/input_size*img_width
            y2 = (yc + h/2)/input_size*img_height
            prob = row[4].max()
            if prob < 0.7:
                continue
            mask_vector = row[5:]  # all mask elements
            mask = get_mask(mask_vector, (x1, y1, x2, y2), img_width, img_height)
            if mask is None:
                continue  # skip invalid mask
            polygon = get_polygon(mask)
            objects.append([x1, y1, x2, y2, yolo_classes[0], prob, polygon])
        t4 = time.time()

        # ---------------- Draw ----------------
        for obj in objects:
            x1, y1, x2, y2, label, prob, polygon = obj
            # Adjust polygon to image coordinates
            abs_polygon = [(round(x1 + pt[0]), round(y1 + pt[1])) for pt in polygon]
            frame = draw_polygon(frame, abs_polygon, color=(0, 255, 0), alpha=0.6)
            # cv2.putText(frame, f"{label} {prob:.2f}", (int(x1), int(y1)-5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        t5 = time.time()
        fps = fps_tracker.update()
        print("[INFO] FPS: {:.2f}".format(fps))
        cv2.imshow("Tello Real-time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        total_end = time.time()

        print(f"[TIMING] preprocess: {(t1 - t0)*1000:.1f} ms | "
              f"inference: {(t2 - t1)*1000:.1f} ms | "
              f"post-matrix: {(t3 - t2)*1000:.1f} ms | "
              f"mask/polygon: {(t4 - t3)*1000:.1f} ms | "
              f"drawing: {(t5 - t4)*1000:.1f} ms | "
              f"total frame: {(total_end - total_start)*1000:.1f} ms | "
              f"FPS: {fps:.1f}")

except Exception as e:
    print("[ERROR]", e)

finally:
    cv2.destroyAllWindows()
    tello.streamoff()
    tello.end()
