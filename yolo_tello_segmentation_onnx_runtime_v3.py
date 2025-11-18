import cv2
import numpy as np
from djitellopy import Tello
import time
import keyboard
import onnxruntime as ort

# ---------------- PID Class ----------------
class PID:
    def __init__(self, Kp=0.02, Ki=0.0005, Kd=0.02):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.last_output = 0

    def update(self, error):
        # ----- Anti-windup -----
        self.integral += error
        self.integral = np.clip(self.integral, -100, 100)

        derivative = error - self.prev_error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        # ----- Output smoothing -----
        output = 0.7 * self.last_output + 0.3 * output

        self.prev_error = error
        self.last_output = output
        return output

# ---------------- Kalman Filter ----------------
class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.zeros((4, 1), np.float32)
        self.kalman.statePost = np.zeros((4, 1), np.float32)

    def predict(self):
        prediction = self.kalman.predict()
        return prediction

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)



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
# ---------------- Frame Preprocessing ----------------
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    frame_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    frame_eq = cv2.GaussianBlur(frame_eq, (3, 3), 0)
    return frame_eq

# ---------------- Window Detector ----------------
class WindowDetector:
    def __init__(self, model_path=r"C:\softwares\yolov8_env64\runs\segment\window_segmentation_model8\weights\best.onnx",
                 conf_thresh=0.92):
        # self.kf = KalmanFilter()
        self.alpha = 0.2
        self.smoothed_center = None
        self.no_detection_start_time = None
        self.searching = False
        self.model_path = r"C:\softwares\yolov8_env64\runs\segment\window_segmentation_model8\weights\best.onnx"
        # Enable CPU optimizations
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.intra_op_num_threads = 4  # or the number of CPU cores you want to use
        self.session = ort.InferenceSession(model_path, sess_options=self.sess_options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.yolo_classes = ["window"]
        self.fps_tracker = FPSTracker()
        self.conf_thresh = conf_thresh
        self.forward_done = False
        self.search_yaw_speed = 20  # yaw speed for search mode (adjust 20–40)
        self.device = "cpu"
        print("[INFO] YOLO model loaded.")
        self.big_x = 0
        self.last_x_error = 0
        self.x_big = 0.05


    def perform_search(self, tello):
        print("[INFO] Searching...")
        current_time = time.time()
        if not hasattr(self, "last_search_time"):
            self.last_search_time = current_time

        dt = current_time - self.last_search_time
        if dt <= 0 or dt > 1:
            dt = 0.05
        self.last_search_time = current_time

        if not hasattr(self, "first_search_done"):
            self.first_search_done = False
            self.search_yaw_angle = 0
            self.search_direction = 1
            self.throttle_direction = 1
            self.throttle_magnitude = 10
            self.throttle_time_accumulator = 0
            self.throttle_speed = self.search_yaw_speed / 2

        if not self.first_search_done:
            delta_angle = self.search_direction * self.search_yaw_speed * dt
            self.search_yaw_angle += delta_angle
            if self.search_yaw_angle >= 140:
                self.search_yaw_angle = 140
                self.search_direction = -1
            elif self.search_yaw_angle <= -140:
                self.search_yaw_angle = -140
                self.first_search_done = True
            tello.send_rc_control(0, 0, 0, self.search_direction * self.search_yaw_speed)
        else:
            delta_angle = self.search_direction * self.search_yaw_speed * dt
            self.search_yaw_angle += delta_angle
            if self.search_yaw_angle >= 140:
                self.search_yaw_angle = 140
                self.search_direction = -1
            elif self.search_yaw_angle <= -140:
                self.search_yaw_angle = -140
                self.search_direction = 1
            self.throttle_time_accumulator += dt
            yaw_cycle_duration = (140 * 2) / self.search_yaw_speed
            throttle_toggle_interval = yaw_cycle_duration * 2
            if self.throttle_time_accumulator >= throttle_toggle_interval:
                self.throttle_direction = -self.throttle_direction
                self.throttle_time_accumulator = 0
                self.throttle_magnitude = min(self.throttle_magnitude + 5, 100)
            throttle_change = self.throttle_direction * self.throttle_magnitude
            yaw_change = self.search_direction * self.search_yaw_speed
            tello.send_rc_control(0, 0, throttle_change, yaw_change)

    def process(self, frame, pid_yaw, tello, pid_throttle, pid_roll, pid_forward):
        total_start = time.time()
        if self.forward_done:
            pid_yaw.integral = 0
            pid_yaw.prev_error = 0
            pid_roll.integral = 0
            pid_roll.prev_error = 0
            pid_throttle.integral = 0
            pid_throttle.prev_error = 0
            tello.send_rc_control(0, 2, 0, 0)
            return frame
        input_size = 704
        mask_res = 176


        frame_h, frame_w = frame.shape[:2]
        VERTICAL_OFFSET = -145
        frame_center = (frame_w // 2, frame_h // 2 + VERTICAL_OFFSET)
        cv2.circle(frame, (frame_center[0], frame_center[1]), 8 , (0, 255, 255), 2)
        img_height, img_width = frame.shape[:2]
        t0 = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img_rgb, (input_size, input_size)).astype("float32") / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))[None, ...]
        t1 = time.time()


        raw_frame = frame.copy()
        frame_draw = frame.copy()
        preprocessed = preprocess_frame(raw_frame)


        outputs = self.session.run(self.output_names, {self.input_name: input_img})
        output0, output1 = outputs[0], outputs[1]
        output0 = output0[0].transpose()
        output1 = output1[0].reshape(32, mask_res*mask_res)
        t2 = time.time()

        conf_mask = output0[:, 4] > 0.6
        output0 = output0[conf_mask]
        output0 = output0[:200]
        boxes = output0[:, :5]
        masks = np.matmul(output0[:, 5:], output1, out=np.empty((output0.shape[0], output1.shape[1]), dtype=np.float32))
        boxes = np.hstack([boxes, masks])
        t3 = time.time()

        # --- Use only the single largest detected mask ---
        best_area = 0
        best_obj = None

        for row in boxes:
            xc, yc, w, h = row[:4]
            x1 = (xc - w / 2) / input_size * img_width
            y1 = (yc - h / 2) / input_size * img_height
            x2 = (xc + w / 2) / input_size * img_width
            y2 = (yc + h / 2) / input_size * img_height
            prob = row[4].max()
            if prob < 0.7:
                continue

            mask_vector = row[5:]
            mask = get_mask(mask_vector, (x1, y1, x2, y2), img_width, img_height)
            if mask is None:
                continue

            # find contour area for mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            area = max(cv2.contourArea(c) for c in contours)
            if area > best_area:
                polygon = get_polygon(mask)
                best_area = area
                best_obj = [x1, y1, x2, y2, self.yolo_classes[0], prob, polygon]

        # Replace "objects" list with only the best one
        objects = [best_obj] if best_obj is not None else []

        t4 = time.time()

        # ---------------- Draw ----------------
        for obj in objects:
            x1, y1, x2, y2, label, prob, polygon = obj
            # Adjust polygon to image coordinates
            abs_polygon = [(round(x1 + pt[0]), round(y1 + pt[1])) for pt in polygon]
            frame = draw_polygon(frame, abs_polygon, color=(0, 255, 0), alpha=0.5)

        t5 = time.time()
        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        full_mask_bool = np.zeros((img_height, img_width), dtype=np.uint8)

        for obj in objects:
            x1, y1, x2, y2, label, prob, polygon = obj
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Get the per-object mask again
            mask_vector = row[5:]  # if you have "row" from earlier in the same scope
            mask = get_mask(mask_vector, (x1, y1, x2, y2), img_width, img_height)
            if mask is None:
                continue

            # Threshold mask → convert grayscale [0–255] to binary (0,255)
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            # Clamp coordinates to stay within frame
            x1 = max(0, min(x1, img_width))
            x2 = max(0, min(x2, img_width))
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))

            # Skip zero-area or invalid boxes
            if x2 <= x1 or y2 <= y1:
                # print(f"[WARNING] Invalid box skipped: ({x1},{y1})–({x2},{y2})")
                continue


            # Merge cropped mask into full-frame mask safely
            try:
                full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask)
                # Convert 0/255 mask to boolean: 0 -> False, 255 -> True
                full_mask_bool = full_mask.astype(bool)

            except ValueError as e:
                # print(f"[ERROR] Mask merge failed at ({x1},{y1},{x2},{y2}): {e}")
                continue

        # Show final full-frame mask
        # cv2.imshow("Full Frame Mask", full_mask)
        total_end = time.time()
        fps = self.fps_tracker.update()
        print("[INFO] FPS: {:.2f}".format(fps))
        # print("mask type:", type(full_mask_bool))
        # print("mask shape:", full_mask_bool.shape)
        # print("mask dtype:", full_mask_bool.dtype)
        # print("mask :",
        #       full_mask)
        # print(np.unique(full_mask_bool))

        best_area = 0
        approx_poly = None
        final_mask_bin = None

        # ----------- NEW MASK HANDLING ----------------
        # full_mask_bool is your boolean mask (True/False)
        mask_resized = cv2.resize(
            full_mask_bool.astype(np.uint8),  # convert bool -> 0/1 for cv2 resize
            (frame.shape[1], frame.shape[0]),  # width, height
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)  # convert back to bool
        final_mask_bin = mask_resized

        mask_np = mask_resized.astype(np.uint8)  # for findContours

        # find contours
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            if area >= 20000:
                if area > best_area:
                    best_area = area
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                    final_mask_bin = mask_resized  # your boolean mask

        # print("mask_bin type:", type(mask_resized))
        # print("mask_bin shape:", mask_resized.shape)
        # print("mask_bin dtype:", mask_resized.dtype)
        # print("mask_bin :",
        #     mask_resized)
        #
        # print("approx_poly type:", type(approx_poly))
        # if approx_poly is not None:
        #     print("approx_poly shape:", approx_poly.shape)
        #     print("approx_poly dtype:", approx_poly.dtype)
        #     print("approx_poly values:\n", approx_poly.reshape(-1, 2))

        if approx_poly is not None and final_mask_bin is not None:
            # ---- Compute centroid using NumPy ----
            ys, xs = np.where(final_mask_bin)  # use the filtered mask
            if len(ys) > 0:
                y_center = int(ys.mean())
                x_center = int(xs.mean())

                # ---------------- Smoothed Center ----------------
                if self.smoothed_center is None:
                    self.smoothed_center = (x_center, y_center)
                else:
                    alpha = 0.75  # smoothing factor (0 < alpha <= 1)
                    sm_x = int(self.smoothed_center[0] * (1 - alpha) + x_center * alpha)
                    sm_y = int(self.smoothed_center[1] * (1 - alpha) + y_center * alpha)
                    self.smoothed_center = (sm_x, sm_y)

                cX, cY = self.smoothed_center

                # --- DRAW ---
                cv2.drawContours(frame_draw, [approx_poly], 0, (0, 255, 0), 2)
                cv2.circle(frame, (cX, cY), 8, (0, 255, 0), -1)
                # cv2.putText(frame, f"Center: ({cX}, {cY})",
                #             (cX + 10, cY - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            def order_corners_clockwise(approx_poly):
                pts = approx_poly.reshape(-1, 2).astype(np.float32)
                center = pts.mean(axis=0)
                angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
                pts = pts[np.argsort(angles)]
                start = np.argmin(pts.sum(axis=1))
                pts = np.roll(pts, -start, axis=0)
                return pts
            # draw edges if polygon has 4 points
            if len(approx_poly) == 4:
                pts = order_corners_clockwise(approx_poly)
                top_left, top_right, bottom_right, bottom_left = pts

                left_len = np.linalg.norm(top_left - bottom_left)
                right_len = np.linalg.norm(top_right - bottom_right)
                top_len = np.linalg.norm(top_left - top_right)
                bottom_len = np.linalg.norm(bottom_left - bottom_right)
                avg_side = max((left_len + right_len) * 0.5, 1.0)
                yaw_error = (right_len - left_len) / avg_side
                # print("yaw_error", yaw_error)
                if not hasattr(self, "smoothed_yaw_error"):
                    self.smoothed_yaw_error = yaw_error
                else:
                    self.smoothed_yaw_error = 0.6 * self.smoothed_yaw_error + 0.4 * yaw_error  # EMA smoothing

                yaw_error = self.smoothed_yaw_error

                # draw edges
                cv2.line(frame, tuple(top_left.astype(int)), tuple(bottom_left.astype(int)), (0, 0, 255), 4)
                cv2.line(frame, tuple(top_right.astype(int)), tuple(bottom_right.astype(int)), (0, 0, 255), 4)
                # cv2.line(frame, tuple(top_left.astype(int)), tuple(top_right.astype(int)), (255, 255, 0), 4)
                # cv2.line(frame, tuple(bottom_left.astype(int)), tuple(bottom_right.astype(int)), (255, 255, 0), 4)
                print(f"yaw error is : {yaw_error}")
            else:
                yaw_error = 0

            # overlay mask
            overlay = frame.copy()
            overlay[final_mask_bin] = (0, 0, 255)
            frame_draw = cv2.addWeighted(frame_draw, 1.0, overlay, 0.8, 0)

            cX, cY = self.smoothed_center
            error_x = cX - frame_center[0]
            error_y = cY - frame_center[1]

            self.x_diff = error_x - self.last_x_error
            self.last_x_error = error_x
            if len(approx_poly) == 4:
                mean_top_bottom = (top_len + bottom_len) / 2
                if abs(self.x_diff) > mean_top_bottom /6 :
                    x_huge = True
                else:
                    x_huge = False
                print("[INFO] x_diff is: {:.2f}".format(x_huge))
                print("x_diff :" , self.x_diff)

            cv2.circle(frame_draw, (cX, cY), 8, (0, 255, 0), -1)
            CENTER_THRESHOLD_X = 6
            CENTER_THRESHOLD_Y = 10
            YAW_THRESHOLD = 0.05
            centered_horizontally = abs(error_x) < CENTER_THRESHOLD_X
            centered_vertically = abs(error_y) < CENTER_THRESHOLD_Y
            yaw_center = yaw_error < YAW_THRESHOLD

            if centered_horizontally and centered_vertically and yaw_center and not self.forward_done and not x_huge:
                self.forward_done = True
                tello.send_rc_control(0, 0, 0, 0)
                print("area: ", best_area)
                time.sleep(0.3)
                if best_area > 200000:
                    forward_time = 6.5
                elif best_area > 180000:
                    forward_time = 8.5
                elif best_area > 160000:
                    forward_time = 10.5
                elif best_area > 140000:
                    forward_time = 12.25
                elif best_area > 120000:
                    forward_time = 14.5
                elif best_area > 100000:
                    forward_time = 16.5
                elif best_area > 70000:
                    forward_time = 17.5
                else:
                    forward_time = 8
                print("forward time is :", forward_time)
                pid_yaw.integral = pid_yaw.prev_error = 0
                pid_roll.integral = pid_roll.prev_error = 0
                pid_throttle.integral = pid_throttle.prev_error = 0
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                tello.send_rc_control(0, 20, 0, 0)
                print("forward time" , forward_time)
                start_time = time.time()

                while time.time() - start_time < forward_time:
                    # Keep GUI responsive and check for 'q'
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or keyboard.is_pressed('q'):
                        print("[INFO] Emergency land triggered by user.")
                        tello.send_rc_control(0, 0, 0, 0)
                        tello.land()
                        break

                    tello.send_rc_control(0, 20, 0, 0)
                    time.sleep(0.05)

                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(1)
                self.forward_done = False
                self.searching = True
                return frame_draw
            ### all numbers can be tuned for yaw,throttle and roll speeds
            yaw_speed = int(np.clip(pid_yaw.update(yaw_error * 600), -40, 40)) if abs(yaw_error) > 0.006 else 0
            print("yaw_speed:", yaw_speed)
            print("\n")
            print("yaw_error:", yaw_error,"\n")

            throttle = int(np.clip(pid_throttle.update(-error_y), -30, 30)) if abs(error_y) > 7 else 0
            print("error_y" , error_y)


            print("error_x" , error_x)
            if abs(error_x) > 7:
                roll_output = pid_roll.update(error_x)
                roll_speed = int(np.clip(roll_output, -40, 40))
            else:
                roll_speed = 0
            # ---- Compute roll (horizontal alignment) ----
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                error_x = cX - frame_center_x
                error_y = cY - frame_center_y

                # ---- Roll PID ----



            # roll_speed = int(np.clip(pid_roll.update(-error_x), -20, 20)) if abs(error_x) > 20 else 0
            forward_backward_speed = 0
            if abs(yaw_speed) + abs(roll_speed) > 70:
                yaw_speed = int(yaw_speed * 0.7)
                roll_speed = int(roll_speed * 0.7)
                # ---- Forward/backward compensation ----
            forward_comp = int(0.10 * abs(yaw_speed) + 0.20 * abs(roll_speed))  ###Tune these two numbers
            forward_backward_speed = -forward_comp if (abs(yaw_speed) > 2 and abs(roll_speed) > 2) else 0
            # --- Damp roll while yaw is not aligned ---
            yaw_alignment_factor = max(0.0, 1.0 - (abs(yaw_error) * 9))  # tune 6–10
            if roll_speed > 10:
                roll_speed = 10
            elif roll_speed < -10:
                roll_speed = -10
            roll_speed = int(roll_speed * yaw_alignment_factor)
            print(f"roll_speed: {roll_speed}")
            # error_ori = 0
            # same_sgn = False
            # if roll_speed > 0 and yaw_error > 0:
            #     same_sgn = True
            # if yaw_error > 0.13 and same_sgn:
            #     error_ori = 1
            # elif yaw_error < -0.13 and same_sgn:
            #     error_ori = -1
            #
            # if error_ori == 1:
            #     tello.send_rc_control(-5, 0, 0,5)
            # elif error_ori == -1:
            #     tello.send_rc_control(+5, 0, 0,-5)
            # else:
            #     tello.send_rc_control(roll_speed, forward_backward_speed, throttle, yaw_speed)

            tello.send_rc_control(roll_speed, forward_backward_speed, throttle, yaw_speed)

            try:
                print(roll_speed, forward_backward_speed, throttle, yaw_speed)
            except:
                print("_")
            # predicted = self.kf.predict()
            # predicted = (int(predicted[0]), int(predicted[1]))
            # cv2.circle(frame_draw, predicted, 5, (255, 0, 0), 4)
        else:
            if self.no_detection_start_time is None:
                self.no_detection_start_time = time.time()
            elif time.time() - self.no_detection_start_time > 4:
                self.searching = True
            if self.searching:
                self.perform_search(tello)
            else:
                tello.send_rc_control(0, 0, 0, 0)

        return frame_draw


# ---------------- Drone Controller ----------------
class DroneController:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        # self.tello.takeoff()
        print("[INFO] Battery:", self.tello.get_battery())
        self.tello.send_rc_control(0, 0, 30, 0)
        time.sleep(7)
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        self.pid_yaw = PID(Kp=0.2, Ki=0.01, Kd=0.05)
        self.pid_throttle = PID(Kp=0.35, Ki=0.01, Kd=0.18)
        self.pid_roll = PID(Kp=0.24, Ki=0.002, Kd=0.03)
        self.pid_forward = PID(Kp=0.002, Ki=0.0001, Kd=0.005)
        self.detector = WindowDetector()

    def run(self):
        try:

            while True:
                frame = self.tello.get_frame_read().frame
                if keyboard.is_pressed('e'):
                    Tello.emergency()
                    break
                output = self.detector.process(frame, self.pid_yaw, self.tello,
                                               self.pid_throttle, self.pid_roll, self.pid_forward)
                cv2.imshow("Window Detection", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.tello.send_rc_control(0, 0, 0, 0)
                    self.tello.land()
                    break
        except Exception as e:
            print(f"[ERROR] {e}")
            self.tello.send_rc_control(0, 0, 0, 0)
            self.tello.land()
            cv2.destroyAllWindows()
            self.tello.streamoff()
        finally:
            cv2.destroyAllWindows()
            self.tello.streamoff()

# ---------------- Main ----------------
if __name__ == "__main__":
    controller = DroneController()
    controller.run()