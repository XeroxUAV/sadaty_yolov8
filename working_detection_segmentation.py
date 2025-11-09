import cv2
import numpy as np
from djitellopy import Tello
import time
from ultralytics import YOLO
import keyboard
import torch
import torch.nn.functional as F

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
    def __init__(self, model_path=r"C:\softwares\yolov8_env64\runs\segment\window_segmentation_model8\weights\best.pt",
                 conf_thresh=0.6):
        self.kf = KalmanFilter()
        self.alpha = 0.2
        self.smoothed_center = None
        self.no_detection_start_time = None
        self.searching = False
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.forward_done = False
        self.search_yaw_speed = 20  # yaw speed for search mode (adjust 20–40)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[INFO] YOLO model loaded.")
        print("loaded on:" , self.device)

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
        if self.forward_done:
            pid_yaw.integral = 0
            pid_yaw.prev_error = 0
            pid_roll.integral = 0
            pid_roll.prev_error = 0
            pid_throttle.integral = 0
            pid_throttle.prev_error = 0
            tello.send_rc_control(0, 2, 0, 0)
            return frame

        frame_h, frame_w = frame.shape[:2]
        VERTICAL_OFFSET = -100
        frame_center = (frame_w // 2, frame_h // 2 + VERTICAL_OFFSET)
        cv2.circle(frame, (frame_center[0], frame_center[1]), 8 , (0, 255, 255), 2)

        raw_frame = frame.copy()
        frame_draw = frame.copy()
        preprocessed = preprocess_frame(raw_frame)
        results = self.model(preprocessed, conf=self.conf_thresh, verbose=False)[0]

        best_area = 0
        approx_poly = None
        final_mask_bin = None

        # ----------- NEW MASK HANDLING ----------------
        if results.masks is not None and len(results.masks) > 0:
            for mask in results.masks.data.to(self.device):
                # interpolate to frame size
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(frame.shape[0], frame.shape[1]),
                    mode="bilinear",
                    align_corners=False
                ).squeeze()
                mask_bin = mask_resized > 0.75
                mask_np = mask_bin.cpu().numpy().astype("uint8")

                # find contours
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area < 20000:
                    continue
                if area > best_area:
                    best_area = area
                    # approximate polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                    final_mask_bin = mask_bin

        if approx_poly is not None and final_mask_bin is not None:
            # ---- Compute smoothed centroid on GPU ----
            ys, xs = torch.where(mask_bin)
            if len(ys) > 0:
                y_center = int(ys.float().mean().item())
                x_center = int(xs.float().mean().item())

                # ---------------- Smoothed Center ----------------
                if self.smoothed_center is None:
                    self.smoothed_center = (x_center, y_center)
                else:
                    alpha = 0.2  # smoothing factor (0 < alpha <= 1)
                    sm_x = int(self.smoothed_center[0] * (1 - alpha) + x_center * alpha)
                    sm_y = int(self.smoothed_center[1] * (1 - alpha) + y_center * alpha)
                    self.smoothed_center = (sm_x, sm_y)

                cX, cY = self.smoothed_center

                # --- DRAW ---
                cv2.drawContours(frame_draw, [approx_poly], 0, (0, 255, 0), 2)
                cv2.circle(frame, (cX, cY), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"Center: ({cX}, {cY})",
                            (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


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
                # top_len = np.linalg.norm(top_left - top_right)
                # bottom_len = np.linalg.norm(bottom_left - bottom_right)
                avg_side = max((left_len + right_len) * 0.5, 1.0)
                yaw_error = (right_len - left_len) / avg_side
                if not hasattr(self, "smoothed_yaw_error"):
                    self.smoothed_yaw_error = yaw_error
                else:
                    self.smoothed_yaw_error = 0.8 * self.smoothed_yaw_error + 0.2 * yaw_error  # EMA smoothing

                yaw_error = self.smoothed_yaw_error

                # draw edges
                cv2.line(frame, tuple(top_left.astype(int)), tuple(bottom_left.astype(int)), (0, 0, 255), 4)
                cv2.line(frame, tuple(top_right.astype(int)), tuple(bottom_right.astype(int)), (0, 0, 255), 4)
                # cv2.line(frame, tuple(top_left.astype(int)), tuple(top_right.astype(int)), (255, 255, 0), 4)
                # cv2.line(frame, tuple(bottom_left.astype(int)), tuple(bottom_right.astype(int)), (255, 255, 0), 4)
                cv2.putText(frame, f"Yaw Err: {yaw_error:.3f}", (600, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                yaw_error = 0

            # overlay mask
            overlay = frame.copy()
            overlay[final_mask_bin.cpu().numpy().astype(bool)] = (0, 0, 255)
            frame_draw = cv2.addWeighted(frame_draw, 1.0, overlay, 0.8, 0)

            # PID and movement logic unchanged
            cX, cY = self.smoothed_center
            error_x = cX - frame_center[0]
            error_y = cY - frame_center[1]
            # cv2.putText(frame, f"X Err: {error_x:.3f}", (600, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # cv2.putText(frame, f"Y Err: {error_y:.3f}", (600, 140),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.circle(frame_draw, (cX, cY), 8, (0, 255, 0), -1)
            CENTER_THRESHOLD_X = 13
            CENTER_THRESHOLD_Y = 10
            YAW_THRESHOLD = 0.05
            centered_horizontally = abs(error_x) < CENTER_THRESHOLD_X
            centered_vertically = abs(error_y) < CENTER_THRESHOLD_Y
            yaw_center = yaw_error < YAW_THRESHOLD

            if centered_horizontally and centered_vertically and yaw_center and not self.forward_done:
                self.forward_done = True
                tello.send_rc_control(0, 0, 0, 0)
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

            yaw_speed = int(np.clip(pid_yaw.update(yaw_error * 300), -40, 40)) if abs(yaw_error) > 0.006 else 0
            print("yaw_speed:", yaw_speed)
            print("yaw_error:", yaw_error)

            throttle = int(np.clip(pid_throttle.update(-error_y), -30, 30)) if abs(error_y) > 10 else 0
            print("error_y" , error_y)


            print("erro_x" , error_x)
            if abs(error_x) > 10:
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
            # --- Anti-forward-drift compensation ---
            forward_comp = int(0.3 * abs(yaw_speed) + 0.35 * abs(roll_speed)) ###Tune these two numbers
            forward_backward_speed = -forward_comp if (abs(yaw_speed) > 0 or abs(roll_speed) > 0) else 0
            # --- Damp roll while yaw is not aligned ---
            yaw_alignment_factor = max(0.0, 1.0 - abs(yaw_error) * 7)  # tune 6–10
            if roll_speed > 10:
                roll_speed = 10
            elif roll_speed < -10:
                roll_speed = -10
            roll_speed = int(roll_speed * yaw_alignment_factor)

            tello.send_rc_control(roll_speed, forward_backward_speed, throttle, yaw_speed)
            try:
                print(roll_speed, forward_backward_speed, throttle, yaw_speed)
            except:
                print("_")
            predicted = self.kf.predict()
            predicted = (int(predicted[0]), int(predicted[1]))
            cv2.circle(frame_draw, predicted, 5, (255, 0, 0), 4)
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
        time.sleep(3)
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        self.pid_yaw = PID(Kp=0.2, Ki=0.01, Kd=0.05)
        self.pid_throttle = PID(Kp=0.35, Ki=0.01, Kd=0.18)
        self.pid_roll = PID(Kp=0.18, Ki=0.002, Kd=0.03)
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
