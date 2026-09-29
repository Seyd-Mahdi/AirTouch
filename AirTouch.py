"""
Vision-Core: Professional Computer Vision System with Advanced Gesture Control
FINAL EDITION - High-Performance Multithreaded Architecture

FEATURES:
- Multithreaded Camera Acquisition (Eliminates Lag)
- Continuous Drag & Drawing (Paint Mode)
- Relative Distance Scaling (25% of Hand Size)
- Apple Vision Pro Style Scrolling (Y-Velocity Tracking)
- Hand Landmark Visualization (MediaPipe Drawing)
- Full System Tray Integration
- Ultra-High Performance Optimization
"""

import cv2
import mediapipe as mp
import pyautogui
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
from collections import deque
import threading
from PIL import Image, ImageDraw
import pystray
from queue import Queue


# ==================== PERFORMANCE OPTIMIZATION ====================
# Apple Vision Pro style responsiveness
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
# Disable keyboard interrupt delay
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


class CameraThread(threading.Thread):
    """
    Dedicated thread for camera frame acquisition.
    Eliminates blocking I/O from main processing loop.
    """

    def __init__(self, camera_id=0, frame_width=1280, frame_height=720, fps=30):
        """
        Initialize camera thread.

        Args:
            camera_id (int): Camera device ID
            frame_width (int): Frame width
            frame_height (int): Frame height
            fps (int): Target FPS
        """
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.frame_queue = Queue(maxsize=2)  # Only keep latest frames
        self.running = True

        
    def run(self):
        """Continuously capture frames and place in queue."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Clear queue to always get the latest frame (non-blocking)
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                self.frame_queue.put((ret, frame))

    def get_frame(self):
        """
        Get latest frame from queue (non-blocking).

        Returns:
            tuple: (success: bool, frame: numpy.ndarray)
        """
        try:
            return self.frame_queue.get_nowait()
        except:
            return None, None

    def stop(self):
        """Stop camera thread and release resources."""
        self.running = False
        self.cap.release()


class VisionCore:
    """
    Professional Vision-Core System - Final Edition
    High-performance multithreaded gesture control with painting support.
    """

    def __init__(self, frame_width=640, frame_height=480):
        """
        Initialize Vision-Core system.

        Args:
            frame_width (int): Webcam frame width
            frame_height (int): Webcam frame height
        """
        # ==================== Camera Setup (Multithreaded) ====================
        self.camera_thread = CameraThread(0, frame_width, frame_height, 30)
        self.camera_thread.start()
        self.frame_width = frame_width
        self.frame_height = frame_height

        # ==================== MediaPipe Initialization ====================
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face = mp.solutions.face_detection

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8
        )

        self.face_detection = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3
        )

        # ==================== Screen & Display Configuration ====================
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

        # ==================== Gesture Control Settings ====================
        # Mouse smoothing buffer
        self.mouse_smoothing_buffer = deque(maxlen=5)

        # Drag state tracking (NEW: Continuous drag support)
        self.left_drag_active = False
        self.right_click_active = False
        self.left_drag_start_time = 0
        self.right_click_start_time = 0
        self.drag_cooldown = 0.05  # 50ms minimum

        # Adaptive pinch thresholding (UPDATED: 25% of hand size)
        self.pinch_threshold_percentage = 0.25  # 25% (changed from 15%)
        self.adaptive_pinch_threshold = 30
        self.reference_scale = 0  # Wrist to Index Base

        # Scroll state with Y-velocity tracking (UPDATED: Apple Vision Pro style)
        self.scroll_active = False
        self.scroll_last_y = 0
        self.scroll_last_time = time.time()
        self.scroll_velocity = 0
        self.scroll_history = deque(maxlen=5)  # Track last 5 movements for smoothing
        self.scroll_dead_zone = 2  # Pixels: Ignore tiny movements to prevent shaking
        self.scroll_velocity_multiplier = 50  # Linear multiplier for scroll speed (UPDATED)
        self.scroll_min_activation_threshold = 15  # Tight threshold for scroll activation (UPDATED)
        self.scroll_max_cooldown = 0.016  # ~60fps (16ms between scroll events)

        # ==================== Face Detection Timeout (Security Lock) ====================
        self.face_absent_frames = 0
        self.face_timeout_frames = 300  # ~5 seconds at 30 FPS
        self.lock_triggered = False

        # ==================== Health Guard Settings ====================
        frame_area = frame_width * frame_height
        self.proximity_threshold = 0.15 * frame_area
        self.last_face_y = 0

        # ==================== Spine Guard Settings ====================
        self.spine_guard_frames = 0
        self.spine_guard_threshold = 200
        self.face_y_threshold = 0.4

        # ==================== Hand Tracking (MAJOR UPDATE: Palm Center Tracking) ====================
        # Palm Center Tracking (Landmark 9 - Middle Finger MCP) - STABLE TRACKING POINT
        # The palm center is independent of finger positions - gestures don't affect cursor
        self.palm_x = 0
        self.palm_y = 0
        self.palm_smoothing_buffer = deque(maxlen=10)  # Heavy smoothing for palm (10 frames)
        self.palm_alpha = 0.1  # Heavy low-pass filter (0.1 = very smooth, ultra-stable)

        # Finger tracking for gesture detection only (doesn't affect cursor)
        self.weighted_x = 0
        self.weighted_y = 0
        self.alpha = 0.2  # Reference smoothing (not used for cursor)
        self.hand_depth = 0.5
        self.precision_mode = False

        # ==================== Background Execution ====================
        self.paused = False
        self.running = True
        self.tray_icon = None

        # ==================== FPS Monitoring ====================
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

        # ==================== Auto-Calibration ====================
        self.calibration_complete = False

    # ==================== FPS Calculation ====================
    def get_fps(self):
        """Calculate frames per second."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

    # ==================== Auto-Calibration (Hand Size Based) ====================
    def auto_calibrate_hand_size(self, hand_landmarks, frame_width, frame_height, num_samples=30):
        """Auto-calibrate thresholds based on detected hand size."""
        if not hasattr(self, 'calibration_samples'):
            self.calibration_samples = []

        reference_scale = self.calculate_reference_scale(hand_landmarks, frame_width, frame_height)
        self.calibration_samples.append(reference_scale)

        if len(self.calibration_samples) >= num_samples:
            avg_hand_size = np.mean(self.calibration_samples)
            self.adaptive_pinch_threshold = max(15, int(avg_hand_size * 0.25))
            self.scroll_min_activation_threshold = max(12, int(avg_hand_size * 0.12))
            self.calibration_complete = True
            print(f"✓ Calibration Complete! Hand: {avg_hand_size:.0f}px | Pinch: {self.adaptive_pinch_threshold:.0f}px | Scroll: {self.scroll_min_activation_threshold:.0f}px")
            return True
        return False

    # ==================== Mouse Movement Smoothing ====================
    def smooth_mouse_position(self, x, y):
        """Smooth mouse movement to reduce jitter."""
        self.mouse_smoothing_buffer.append((x, y))
        avg_x = np.mean([pos[0] for pos in self.mouse_smoothing_buffer])
        avg_y = np.mean([pos[1] for pos in self.mouse_smoothing_buffer])
        return int(avg_x), int(avg_y)

    # ==================== Kalman Smoothing ====================
    def apply_kalman_smoothing(self, x, y):
        """Apply exponential smoothing for ultra-precision tracking."""
        if self.weighted_x == 0:
            self.weighted_x = x
            self.weighted_y = y

        self.weighted_x = self.alpha * x + (1 - self.alpha) * self.weighted_x
        self.weighted_y = self.alpha * y + (1 - self.alpha) * self.weighted_y

        return int(self.weighted_x), int(self.weighted_y)

    # ==================== Dynamic Scaling ====================
    def apply_dynamic_scaling(self, screen_x, screen_y, hand_z):
        """Apply precision scaling based on hand depth."""
        self.hand_depth = hand_z

        precision_factor = 0.3 + (0.7 * (1 - hand_z))

        current_x, current_y = pyautogui.position()
        screen_x = int(current_x + (screen_x - current_x) * precision_factor)
        screen_y = int(current_y + (screen_y - current_y) * precision_factor)

        self.precision_mode = hand_z > 0.6

        return screen_x, screen_y

    # ==================== Palm Center Calculation (Landmark 9 - Stable Tracking) ====================
    def calculate_palm_center(self, hand_landmarks, frame_width, frame_height):
        """
        Calculate palm center position from Landmark 9 (Middle Finger MCP).
        This is the most stable point on the hand, independent of finger movements.

        Landmark 9: Middle Finger MCP (Metacarpophalangeal Joint)
        - Closest to actual palm center
        - Doesn't move when fingers bend/pinch
        - Provides jitter-free cursor tracking

        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_width (int): Frame width
            frame_height (int): Frame height

        Returns:
            tuple: (palm_x, palm_y) - screen coordinates
        """
        # Get Landmark 9 (Middle Finger MCP - Palm Center)
        palm_landmark = hand_landmarks.landmark[9]

        # Convert to frame coordinates
        palm_x_normalized = palm_landmark.x
        palm_y_normalized = palm_landmark.y

        # Apply margin-based screen mapping (same as before)
        margin = 0.15
        mapped_x = (palm_x_normalized - margin) / (1 - 2 * margin)
        mapped_y = (palm_y_normalized - margin) / (1 - 2 * margin)

        # Clamp to valid range
        mapped_x = max(0, min(mapped_x, 1))
        mapped_y = max(0, min(mapped_y, 1))

        # Convert to screen coordinates (720p full-screen coverage)
        screen_x = int(mapped_x * self.screen_width)
        screen_y = int(mapped_y * self.screen_height)

        return screen_x, screen_y

    # ==================== Heavy Low-Pass Filter (Palm Smoothing) ====================
    def apply_heavy_palm_smoothing(self, palm_x, palm_y):
        """
        Apply heavy low-pass filter to palm position for ultra-stable cursor.
        Uses exponential smoothing with alpha=0.1 (very smooth, no jitter).

        Args:
            palm_x (int): Raw palm X coordinate
            palm_y (int): Raw palm Y coordinate

        Returns:
            tuple: (smoothed_x, smoothed_y)
        """
        # Initialize on first call
        if self.palm_x == 0 and self.palm_y == 0:
            self.palm_x = palm_x
            self.palm_y = palm_y

        # Apply exponential smoothing: new_value = alpha * current + (1 - alpha) * previous
        # With alpha=0.1: 90% previous + 10% current = extremely smooth, stable tracking
        self.palm_x = self.palm_alpha * palm_x + (1 - self.palm_alpha) * self.palm_x
        self.palm_y = self.palm_alpha * palm_y + (1 - self.palm_alpha) * self.palm_y

        return int(self.palm_x), int(self.palm_y)

    # ==================== Hand Size Calculation (Reference Scale) ====================
    def calculate_reference_scale(self, hand_landmarks, frame_width, frame_height):
        """
        Calculate reference scale (Wrist to Index Base distance).
        Used for relative distance scaling.
        """
        wrist = hand_landmarks.landmark[0]
        index_base = hand_landmarks.landmark[5]

        wrist_x = int(wrist.x * frame_width)
        wrist_y = int(wrist.y * frame_height)
        index_base_x = int(index_base.x * frame_width)
        index_base_y = int(index_base.y * frame_height)

        reference_scale = np.sqrt((index_base_x - wrist_x)**2 + (index_base_y - wrist_y)**2)
        return max(20, reference_scale)  # Minimum scale threshold

    # ==================== Pinch Detection with Adaptive Threshold ====================
    def detect_pinch(self, landmark1, landmark2, frame_width, frame_height, reference_scale, pinch_type="normal"):
        """
        Detect pinch using relative distance scaling.

        Args:
            landmark1: First finger landmark
            landmark2: Second finger landmark
            frame_width (int): Frame width
            frame_height (int): Frame height
            reference_scale (float): Hand size reference
            pinch_type (str): "normal" for clicks (25%), "scroll" for tight grip (tight threshold)

        Returns:
            tuple: (is_pinched: bool, distance: float)
        """
        p1_x = int(landmark1.x * frame_width)
        p1_y = int(landmark1.y * frame_height)
        p2_x = int(landmark2.x * frame_width)
        p2_y = int(landmark2.y * frame_height)

        distance = np.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)

        # UPDATED: Different thresholds for different gestures
        if pinch_type == "scroll":
            # Scroll: Tighter threshold for "tightly pressed together"
            threshold = self.scroll_min_activation_threshold  # 15px tight grip
        else:
            # Normal clicks: 25% of reference scale
            threshold = max(15, reference_scale * self.pinch_threshold_percentage)

        self.adaptive_pinch_threshold = threshold
        is_pinched = distance < threshold
        return is_pinched, distance

    # ==================== Draw Hand Landmarks & Gesture Feedback ====================
    def draw_hand_landmarks(self, frame, hand_landmarks, gesture_state):
        """
        Draw hand skeleton and fingertip circles with gesture feedback.
        Includes yellow connection line for scroll mode and PALM CENTER indicator.
        """
        h, w, c = frame.shape

        # Draw hand skeleton (MediaPipe drawing)
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

        # ==================== PALM CENTER INDICATOR (Landmark 9 - Cursor Control Point) ====================
        # Draw large cross-hair on palm center to show where cursor is tracking
        palm_landmark = hand_landmarks.landmark[9]
        palm_x = int(palm_landmark.x * w)
        palm_y = int(palm_landmark.y * h)

        # Draw cross-hair marker (cyan/light blue for visibility)
        marker_size = 15
        marker_thickness = 2
        marker_color = (255, 255, 0)  # Cyan (stable tracking point)

        # Vertical line
        cv2.line(frame, (palm_x, palm_y - marker_size), (palm_x, palm_y + marker_size), marker_color, marker_thickness)
        # Horizontal line
        cv2.line(frame, (palm_x - marker_size, palm_y), (palm_x + marker_size, palm_y), marker_color, marker_thickness)
        # Circle around palm center
        cv2.circle(frame, (palm_x, palm_y), marker_size, marker_color, 2)

        # Label
        cv2.putText(frame, "PALM", (palm_x + 20, palm_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 1)

        # ==================== YELLOW SCROLL LINE (Visual Distinction) ====================
        # When scroll mode is active, draw yellow line between Index & Middle
        if gesture_state.get("scroll_active"):
            index_landmark = hand_landmarks.landmark[8]
            middle_landmark = hand_landmarks.landmark[12]

            index_x = int(index_landmark.x * w)
            index_y = int(index_landmark.y * h)
            middle_x = int(middle_landmark.x * w)
            middle_y = int(middle_landmark.y * h)

            # Draw thick yellow line to indicate scroll mode
            cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (0, 255, 255), 4)

        # Draw enhanced feedback circles on key fingertips
        landmarks_feedback = {
            4: "Thumb",
            8: "Index",
            12: "Middle"
        }

        for landmark_idx, name in landmarks_feedback.items():
            landmark = hand_landmarks.landmark[landmark_idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            # Determine circle color based on active gestures
            if landmark_idx == 4:  # Thumb
                if gesture_state.get("left_drag_active"):
                    circle_color = (0, 255, 0)  # Green
                    radius, thickness = 12, 3
                elif gesture_state.get("right_click_active"):
                    circle_color = (255, 0, 0)  # Blue
                    radius, thickness = 12, 3
                else:
                    circle_color = (200, 50, 50)
                    radius, thickness = 8, 2
            elif landmark_idx == 8:  # Index
                if gesture_state.get("left_drag_active"):
                    circle_color = (0, 255, 0)  # Green
                    radius, thickness = 12, 3
                elif gesture_state.get("scroll_active"):
                    circle_color = (0, 255, 255)  # Yellow
                    radius, thickness = 12, 3
                else:
                    circle_color = (50, 200, 50)
                    radius, thickness = 8, 2
            elif landmark_idx == 12:  # Middle
                if gesture_state.get("right_click_active"):
                    circle_color = (255, 0, 0)  # Blue
                    radius, thickness = 12, 3
                elif gesture_state.get("scroll_active"):
                    circle_color = (0, 255, 255)  # Yellow
                    radius, thickness = 12, 3
                else:
                    circle_color = (50, 50, 200)
                    radius, thickness = 8, 2

            cv2.circle(frame, (x, y), radius, circle_color, thickness)

    # ==================== Hand Gesture Processing (FINAL VERSION) ====================
    def process_hand_gestures(self, frame, hand_landmarks):
        """
        Process all hand gestures with continuous drag support.

        GESTURES:
        - LEFT DRAG: Index + Thumb pinch (continuous drag with mouseDown/mouseUp)
        - RIGHT CLICK: Middle + Thumb pinch (single right-click event)
        - SCROLL: Index + Middle touching (Y-velocity tracking - Apple Vision Pro style)
        - MOUSE: Index finger controls cursor position

        Returns:
            dict: Gesture state with all active gestures and states
        """
        if hand_landmarks is None:
            return {
                "left_drag_active": False,
                "right_click_active": False,
                "scroll_active": False,
                "left_distance": 0,
                "right_distance": 0,
                "scroll_distance": 0
            }

        h, w, c = frame.shape

        # ==================== AUTO-CALIBRATION (First 30 frames) ====================
        if not self.calibration_complete:
            self.auto_calibrate_hand_size(hand_landmarks, w, h)

        # ==================== Extract Key Landmarks ====================
        thumb_tip = hand_landmarks.landmark[4]
        index_base = hand_landmarks.landmark[5]
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]

        # ==================== Calculate Reference Scale (Hand Size) ====================
        reference_scale = self.calculate_reference_scale(hand_landmarks, w, h)
        self.reference_scale = reference_scale

        # ==================== MAJOR UPDATE: Mouse Cursor Control (PALM CENTER - Landmark 9) ====================
        # Calculate palm center position (Landmark 9 - Middle Finger MCP)
        # This is INDEPENDENT of finger movements - gestures don't affect cursor position
        screen_x, screen_y = self.calculate_palm_center(hand_landmarks, w, h)

        # Apply HEAVY low-pass filter for ultra-stable, jitter-free tracking
        # Palm smoothing is much stronger than finger tracking (alpha=0.1 vs 0.2)
        screen_x, screen_y = self.apply_heavy_palm_smoothing(screen_x, screen_y)

        # Move mouse cursor to palm-based position (independent of gestures)
        pyautogui.moveTo(screen_x, screen_y)

        # ==================== Detect All Pinch Gestures ====================
        left_pinched, left_distance = self.detect_pinch(
            index_finger_tip, thumb_tip, w, h, reference_scale, pinch_type="normal"
        )

        right_pinched, right_distance = self.detect_pinch(
            middle_finger_tip, thumb_tip, w, h, reference_scale, pinch_type="normal"
        )

        # UPDATED: Scroll uses tighter threshold for "tightly pressed together"
        scroll_pinched, scroll_distance = self.detect_pinch(
            index_finger_tip, middle_finger_tip, w, h, reference_scale, pinch_type="scroll"
        )

        # ==================== LEFT DRAG (Continuous Drawing - Palm Locked) ====================
        if left_pinched and not right_pinched and not scroll_pinched:
            if not self.left_drag_active:
                pyautogui.mouseDown()
                self.left_drag_active = True
        else:
            if self.left_drag_active:
                pyautogui.mouseUp()
                self.left_drag_active = False

        # ==================== RIGHT CLICK ====================
        if right_pinched and not left_pinched and not scroll_pinched:
            current_time = time.time()
            if not self.right_click_active:
                pyautogui.rightClick()
                self.right_click_active = True
                self.right_click_start_time = current_time
            else:
                # Debounce: Only one right-click per 500ms
                if current_time - self.right_click_start_time > 0.5:
                    self.right_click_active = False
        else:
            self.right_click_active = False

        # ==================== SCROLL (Palm Center Velocity-Based - dy * -100) ====================
        if scroll_pinched:
            self.scroll_active = True
            palm_landmark = hand_landmarks.landmark[9]
            palm_y_current = int(palm_landmark.y * h)

            if self.scroll_last_y == 0:
                self.scroll_last_y = palm_y_current
            else:
                # Velocity-based: dy * -100 (normalized)
                dy = (self.scroll_last_y - palm_y_current) / h
                self.scroll_velocity = dy

                # Dead-zone check (2px at 720p)
                if abs(dy) > 0.003:
                    self.scroll_history.append(dy)
                    smoothed_dy = np.mean(list(self.scroll_history))

                    current_time = time.time()
                    if current_time - self.scroll_last_time > self.scroll_max_cooldown:
                        scroll_amount = int(smoothed_dy * -1000)
                        if scroll_amount != 0:
                            pyautogui.scroll(scroll_amount)
                            self.scroll_last_time = current_time

                self.scroll_last_y = palm_y_current
        else:
            self.scroll_active = False
            self.scroll_last_y = 0
            self.scroll_history.clear()
            self.scroll_velocity = 0

        # ==================== Draw Hand Landmarks with Feedback ====================
        gesture_state = {
            "left_drag_active": self.left_drag_active,
            "right_click_active": self.right_click_active,
            "scroll_active": self.scroll_active,
            "left_distance": left_distance,
            "right_distance": right_distance,
            "scroll_distance": scroll_distance
        }

        self.draw_hand_landmarks(frame, hand_landmarks, gesture_state)

        return gesture_state

    # ==================== Face Detection & Health Guard ====================
    def process_face_detection(self, frame, face_results):
        """Process face detection for health monitoring and security."""
        h, w, c = frame.shape

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, bbox_w, bbox_h = (
                    int(bboxC.xmin * w),
                    int(bboxC.ymin * h),
                    int(bboxC.width * w),
                    int(bboxC.height * h)
                )

                cv2.rectangle(frame, (x, y), (x + bbox_w, y + bbox_h), (0, 255, 0), 2)

                # Enhanced Proximity Sensor
                face_area = bbox_w * bbox_h
                if face_area > self.proximity_threshold:
                    cv2.putText(
                        frame,
                        "TOO CLOSE",
                        (w // 2 - 100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3
                    )

                # Spine Guard
                face_center_y = (y + bbox_h // 2) / h
                self.last_face_y = face_center_y

                if face_center_y > self.face_y_threshold:
                    self.spine_guard_frames += 1
                else:
                    self.spine_guard_frames = 0

                if self.spine_guard_frames > self.spine_guard_threshold:
                    cv2.putText(
                        frame,
                        "STAY UPRIGHT: Spine Health Warning",
                        (w // 2 - 250, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 165, 255),
                        3
                    )

            self.face_absent_frames = 0
        else:
            self.face_absent_frames += 1
            self.spine_guard_frames = 0

    # ==================== Security Lock ====================
    def check_security_lock(self):
        """Trigger Windows lock if face absent for 5 seconds."""
        if self.face_absent_frames >= self.face_timeout_frames and not self.lock_triggered:
            self.lock_triggered = True
            os.system("rundll32.exe user32.dll,LockWorkStation")

        if self.face_absent_frames == 0:
            self.lock_triggered = False

    # ==================== UI Rendering ====================
    def draw_ui(self, frame, fps, gesture_state=None):
        """Draw comprehensive UI with gesture and system feedback."""
        h, w, c = frame.shape

        # FPS Display
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Face Timeout Indicator
        face_timer_text = f"Face: {self.face_absent_frames}/{self.face_timeout_frames}"
        timer_color = (0, 0, 255) if self.face_absent_frames > self.face_timeout_frames * 0.5 else (0, 255, 255)
        cv2.putText(
            frame,
            face_timer_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            timer_color,
            2
        )

        # Advanced Gesture Feedback
        if gesture_state:
            # Left Drag Status
            left_status = "DRAGGING" if gesture_state.get("left_drag_active") else "Ready"
            left_color = (0, 255, 0) if gesture_state.get("left_drag_active") else (100, 100, 100)
            cv2.putText(
                frame,
                f"Drag: {gesture_state.get('left_distance', 0):.0f}px [{left_status}]",
                (10, h - 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                left_color,
                2
            )

            # Right Click Status
            right_status = "R-CLICK" if gesture_state.get("right_click_active") else "Ready"
            right_color = (255, 0, 0) if gesture_state.get("right_click_active") else (100, 100, 100)
            cv2.putText(
                frame,
                f"R-Click: {gesture_state.get('right_distance', 0):.0f}px [{right_status}]",
                (10, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                right_color,
                2
            )

            # Scroll Status (with velocity)
            scroll_status = "SCROLLING ⚡" if gesture_state.get("scroll_active") else "Idle"
            scroll_color = (0, 255, 255) if gesture_state.get("scroll_active") else (100, 100, 100)
            cv2.putText(
                frame,
                f"Scroll: {gesture_state.get('scroll_distance', 0):.0f}px [{scroll_status}]",
                (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                scroll_color,
                2
            )

            # Velocity Multiplier Display (shows scroll responsiveness)
            multiplier_text = f"Velocity: {self.scroll_velocity:.1f}px/frame | Multiplier: x{self.scroll_velocity_multiplier}"
            multiplier_color = (0, 255, 255) if gesture_state.get("scroll_active") else (100, 100, 100)
            cv2.putText(
                frame,
                multiplier_text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                multiplier_color,
                1
            )

            # Reference Scale Display
            cv2.putText(
                frame,
                f"Reference: {self.reference_scale:.1f}px | Threshold: {self.adaptive_pinch_threshold:.1f}px",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 150, 255),
                2
            )

        # Mouse Position
        mouse_x, mouse_y = pyautogui.position()
        cv2.putText(
            frame,
            f"Mouse: ({mouse_x}, {mouse_y})",
            (w - 300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # ==================== PALM CENTER TRACKING INFO ====================
        # Show palm smoothing status and information
        cv2.putText(
            frame,
            f"Palm Tracking: Alpha={self.palm_alpha} (Heavy Smoothing)",
            (w - 300, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # System Status
        status_text = "LOCKED" if self.lock_triggered else "ACTIVE"
        status_color = (0, 0, 255) if self.lock_triggered else (0, 255, 0)
        cv2.putText(
            frame,
            f"Status: {status_text}",
            (w - 300, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        # Precision Mode
        precision_text = "PRECISION" if self.precision_mode else "Normal"
        precision_color = (0, 255, 255) if self.precision_mode else (100, 100, 100)
        cv2.putText(
            frame,
            precision_text,
            (w - 300, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            precision_color,
            2
        )

        # Pause indicator
        if self.paused:
            cv2.putText(
                frame,
                "PAUSED",
                (w // 2 - 60, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3
            )

        return frame

    # ==================== System Tray Integration ====================
    def create_tray_image(self):
        """Create system tray icon."""
        size = (64, 64)
        image = Image.new('RGB', size, color='darkblue')
        draw = ImageDraw.Draw(image)
        draw.text((10, 20), "VC", fill='white')
        return image

    def on_pause(self, icon, item):
        """Toggle pause from system tray."""
        self.paused = not self.paused
        state = "PAUSED" if self.paused else "RUNNING"
        print(f"\n{'='*60}")
        print(f"Vision-Core: {state}")
        print(f"{'='*60}")

    def on_quit(self, icon, item):
        """Quit from system tray."""
        print("\n" + "="*60)
        print("Shutting down Vision-Core...")
        print("="*60)
        self.running = False
        icon.stop()

    def setup_tray_icon(self):
        """Initialize system tray with menu."""
        try:
            menu = pystray.Menu(
                pystray.MenuItem('Pause/Resume', self.on_pause),
                pystray.MenuItem('Quit', self.on_quit)
            )

            icon_image = self.create_tray_image()
            self.tray_icon = pystray.Icon(
                "Vision-Core",
                icon_image,
                "Vision-Core: Professional Gesture Control",
                menu
            )

            tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
            tray_thread.start()
        except Exception as e:
            print(f"Warning: System tray unavailable: {e}")

    # ==================== MAIN EXECUTION LOOP ====================
    def run(self):
        """
        Main execution loop with multithreaded architecture.
        - Camera acquisition on dedicated thread
        - Processing on main thread
        - No blocking I/O
        """
        print("=" * 60)
        print("Vision-Core Professional Edition - STARTING")
        print("=" * 60)
        print("MAJOR UPDATE: PALM CENTER TRACKING")
        print("=" * 60)
        print("ARCHITECTURE:")
        print("  ✓ Multithreaded Camera Acquisition (Zero Lag)")
        print("  ✓ Palm Center Cursor Control (Landmark 9 - Ultra Stable)")
        print("  ✓ Heavy Low-Pass Filter (10-frame smoothing, alpha=0.1)")
        print("  ✓ Independent Gestures (Pinches don't affect cursor)")
        print("  ✓ Continuous Drag & Drawing (Paint Mode)")
        print("  ✓ Relative Distance Scaling (25% Reference)")
        print("  ✓ Apple Vision Pro Scrolling (FIXED & OPTIMIZED)")
        print("  ✓ Hand Landmark Visualization + Palm Indicator")
        print("  ✓ Full System Tray Integration")
        print("=" * 60)
        print("GESTURES (Cursor locked to PALM, not fingers):")
        print("  🎨 LEFT DRAG: Index + Thumb pinch (Green)")
        print("  🖱️  RIGHT CLICK: Middle + Thumb pinch (Blue)")
        print("  📜 SCROLL: Index + Middle tightly pressed (Yellow line)")
        print("=" * 60)
        print("Visual Feedback:")
        print("  🎯 Cyan Cross-Hair: Palm center (cursor tracking point)")
        print("  🟢 Green Circles: Left drag active")
        print("  🔵 Blue Circles: Right click active")
        print("  🟡 Yellow Line: Scroll mode active")
        print("=" * 60)
        print("Controls: System Tray | Keyboard: 'q' to quit")
        print("=" * 60)

        self.setup_tray_icon()

        try:
            while self.running:
                # Get frame from dedicated camera thread (non-blocking)
                ret, frame = self.camera_thread.get_frame()
                if frame is None:
                    continue

                if self.paused:
                    frame_paused = cv2.flip(frame, 1)
                    cv2.putText(
                        frame_paused,
                        "PAUSED - Check system tray to resume",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow("Vision-Core", frame_paused)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                    continue

                # Process frame
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ==================== Hand Gesture Processing ====================
                gesture_state = None
                hand_results = self.hands.process(rgb_frame)
                if hand_results.multi_hand_landmarks and hand_results.multi_hand_landmarks[0]:
                    gesture_state = self.process_hand_gestures(
                        frame,
                        hand_results.multi_hand_landmarks[0]
                    )

                # ==================== Face Detection ====================
                face_results = self.face_detection.process(rgb_frame)
                self.process_face_detection(frame, face_results)

                # ==================== Security Check ====================
                self.check_security_lock()

                # ==================== UI Rendering ====================
                fps = self.get_fps()
                frame = self.draw_ui(frame, fps, gesture_state)

                # ==================== Display ====================
                cv2.imshow("Vision-Core", frame)

                # ==================== Input Handling ====================
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExit signal received...")
                    self.running = False

        except KeyboardInterrupt:
            print("\nKeyboard interrupt...")
            self.running = False
        except Exception as e:
            print(f"Error: {e}")
            self.running = False
        finally:
            self.cleanup()

    # ==================== Cleanup ====================
    def cleanup(self):
        """Release all resources and shutdown gracefully."""
        print("Cleaning up resources...")

        # Release drag if active
        if self.left_drag_active:
            pyautogui.mouseUp()

        # Stop camera thread
        self.camera_thread.stop()

        # Close OpenCV
        cv2.destroyAllWindows()

        # Release MediaPipe
        self.hands.close()
        self.face_detection.close()

        print("=" * 60)
        print("Vision-Core Professional Edition - SHUTDOWN COMPLETE")
        print("=" * 60)


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        vision_core = VisionCore(frame_width=640, frame_height=480)
        vision_core.run()

    except ImportError as e:
        print(f"Error: Missing library - {e}")
        print("\nInstall with:")
        print("  pip install opencv-python mediapipe pyautogui numpy pillow pystray")

    except Exception as e:
        print(f"Fatal error: {e}")
