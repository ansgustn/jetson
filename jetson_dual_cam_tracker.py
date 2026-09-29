#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Python 2로 실행된 경우 자동으로 python3로 전환
if sys.version_info[0] < 3:
    os.execvp("python3", ["python3"] + sys.argv)

"""
jetson_dual_cam_tracker.py
==========================
NVIDIA Jetson Nano(ARM Cortex-A57 4코어, 4GB RAM) 전용
듀얼 웹카메라(2대) 동시 구동 초경량 실시간 손 추적 및 언리얼 엔진(OSC) 스트리머.
"""

# Jetson Nano 최적화 환경 변수 자동 설정 (OpenBLAS Illegal instruction 방지 및 GPU EGL 가속, 로그 소음 억제)
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import math
import platform
import sys
import threading
import time
import cv2
import numpy as np

# OSC 통신 및 트래킹 유틸리티
import mediapipe as mp
from hand_tracker_utils import (
    RobustHandTracker,
    DualCamKnobFuser,
    calc_angle,
    pinch_ratio_landmarks,
    SimpleUDPClient,
)

# ── 기본 설정 상수 ──
DEFAULT_UE_IP = "192.168.137.1"
DEFAULT_UE_PORT = 8000
DEFAULT_OSC_ADDR = "/mediapipe/knob/angle"  # 언리얼 BP_Knob 통합 수신 주소
DEFAULT_OSC_ADDR1 = "/mediapipe/cam1/angle"
DEFAULT_OSC_ADDR2 = "/mediapipe/cam2/angle"
PINCH_WHEN_NO_HAND = 1.5


class ThreadedCamera:
    """
    독립 스레드에서 카메라 프레임을 지속적으로 읽어와 지연 없는 최신 프레임만 제공하는 고성능 캡처 클래스.
    """
    def __init__(self, cam_index, width=640, height=480, fps=30, flip=True):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip

        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

        self._init_camera()

    def _init_camera(self):
        sys_name = platform.system()
        if sys_name == 'Linux':
            # Jetson Nano Linux: V4L2 백엔드 사용
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.cam_index)

            # [중요] 웹캠 2대 동시 연결 시 USB 2.0/3.0 대역폭 초과 방지를 위해 MJPG 강제 지정
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except Exception:
                pass
        elif sys_name == 'Windows':
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.cam_index)
        else:
            self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 카메라 {self.cam_index}를 열 수 없습니다.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 초기 1프레임 워밍업 확인
        ret, frame = self.cap.read()
        if ret and frame is not None:
            if self.flip:
                frame = cv2.flip(frame, 1)
            self.frame = frame
            self.ret = True

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                if self.flip:
                    frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame
                    self.ret = True
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def create_mediapipe_detector(max_num_hands=1):
    """
    Jetson Nano에 최적화된 MediaPipe Hands 검출기를 생성합니다.
    Tasks API(hand_landmarker.task) 시도 후, Jetson 권장 Solutions API(model_complexity=0)로 폴백.
    """
    task_model_path = 'hand_landmarker.task'
    if os.path.exists(task_model_path) and hasattr(mp, 'tasks') and hasattr(mp.tasks, 'vision'):
        try:
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=task_model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            detector = HandLandmarker.create_from_options(options)
            return 'tasks', detector
        except Exception as e:
            pass

    # Solutions Hands API (Jetson Nano aarch64 최고 호환성 & 초경량)
    mp_hands = mp.solutions.hands
    try:
        detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,  # 0: 초경량 모바일 모델
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except TypeError:
        detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return 'solutions', detector


def draw_hand_overlay(frame, pixel_lms, angle, pinch, is_tracked, cam_title, fps_val):
    """HUD 및 감지된 손 랜드마크를 시각화합니다."""
    if is_tracked and pixel_lms is not None:
        pt_wrist = (int(pixel_lms[0][0]), int(pixel_lms[0][1]))
        pt_middle = (int(pixel_lms[9][0]), int(pixel_lms[9][1]))
        pt_thumb = (int(pixel_lms[4][0]), int(pixel_lms[4][1]))
        pt_index = (int(pixel_lms[8][0]), int(pixel_lms[8][1]))
        pt_idx_mcp = (int(pixel_lms[5][0]), int(pixel_lms[5][1]))
        pt_pnk_mcp = (int(pixel_lms[17][0]), int(pixel_lms[17][1]))

        # 종축(손목-중지뿌리, 초록) + 횡축(검지뿌리-새끼뿌리 너클, 하늘색) 합성 회전선
        cv2.line(frame, pt_wrist, pt_middle, (0, 255, 0), 2)
        cv2.line(frame, pt_idx_mcp, pt_pnk_mcp, (255, 255, 0), 2)
        cv2.circle(frame, pt_wrist, 5, (255, 0, 0), -1)
        cv2.circle(frame, pt_middle, 5, (0, 0, 255), -1)

        # 핀치선 (0.70 이하 잡으면 초록, 0.70 초과 놓으면 빨강)
        pinch_color = (0, 255, 0) if pinch <= 0.70 else (0, 0, 255)
        cv2.line(frame, pt_thumb, pt_index, pinch_color, 2)
        cv2.circle(frame, pt_thumb, 4, pinch_color, -1)
        cv2.circle(frame, pt_index, 4, pinch_color, -1)

    # 상단 텍스트 정보 표시
    cv2.putText(frame, f"[{cam_title}] FPS: {fps_val:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    if is_tracked:
        status_text = f"Angle: {angle:.1f} deg | Pinch: {pinch:.2f}"
        status_color = (0, 255, 255)
    else:
        status_text = "Hand LOST (Release)"
        status_color = (0, 0, 255)
    cv2.putText(frame, status_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


def main():
    parser = argparse.ArgumentParser(description="Jetson Nano Ultra-Lightweight Dual-Webcam Hand Tracker & OSC Streamer")
    # 카메라 설정
    parser.add_argument('--cam1', type=int, default=0, help="Camera 1 device index (default: 0)")
    parser.add_argument('--cam2', type=int, default=1, help="Camera 2 device index (default: 1, set to -1 for single-cam)")
    parser.add_argument('--width', type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument('--height', type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument('--infer-width', type=int, default=640, help="Inference width (default: 640)")
    parser.add_argument('--infer-height', type=int, default=480, help="Inference height (default: 480)")
    parser.add_argument('--high-res', '--high_res', dest='high_res', action='store_true', default=True,
                        help="Run MediaPipe at full 640x480 resolution (maximizes precision, default: True)")
    parser.add_argument('--low-res', '--downscale', dest='high_res', action='store_false',
                        help="Downscale to 320x240 (only if extreme FPS needed)")
    # 타겟 손 설정 (두 카메라 모두 기본값 Any로 손 회전 시 라벨 뒤집힘 방지)
    parser.add_argument('--hand', '--target-hand', '--target_hand', type=str, default='Any', choices=['Right', 'Left', 'Auto', 'Any'],
                        help="Target hand for both cameras (default: Any)")
    parser.add_argument('--hand1', type=str, default=None, choices=['Right', 'Left', 'Auto', 'Any'],
                        help="Target hand override for Camera 1 (default: same as --hand)")
    parser.add_argument('--hand2', type=str, default=None, choices=['Right', 'Left', 'Auto', 'Any'],
                        help="Target hand override for Camera 2 (default: same as --hand)")
    parser.add_argument('--max-hands', type=int, default=1, help="Max hands per camera (default: 1 for robust knob pinch)")
    # 연산 부하 절감 옵션
    parser.add_argument('--interleave', action='store_true', default=True,
                        help="Alternate inference between Cam1 and Cam2 each frame (default: True, optimal for dual cam)")
    parser.add_argument('--no-interleave', dest='interleave', action='store_false',
                        help="Infer both cameras every single frame")
    parser.add_argument('--headless', action='store_true', help="Headless mode (no OpenCV GUI window, max FPS)")
    parser.add_argument('--fps', type=int, default=30, help="Target loop FPS cap (default: 30)")
    # OSC 및 융합(Fusion) 설정
    parser.add_argument('--ip', type=str, default=DEFAULT_UE_IP, help=f"Unreal Engine IP (default: {DEFAULT_UE_IP})")
    parser.add_argument('--port', type=int, default=DEFAULT_UE_PORT, help=f"Unreal Engine Port (default: {DEFAULT_UE_PORT})")
    parser.add_argument('--addr', '--address', type=str, default=DEFAULT_OSC_ADDR,
                        help=f"Unified OSC address for fused stream (default: {DEFAULT_OSC_ADDR})")
    parser.add_argument('--addr1', type=str, default=DEFAULT_OSC_ADDR1, help=f"OSC Addr Cam1 (default: {DEFAULT_OSC_ADDR1})")
    parser.add_argument('--addr2', type=str, default=DEFAULT_OSC_ADDR2, help=f"OSC Addr Cam2 (default: {DEFAULT_OSC_ADDR2})")
    parser.add_argument('--fuse', action='store_true', default=True,
                        help="Enable intelligent dual-camera sensor fusion (consensus stream, default: True)")
    parser.add_argument('--no-fuse', dest='fuse', action='store_false',
                        help="Disable fusion and send 2 independent OSC streams")
    parser.add_argument('--grab-thresh', type=float, default=0.70,
                        help="Pinch threshold to trigger Grab (default: 0.70)")
    parser.add_argument('--release-thresh', type=float, default=0.70,
                        help="Pinch threshold to trigger Release (default: 0.70)")
    parser.add_argument('--angle-gain', type=float, default=1.0,
                        help="Angle rotation multiplier/gain (default: 1.0, e.g. 1.2~1.5 for higher sensitivity)")
    parser.add_argument('--invert-angle', dest='invert_angle', action='store_true', default=False,
                        help="Invert rotation delta")
    parser.add_argument('--no-invert-angle', dest='invert_angle', action='store_false',
                        help="Do not invert rotation delta (default: False)")
    parser.add_argument('--no-osc', action='store_true', help="Disable OSC sending (for local test)")
    args, unknown = parser.parse_known_args()

    target_hand1 = args.hand1 if args.hand1 is not None else args.hand
    target_hand2 = args.hand2 if args.hand2 is not None else args.hand

    if args.high_res:
        args.infer_width = args.width
        args.infer_height = args.height
    else:
        args.infer_width = 320
        args.infer_height = 240

    print("==================================================================")
    print("[START] Jetson Nano 듀얼 웹캠 지능형 손 추적 및 센서 융합 시스템")
    print(f" - 카메라 1: 인덱스 {args.cam1} (타겟 손: {target_hand1})")
    print(f" - 카메라 2: 인덱스 {args.cam2} (타겟 손: {target_hand2})")
    print(f" - 센서 융합(Fusion): {'[ON] 2대 시야 통합 (채터링/핑퐁 방지)' if args.fuse else '[OFF] 독립 스트림'}")
    if args.fuse:
        print(f" - 통합 OSC 대상: {args.ip}:{args.port} [{args.addr}] (그랩: <={args.grab_thresh}, 놓음: >={args.release_thresh})")
        print(f" - 각도 감도 배율(Angle Gain): {args.angle_gain}x")
    else:
        print(f" - 독립 OSC 대상: Cam1 [{args.addr1}], Cam2 [{args.addr2}]")
    print(f" - 캡처 해상도: {args.width}x{args.height} | 추론 해상도: {args.infer_width}x{args.infer_height} (고해상도: {args.high_res})")
    print(f" - 1프레임 교차 추론(Interleave): {args.interleave} | 헤드리스(Headless): {args.headless}")
    print("==================================================================")

    # 1. OSC 클라이언트 초기화
    osc_client = None
    if not args.no_osc and SimpleUDPClient is not None:
        try:
            osc_client = SimpleUDPClient(args.ip, args.port)
            print(f"[OSC] 언리얼 엔진 스트리밍 준비 완료: {args.ip}:{args.port}")
        except Exception as e:
            print(f"[경고] OSC 초기화 실패: {e}")

    # 2. 비동기 스레드 카메라 2대 시작
    print("[카메라] 웹캠 1 & 2 초기화 중...")
    cam1 = ThreadedCamera(args.cam1, args.width, args.height, args.fps).start()
    cam2 = ThreadedCamera(args.cam2, args.width, args.height, args.fps).start()
    time.sleep(1.0)  # 카메라 안정화 대기

    # 3. MediaPipe 및 지능형 트래커 초기화 (카메라별 독립 인스턴스)
    api_type1, detector1 = create_mediapipe_detector(max_num_hands=args.max_hands)
    api_type2, detector2 = create_mediapipe_detector(max_num_hands=args.max_hands)
    print(f"[MediaPipe] Cam1: {api_type1} API / Cam2: {api_type2} API 로드 완료 (model_complexity=0)")

    tracker1 = RobustHandTracker(target_handedness=target_hand1, max_lost_frames=6, max_jump_dist=120.0, is_mirrored=True)
    tracker2 = RobustHandTracker(target_handedness=target_hand2, max_lost_frames=6, max_jump_dist=120.0, is_mirrored=True)

    # 듀얼 카메라 지능형 센서 융합기
    fuser = DualCamKnobFuser(
        grab_threshold=args.grab_thresh,
        release_threshold=args.release_thresh,
        invert_angle=args.invert_angle,
        angle_gain=args.angle_gain
    )

    # 상태 관리 변수
    last_sent_angle1 = 0.0
    last_sent_angle2 = 0.0
    fused_angle = 0.0
    fused_pinch = 1.5
    is_grabbed = False
    is_fused_tracked = False

    # 캐시된 직전 상태 (교차 추론 시 재사용)
    cache1 = {'lms': None, 'angle': np.nan, 'pinch': PINCH_WHEN_NO_HAND, 'center': None, 'tracked': False}
    cache2 = {'lms': None, 'angle': np.nan, 'pinch': PINCH_WHEN_NO_HAND, 'center': None, 'tracked': False}

    frame_interval = 1.0 / max(1, args.fps)
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()
    console_timer = time.time()
    loop_index = 0

    print("\n[안내] 듀얼 트래킹 루프가 시작되었습니다. (종료: 'q' 키 또는 Ctrl+C)\n")

    try:
        while True:
            t_start = time.time()
            loop_index += 1

            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()

            if not ret1 or frame1 is None or not ret2 or frame2 is None:
                time.sleep(0.005)
                continue

            h, w = frame1.shape[:2]
            cur_time = time.time()

            # ── [Inference Scheduling] ──
            do_infer_cam1 = True
            do_infer_cam2 = True
            if args.interleave:
                if loop_index % 2 == 1:
                    do_infer_cam2 = False
                else:
                    do_infer_cam1 = False

            # ── [Cam 1 MediaPipe 추론] ──
            if do_infer_cam1:
                infer_frame1 = cv2.resize(frame1, (args.infer_width, args.infer_height), interpolation=cv2.INTER_LINEAR)
                rgb_infer1 = cv2.cvtColor(infer_frame1, cv2.COLOR_BGR2RGB)

                if api_type1 == 'tasks':
                    mp_img1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_infer1)
                    res1 = detector1.detect(mp_img1)
                    lms1, angle1, pinch1, center1, tracked1 = tracker1.update_mediapipe(res1, w, h, timestamp=cur_time)
                else:
                    res1 = detector1.process(rgb_infer1)
                    lms1, angle1, pinch1, center1, tracked1 = tracker1.update_mediapipe_solutions(res1, w, h, timestamp=cur_time)

                cache1 = {'lms': lms1, 'angle': angle1, 'pinch': pinch1, 'center': center1, 'tracked': tracked1}
            else:
                lms1, angle1, pinch1 = cache1['lms'], cache1['angle'], cache1['pinch']
                center1, tracked1 = cache1['center'], cache1['tracked']

            # ── [Cam 2 MediaPipe 추론] ──
            if do_infer_cam2:
                infer_frame2 = cv2.resize(frame2, (args.infer_width, args.infer_height), interpolation=cv2.INTER_LINEAR)
                rgb_infer2 = cv2.cvtColor(infer_frame2, cv2.COLOR_BGR2RGB)

                if api_type2 == 'tasks':
                    mp_img2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_infer2)
                    res2 = detector2.detect(mp_img2)
                    lms2, angle2, pinch2, center2, tracked2 = tracker2.update_mediapipe(res2, w, h, timestamp=cur_time)
                else:
                    res2 = detector2.process(rgb_infer2)
                    lms2, angle2, pinch2, center2, tracked2 = tracker2.update_mediapipe_solutions(res2, w, h, timestamp=cur_time)

                cache2 = {'lms': lms2, 'angle': angle2, 'pinch': pinch2, 'center': center2, 'tracked': tracked2}
            else:
                lms2, angle2, pinch2 = cache2['lms'], cache2['angle'], cache2['pinch']
                center2, tracked2 = cache2['center'], cache2['tracked']

            # ── [추적 품질 점수(Quality) 획득] ──
            q1 = tracker1.get_tracking_quality()
            q2 = tracker2.get_tracking_quality()

            # ── [언리얼 엔진 OSC 스트리밍: 지능형 융합 or 독립 분기] ──
            fuser_debug = ""
            if args.fuse:
                c1_dict = {'tracked': tracked1, 'angle': angle1, 'pinch': pinch1, 'quality': q1}
                c2_dict = {'tracked': tracked2, 'angle': angle2, 'pinch': pinch2, 'quality': q2}
                fused_angle, fused_pinch, is_grabbed, is_fused_tracked, fuser_debug = fuser.update(
                    c1_dict, c2_dict, timestamp=cur_time
                )
                if osc_client is not None:
                    try:
                        osc_client.send_message(args.addr, [float(fused_angle), float(fused_pinch)])
                    except Exception:
                        pass
            else:
                if osc_client is not None:
                    try:
                        if tracked1 and not np.isnan(angle1):
                            send_a1 = float(angle1 % 360.0)
                            send_p1 = float(pinch1)
                            last_sent_angle1 = send_a1
                            osc_client.send_message(args.addr1, [send_a1, send_p1])
                        else:
                            osc_client.send_message(args.addr1, [float(last_sent_angle1), float(PINCH_WHEN_NO_HAND)])

                        if tracked2 and not np.isnan(angle2):
                            send_a2 = float(angle2 % 360.0)
                            send_p2 = float(pinch2)
                            last_sent_angle2 = send_a2
                            osc_client.send_message(args.addr2, [send_a2, send_p2])
                        else:
                            osc_client.send_message(args.addr2, [float(last_sent_angle2), float(PINCH_WHEN_NO_HAND)])
                    except Exception:
                        pass

            # FPS 측정
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            # ── [실시간 콘솔 상태 표시 (0.3초마다 1줄 갱신)] ──
            if time.time() - console_timer >= 0.3:
                console_timer = time.time()
                c1_str = f"C1:ON(p={pinch1:.2f}|q={q1:.2f})" if tracked1 else "C1:WAIT"
                c2_str = f"C2:ON(p={pinch2:.2f}|q={q2:.2f})" if tracked2 else "C2:WAIT"
                if args.fuse:
                    grab_str = "잡음(GRAB)" if is_grabbed else "놓음(REL)"
                    status_line = f"[융합 ON] FPS: {fps_display:4.1f} | 각도: {fused_angle:5.1f} deg | 핀치: {fused_pinch:4.2f} [{grab_str}] | {c1_str} {c2_str} {fuser_debug}"
                else:
                    status_line = f"[독립 ON] FPS: {fps_display:4.1f} | {c1_str} | {c2_str} | 대상: {args.ip}:{args.port}"
                print(f"\r{status_line}   ", end="", flush=True)

            # ── [통합 화면 시각화 (Headless 아닐 때만)] ──
            if not args.headless:
                draw_hand_overlay(frame1, lms1, angle1, pinch1, tracked1, f"Cam1 ({args.hand1}|q={q1:.2f})", fps_display)
                draw_hand_overlay(frame2, lms2, angle2, pinch2, tracked2, f"Cam2 ({args.hand2}|q={q2:.2f})", fps_display)

                # 2개 프레임을 가로 1개 윈도우로 병합하여 X11 부하 절감
                combined = np.hstack([frame1, frame2])

                # 융합 상태 메인 HUD 배너 오버레이
                if args.fuse:
                    banner_color = (0, 0, 255) if is_grabbed else (0, 255, 0)
                    banner_text = f"FUSED: {fused_angle:.1f} deg | Pinch: {fused_pinch:.2f} ({'GRABBED' if is_grabbed else 'RELEASED'}) {fuser_debug}"
                    cv2.putText(combined, banner_text, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

                cv2.imshow("Jetson Dual-Cam Knob Tracker (Fusion Engine)", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    fuser.calibrate_zero()
                    print("\n[영점 맞춤] 노브 회전 각도가 0도로 캘리브레이션되었습니다.")
                elif key == ord('r'):
                    tracker1.reset()
                    tracker2.reset()
                    fuser.reset()
                    print("\n[알림] 두 카메라와 융합기 상태가 리셋되었습니다.")

            # 발열 쓰로틀링 방지 FPS 제한
            elapsed = time.time() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[알림] 사용자에 의해 중단되었습니다.")
    finally:
        print("\n시스템 자원을 정리합니다...")
        cam1.stop()
        cam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        if hasattr(detector1, 'close'):
            detector1.close()
        if hasattr(detector2, 'close'):
            detector2.close()
        print("정상적으로 종료되었습니다.")


if __name__ == '__main__':
    main()
