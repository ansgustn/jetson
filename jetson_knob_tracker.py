#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Python 2로 실행된 경우 자동으로 python3로 전환
if sys.version_info[0] < 3:
    os.execvp("python3", ["python3"] + sys.argv)

"""
jetson_knob_tracker.py
======================
NVIDIA Jetson Nano(ARM Cortex-A57 4코어, 4GB RAM) 전용 초경량 실시간 손 추적 및 언리얼 엔진(OSC) 연동기.
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
import time
import cv2
import numpy as np

# OSC 통신 및 트래킹 유틸리티
import mediapipe as mp
from hand_tracker_utils import (
    RobustHandTracker,
    OneEuroFilter,
    DualCamKnobFuser,
    calc_angle,
    pinch_ratio_landmarks,
    SimpleUDPClient,
)

# ── 기본 설정 상수 ──
DEFAULT_UE_IP = "192.168.137.1"
DEFAULT_UE_PORT = 8000
DEFAULT_OSC_ADDRESS = "/mediapipe/knob/angle"
PINCH_WHEN_NO_HAND = 1.5


def build_camera_capture(cam_index, width, height, use_gstreamer=False):
    """
    OS 및 장치에 최적화된 VideoCapture 객체를 생성합니다.
    """
    sys_name = platform.system()
    
    if use_gstreamer and sys_name == 'Linux':
        # Jetson Nano CSI 카메라용 GStreamer 파이프라인
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={cam_index} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, format=(string)NV12, framerate=(fraction)30/1 ! "
            f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        )
        print(f"[카메라] Jetson CSI GStreamer 파이프라인 시작: sensor-id={cam_index}")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    elif sys_name == 'Windows':
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    elif sys_name == 'Linux':
        # Jetson Nano USB 웹캠: V4L2 백엔드 명시 및 MJPG 포맷 강제 지정
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        raise RuntimeError(f"카메라 (인덱스 {cam_index})를 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 프레임 버퍼 지연 제거 (임베디드 필수)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def init_mediapipe_detector():
    """
    MediaPipe 손 검출기를 초기화합니다.
    Tasks API(hand_landmarker.task) 우선 시도, 실패 시 Solutions API로 자동 폴백.
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
                num_hands=2,  # 두 손을 동시 감지해야 다른 손 난입 시 목표 손을 선별 가능
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            detector = HandLandmarker.create_from_options(options)
            print("[MediaPipe] Tasks HandLandmarker(최신 API) 로드 완료 (num_hands=2)")
            return 'tasks', detector
        except Exception as e:
            print(f"[알림] Tasks API 초기화 실패 ({e}), Solutions API로 전환합니다.")

    # Fallback: Solutions API (Jetson Nano에서 널리 안정적으로 구동됨)
    mp_hands = mp.solutions.hands
    try:
        detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # 0: 초경량 모바일/임베디드 모델 (Jetson Nano 최고 성능)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except TypeError:
        # MediaPipe v0.8.5 등 구버전에서는 model_complexity 인자 없이 초기화
        detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    print("[MediaPipe] Solutions Hands 로드 완료")
    return 'solutions', detector


def main():
    parser = argparse.ArgumentParser(description="Jetson Nano Ultra-Lightweight Hand Tracking & OSC Streamer")
    parser.add_argument('--cam', type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument('--csi', action='store_true', help="Use Jetson CSI camera via GStreamer")
    parser.add_argument('--width', type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument('--height', type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument('--infer-width', type=int, default=640, help="Inference width (default: 640)")
    parser.add_argument('--infer-height', type=int, default=480, help="Inference height (default: 480)")
    parser.add_argument('--target-hand', '--target_hand', '--hand', type=str, default='Any', choices=['Right', 'Left', 'Auto', 'Any'],
                        help="Target hand to track (default: Any)")
    parser.add_argument('--ip', type=str, default=DEFAULT_UE_IP, help=f"Unreal Engine IP (default: {DEFAULT_UE_IP})")
    parser.add_argument('--port', type=int, default=DEFAULT_UE_PORT, help=f"Unreal Engine OSC Port (default: {DEFAULT_UE_PORT})")
    parser.add_argument('--address', type=str, default=DEFAULT_OSC_ADDRESS, help=f"OSC Address (default: {DEFAULT_OSC_ADDRESS})")
    parser.add_argument('--headless', action='store_true', help="Run in headless mode without GUI preview window")
    parser.add_argument('--grab-thresh', type=float, default=0.70, help="Pinch threshold to trigger Grab (default: 0.70)")
    parser.add_argument('--release-thresh', type=float, default=0.70, help="Pinch threshold to trigger Release (default: 0.70)")
    parser.add_argument('--angle-gain', type=float, default=1.0, help="Angle rotation multiplier/gain (default: 1.0, e.g. 1.2~1.5)")
    parser.add_argument('--invert-angle', dest='invert_angle', action='store_true', default=False,
                        help="Invert rotation angle")
    parser.add_argument('--no-invert-angle', dest='invert_angle', action='store_false',
                        help="Do not invert rotation angle (default: False)")
    parser.add_argument('--fps', type=int, default=30, help="Target loop FPS cap (default: 30)")
    parser.add_argument('--no-osc', action='store_true', help="Disable OSC sending (for standalone test)")
    args, unknown = parser.parse_known_args()
    # 오타나 누락된 대시(-)로 들어온 인자 자동 보정
    for item in unknown:
        if item in ['Right', 'Left', 'Auto', 'Any']:
            args.target_hand = item

    print("==================================================================")
    print("[START] Jetson Nano 고정밀 손 추적기 (Jetson Knob Tracker)")
    print(f" - 카메라: 인덱스 {args.cam} (CSI: {args.csi}, 캡처: {args.width}x{args.height})")
    print(f" - 추론 해상도: {args.infer_width}x{args.infer_height} (고정밀 640x480)")
    print(f" - 목표 손(Target Hand): {args.target_hand}")
    print(f" - 그랩 기준: <={args.grab_thresh} (그랩) / >={args.release_thresh} (놓음)")
    print(f" - 각도 감도 배율(Angle Gain): {args.angle_gain}x")
    print(f" - 각도 반전(Invert Angle): {args.invert_angle}")
    print(f" - 헤드리스(Headless) 모드: {args.headless}")
    print(f" - OSC 전송: {args.ip}:{args.port} [{args.address}] (활성: {not args.no_osc})")
    print("==================================================================")

    # 1. OSC 클라이언트 초기화
    osc_client = None
    if not args.no_osc and SimpleUDPClient is not None:
        try:
            osc_client = SimpleUDPClient(args.ip, args.port)
            print(f"[OSC] 언리얼 엔진 스트리밍 준비 완료: {args.ip}:{args.port}")
        except Exception as e:
            print(f"[경고] OSC 초기화 실패: {e}")

    # 2. 카메라 초기화
    cap = build_camera_capture(args.cam, args.width, args.height, use_gstreamer=args.csi)

    # 3. MediaPipe 검출기 및 지능형 트래커 초기화
    api_type, detector = init_mediapipe_detector()
    tracker = RobustHandTracker(
        target_handedness=args.target_hand,
        max_lost_frames=15,    # 약 0.5초 동안 관성 유지
        max_jump_dist=120.0,   # 급격한 순간이동 배제
        is_mirrored=True       # cv2.flip 거울 반전 시 왼손/오른손 자동 보정
    )

    fuser = DualCamKnobFuser(
        grab_threshold=args.grab_thresh,
        release_threshold=args.release_thresh,
        invert_angle=args.invert_angle,
        angle_gain=args.angle_gain
    )

    # 루프 타이밍 및 상태 관리
    last_sent_angle = 0.0
    frame_interval = 1.0 / max(1, args.fps)
    prev_loop_time = time.time()
    fps_display = 0.0
    fps_counter = 0
    fps_timer = time.time()
    console_timer = time.time()

    is_grabbed = False

    print("\n[안내] 트래킹을 시작합니다. 종료하려면 키보드 'q' 또는 Ctrl+C를 누르세요.\n")

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 거울 모드 반전 (조작 시 직관적인 시야 확보)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ── [고정밀 MediaPipe 추론] ──
            if args.infer_width != w or args.infer_height != h:
                infer_frame = cv2.resize(frame, (args.infer_width, args.infer_height), interpolation=cv2.INTER_LINEAR)
            else:
                infer_frame = frame
            rgb_infer = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)

            current_timestamp = time.time()
            angle = np.nan
            pinch = PINCH_WHEN_NO_HAND
            pixel_lms = None
            is_tracked = False

            if api_type == 'tasks':
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_infer)
                mp_result = detector.detect(mp_image)
                pixel_lms, angle, pinch, center, is_tracked = tracker.update_mediapipe(
                    mp_result, w, h, timestamp=current_timestamp
                )
            else:
                results = detector.process(rgb_infer)
                pixel_lms, angle, pinch, center, is_tracked = tracker.update_mediapipe_solutions(
                    results, w, h, timestamp=current_timestamp
                )

            # ── [지능형 그랩 게이팅 & 회전 누적] ──
            c1_info = {
                'tracked': is_tracked,
                'angle': angle,
                'pinch': pinch,
                'quality': tracker.get_tracking_quality()
            }
            send_angle, send_pinch, is_grabbed, is_fused_tracked, fuser_debug = fuser.update(
                c1_info, {'tracked': False}, timestamp=current_timestamp
            )

            # ── [언리얼 엔진 OSC 전송] ──
            if osc_client is not None:
                try:
                    last_sent_angle = send_angle
                    osc_client.send_message(args.address, [float(send_angle), float(send_pinch)])
                except Exception as e:
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
                if is_tracked:
                    p_state = "잡음(GRAB)" if is_grabbed else "놓음(REL)"
                    status_line = f"[추적 ON] FPS: {fps_display:4.1f} | 각도: {angle:5.1f} deg | 핀치: {pinch:4.2f} [{p_state}] | OSC -> {args.ip}:{args.port}"
                else:
                    status_line = f"[대기 WAIT] FPS: {fps_display:4.1f} | 손 미감지 (손을 비춰주세요) | OSC: Release 전송"
                print(f"\r{status_line}   ", end="", flush=True)

            # ── [시각화 화면 출력 (Headless 모드가 아닐 때만)] ──
            if not args.headless:
                if is_tracked and pixel_lms is not None:
                    # 뼈대 및 핵심 랜드마크 시각화
                    pt_wrist = (int(pixel_lms[0][0]), int(pixel_lms[0][1]))
                    pt_middle = (int(pixel_lms[9][0]), int(pixel_lms[9][1]))
                    pt_thumb = (int(pixel_lms[4][0]), int(pixel_lms[4][1]))
                    pt_index = (int(pixel_lms[8][0]), int(pixel_lms[8][1]))
                    pt_idx_mcp = (int(pixel_lms[5][0]), int(pixel_lms[5][1]))
                    pt_pnk_mcp = (int(pixel_lms[17][0]), int(pixel_lms[17][1]))

                    # 종축(손목-중지, 초록) + 횡축(너클, 하늘색) 회전선
                    cv2.line(frame, pt_wrist, pt_middle, (0, 255, 0), 2)
                    cv2.line(frame, pt_idx_mcp, pt_pnk_mcp, (255, 255, 0), 2)
                    cv2.circle(frame, pt_wrist, 5, (255, 0, 0), -1)
                    cv2.circle(frame, pt_middle, 5, (0, 0, 255), -1)

                    # 엄지 - 검지 핀치선 (잡음: 초록, 놓음: 빨강)
                    pinch_color = (0, 255, 0) if is_grabbed else (0, 0, 255)
                    cv2.line(frame, pt_thumb, pt_index, pinch_color, 2)
                    cv2.circle(frame, pt_thumb, 4, pinch_color, -1)
                    cv2.circle(frame, pt_index, 4, pinch_color, -1)

                    # 핀치 텍스트
                    mid_x = (pt_thumb[0] + pt_index[0]) // 2
                    mid_y = (pt_thumb[1] + pt_index[1]) // 2 - 10
                    state_lbl = f"GRAB ({pinch:.2f})" if is_grabbed else f"REL ({pinch:.2f})"
                    cv2.putText(frame, state_lbl, (mid_x - 30, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pinch_color, 2)

                # HUD 오버레이
                cv2.putText(frame, f"FPS: {fps_display:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                p_desc = "GRAB" if is_grabbed else "REL"
                status_str = f"Angle: {angle:.1f} deg | Pinch: {pinch:.2f} [{p_desc}]" if is_tracked else "Hand LOST (Release)"
                status_color = (0, 255, 0) if (is_tracked and is_grabbed) else ((0, 255, 255) if is_tracked else (0, 0, 255))
                cv2.putText(frame, status_str, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"Locked: {tracker.locked_handedness or args.target_hand}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow("Jetson Knob Tracker Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    tracker.reset()
                    is_grabbed = False
                    grab_confirm_frames = 0
                    release_confirm_frames = 0
                    print("[알림] 트래커 상태가 리셋되었습니다.")

            # FPS 제한으로 CPU 과열 방지
            elapsed = time.time() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[알림] 사용자에 의해 중단되었습니다.")
    finally:
        print("\n시스템을 안전하게 종료합니다...")
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if hasattr(detector, 'close'):
            detector.close()
        print("종료 완료.")


if __name__ == '__main__':
    main()
