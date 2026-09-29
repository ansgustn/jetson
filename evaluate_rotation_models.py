#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Python 2로 실행된 경우 자동으로 python3로 전환
if sys.version_info[0] < 3:
    os.execvp("python3", ["python3"] + sys.argv)

# Jetson Nano 최적화 환경 변수 자동 설정
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import math
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import socket
import json
import argparse
import platform
import gc

# 지능형 단일 손 추적 및 스무딩 모듈, 경량 OSC 클라이언트, 듀얼 카메라 융합 모듈 임포트
from hand_tracker_utils import RobustHandTracker, DualCamKnobFuser, calc_angle, pinch_ratio_landmarks, SimpleUDPClient
import mediapipe as mp

try:
    import pykinect_azure as pykinect
except ImportError:
    pykinect = None

try:
    from rtmlib import Hand
except ImportError:
    Hand = None

try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None
    transforms = None

try:
    from PIL import Image
except ImportError:
    Image = None



# =========================================================================
# [Unreal Engine OSC / UDP 실시간 통신 설정 (mediapipe_knob_osc.py 기반)]
# =========================================================================
ENABLE_OSC = True
UE_IP = "192.168.137.1"               # 언리얼 엔진 실행 PC IP
UE_PORT = 8000                        # 언리얼 엔진 OSC 플러그인 수신 포트
OSC_ADDRESS = "/mediapipe/knob/angle" # BP_Knob이 수신하는 메인 OSC Address
PINCH_WHEN_NO_HAND = 1.5              # 손이 감지되지 않을 때 전송할 핀치값 (언리얼 Release 유도)

# FreiHAND 모델 경로 등록
sys.path.append(os.path.abspath('HandTracking-master'))
try:
    from model import FreiHANDModel  # type: ignore
except ImportError:
    FreiHANDModel = None


def get_camera_backend():
    """OS에 적합한 OpenCV 비디오 캡처 백엔드를 반환합니다."""
    sys_name = platform.system()
    if sys_name == 'Windows':
        return cv2.CAP_DSHOW
    elif sys_name == 'Linux':
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class CameraThread(threading.Thread):
    def __init__(self, cam_type, device_index, cam_name, width=640, height=480, fps=30):
        super().__init__()
        self.cam_type = cam_type
        self.device_index = device_index
        self.cam_name = cam_name
        self.width = width
        self.height = height
        self.fps = fps
        self.running = True
        
        self.lock = threading.Lock()
        self.current_frame = None
        self.camera_ready = False
        
        # ── 1. 카메라 초기화 (Linux V4L2 + MJPG 하드웨어 가속 적용) ──
        try:
            backend = get_camera_backend()
            self.cap = cv2.VideoCapture(self.device_index, backend)
            if not self.cap.isOpened() and backend != cv2.CAP_ANY:
                self.cap = cv2.VideoCapture(self.device_index)

            # Linux V4L2 MJPG 포맷 강제 (웹캠 2대 동시 연결 시 USB 버스 대역폭 포화 방지)
            if platform.system() == 'Linux':
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                except Exception:
                    pass

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                raise ValueError(f"카메라(인덱스 {self.device_index})를 열 수 없습니다.")
            self.camera_ready = True
            print(f"[{self.cam_name}] 웹캠(idx={self.device_index}) {self.width}x{self.height} 초기화 성공!")
        except Exception as e:
            print(f"❌ [{self.cam_name}] 초기화 실패: {e}")
            self.running = False
            return
            
    def run(self):
        while self.running:
            if not self.camera_ready or self.cap is None:
                time.sleep(0.01)
                continue
                
            ret, frame_bgr = self.cap.read()
            if ret and frame_bgr is not None:
                frame = cv2.flip(frame_bgr, 1)
                with self.lock:
                    self.current_frame = frame
            else:
                time.sleep(0.005)
                
        # 스레드 종료 시 카메라 자원 안전하게 해제
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def get_frame(self):
        with self.lock:
            if not self.camera_ready or self.current_frame is None:
                return False, None
            return True, self.current_frame.copy()

    def stop(self):
        self.running = False


def create_mediapipe_detector(max_num_hands=1):
    """
    각 카메라별 독립된 MediaPipe Hands 검출기를 생성합니다.
    (두 대의 카메라가 하나의 검출기를 번갈아 호출하면 이전 프레임의 바운딩 박스가 교차 오염되어
     손가락 튐 및 중지/약지 오인식이 발생하므로, 반드시 카메라마다 독립 인스턴스를 유지해야 합니다.)
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
        except Exception:
            pass

    mp_hands = mp.solutions.hands
    try:
        detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,  # 0: Jetson Nano 최적화 초경량 모델
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


def draw_detailed_hand_skeleton(frame, pixel_lms, angle, pinch, is_tracked, grab_threshold=0.70):
    """
    미디어파이프 21개 관절 본(Bone) 및 엄지/검지/중지 식별 라벨을 명확하게 시각화.
    엄지와 검지를 다른 손가락과 완벽하게 구분할 수 있도록 색상별로 렌더링.
    """
    if not is_tracked or pixel_lms is None or len(pixel_lms) < 21:
        return

    connections = [
        # Palm
        (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index
        (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20)
    ]

    pts = [(int(p[0]), int(p[1])) for p in pixel_lms]

    # 기본 골격 라인
    for start_idx, end_idx in connections:
        cv2.line(frame, pts[start_idx], pts[end_idx], (200, 200, 200), 1)

    # 손가락별 관절 점 그리기
    for i, pt in enumerate(pts):
        if i in [1, 2, 3]:       # Thumb
            cv2.circle(frame, pt, 3, (0, 255, 255), -1)
        elif i in [5, 6, 7]:     # Index
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)
        elif i in [9, 10, 11]:   # Middle
            cv2.circle(frame, pt, 3, (255, 150, 0), -1)
        elif i in [13, 14, 15]:  # Ring
            cv2.circle(frame, pt, 3, (0, 165, 255), -1)
        elif i in [17, 18, 19]:  # Pinky
            cv2.circle(frame, pt, 3, (255, 0, 255), -1)
        elif i == 0:             # Wrist
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    pt_thumb = pts[4]
    pt_index = pts[8]
    pt_middle = pts[12]
    pt_wrist = pts[0]
    pt_mcp = pts[9]
    pt_idx_mcp = pts[5]
    pt_pnk_mcp = pts[17]

    # 회전 축 벡터: 종축(손목 -> 중지 MCP, 초록) + 횡축(검지뿌리 -> 새끼뿌리 너클, 하늘색)
    cv2.line(frame, pt_wrist, pt_mcp, (0, 255, 0), 2)
    cv2.line(frame, pt_idx_mcp, pt_pnk_mcp, (255, 255, 0), 2)

    # 핀치 상태 시각화 (엄지 끝 <-> 검지 끝, 기본 0.70 이하 시 그랩)
    is_pinch = (pinch <= grab_threshold)
    pinch_color = (0, 255, 0) if is_pinch else (0, 0, 255)
    cv2.line(frame, pt_thumb, pt_index, pinch_color, 3 if is_pinch else 2)

    # 손가락 끝 원형 마커
    cv2.circle(frame, pt_thumb, 7, (0, 255, 255), -1)   # Thumb: Yellow
    cv2.circle(frame, pt_index, 7, (0, 255, 0), -1)     # Index: Green
    cv2.circle(frame, pt_middle, 5, (255, 150, 0), -1)  # Middle: Blue

    # 화면에 엄지/검지 명확히 텍스트 표시
    cv2.putText(frame, "Thumb", (pt_thumb[0] - 20, pt_thumb[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(frame, "Index", (pt_index[0] - 20, pt_index[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 핀치 상태 텍스트
    mid_pinch_x = (pt_thumb[0] + pt_index[0]) // 2
    mid_pinch_y = (pt_thumb[1] + pt_index[1]) // 2 - 10
    state_str = f"GRAB ({pinch:.2f})" if is_pinch else f"REL ({pinch:.2f})"
    cv2.putText(frame, state_str, (mid_pinch_x - 30, mid_pinch_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pinch_color, 2)


def main():
    global UE_IP, UE_PORT
    parser = argparse.ArgumentParser(description="Multi-Cam Model-Sequential Evaluator with Robust Hand Tracking")
    parser.add_argument('--models', nargs='+', default=None, help="Models to evaluate, e.g., --models MP RTM Frei (default: all available)")
    parser.add_argument('--mp-only', action='store_true', default=False,
                        help="Run MediaPipe only continuous real-time streaming mode without evaluation steps")
    parser.add_argument('--ip', type=str, default=UE_IP, help=f"Unreal Engine IP (default: {UE_IP})")
    parser.add_argument('--port', type=int, default=UE_PORT, help=f"Unreal Engine Port (default: {UE_PORT})")
    parser.add_argument('--invert-angle', dest='invert_angle', action='store_true', default=False,
                        help="Invert rotation angle")
    parser.add_argument('--no-invert-angle', dest='invert_angle', action='store_false',
                        help="Do not invert rotation angle (default: False)")
    parser.add_argument('--target-hand', type=str, default='Any', choices=['Right', 'Left', 'Auto', 'Any'],
                        help="Target hand to track (default: Any)")
    parser.add_argument('--max-hands', type=int, default=1, choices=[1, 2],
                        help="Max hands per camera (default: 1 for stable knob pinch without finger confusion)")
    parser.add_argument('--grab-thresh', type=float, default=0.70,
                        help="Pinch threshold to trigger Grab (default: 0.70, generous grab detection)")
    parser.add_argument('--release-thresh', type=float, default=0.70,
                        help="Pinch threshold to trigger Release (default: 0.70, release detection)")
    parser.add_argument('--angle-gain', type=float, default=1.0,
                        help="Angle rotation multiplier/gain (default: 1.0, e.g. 1.2~1.5 for higher sensitivity)")
    parser.add_argument('--downscale', action='store_true', default=False,
                        help="Downscale frame for inference (320x240) to boost performance on Jetson Nano")
    parser.add_argument('--jetson', action='store_true', default=False,
                        help="Enable Jetson Nano lightweight optimizations (downscale + memory GC)")
    parser.add_argument('--cam1', type=int, default=0, help="Camera 1 device index (default: 0)")
    parser.add_argument('--cam2', type=int, default=1, help="Camera 2 device index (default: 1, set -1 for 1 camera)")
    parser.add_argument('--width', type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument('--height', type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument('--fps', type=int, default=30, help="Capture FPS (default: 30)")
    parser.add_argument('--headless', action='store_true', default=False,
                        help="Run without GUI preview window")
    args = parser.parse_args()

    use_downscale = args.downscale or args.jetson
    UE_IP = args.ip
    UE_PORT = args.port
    only_mp = args.mp_only or (args.models == ['MP'])

    print("========================================")
    if only_mp:
        print("🎯 [MediaPipe 단독 실시간 스트리밍 모드]")
    else:
        print("🚀 다중 카메라 모델 순차 평가 시스템 (Jetson Nano 대응 최적화)")
    print(f" - 카메라 1: 인덱스 {args.cam1} (일반 웹캠)")
    print(f" - 카메라 2: 인덱스 {args.cam2} ({'일반 웹캠' if args.cam2 >= 0 else '미사용'})")
    print(f" - 목표 손(Target Hand): {args.target_hand}")
    print(f" - 그랩 기준: <={args.grab_thresh} (그랩) / >={args.release_thresh} (놓음)")
    print(f" - 각도 감도 배율(Angle Gain): {args.angle_gain}x")
    print(f" - 언리얼 대상 IP: {UE_IP}:{UE_PORT}")
    print(f" - 각도 반전(Invert Angle): {args.invert_angle}")
    print(f" - 캡처 해상도: {args.width}x{args.height} | 경량 추론 다운스케일: {use_downscale}")
    print(f" - 헤드리스 모드: {args.headless}")
    print("========================================")

    cam_configs = []
    cam_configs.append({'type': 'webcam', 'idx': args.cam1, 'name': f'Camera_1 (Webcam {args.cam1})'})
    if args.cam2 >= 0 and args.cam2 != args.cam1:
        cam_configs.append({'type': 'webcam', 'idx': args.cam2, 'name': f'Camera_2 (Webcam {args.cam2})'})
            
    print(f"최종 할당된 카메라 목록:")
    for cfg in cam_configs:
        print(f" - {cfg['name']}: {cfg['type']} (idx={cfg['idx']})")

    if len(cam_configs) == 0:
        print("연결된 카메라가 없습니다. 종료합니다.")
        return

    print("========================================")
    print("모델 초기화 중입니다...")
    
    freihand_model = None
    freihand_transform = None
    rtm_detector = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if torch is not None else None

    if not only_mp:
        model_path = os.path.join("HandTracking-master", "freihand_custom_model.pth")
        if FreiHANDModel is not None and os.path.exists(model_path):
            try:
                freihand_model = FreiHANDModel(num_keypoints=21).to(device)
                freihand_model.load_state_dict(torch.load(model_path, map_location=device))
                freihand_model.eval()
                freihand_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("✅ FreiHAND 가중치 로드 완료!")
            except Exception as e:
                print(f"⚠️ FreiHAND 로드 실패: {e}")
        else:
            print("⚠️ FreiHAND 모델 가중치가 없어 측정에서 제외됩니다.")

        # RTMPose
        try:
            rtm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            rtm_detector = Hand(to_openpose=False, backend='onnxruntime', device=rtm_device)
            print("✅ RTMPose 로드 완료!")
        except Exception as e:
            print(f"⚠️ RTMPose 로드 실패: {e}")

    print("카메라별 독립 MediaPipe 추론기를 초기화합니다...")
    mp_detectors = {}
    for cfg in cam_configs:
        cname = cfg['name']
        det_type, det = create_mediapipe_detector(max_num_hands=args.max_hands)
        mp_detectors[cname] = (det_type, det)
        print(f"✅ [{cname}] MediaPipe ({det_type}, max_hands={args.max_hands}) 초기화 완료!")

    threads = []
    for cfg in cam_configs:
        t = CameraThread(cfg['type'], cfg['idx'], cfg['name'], width=args.width, height=args.height, fps=args.fps)
        threads.append(t)
        
    for t in threads:
        t.start()
        
    print("\n[안내] 카메라 준비를 기다립니다. (워밍업 3초)")
    time.sleep(3)

    # Unreal Engine OSC 클라이언트 준비
    osc_client = None
    if ENABLE_OSC:
        try:
            osc_client = SimpleUDPClient(UE_IP, UE_PORT)
            print(f"📡 Unreal Engine OSC 스트리밍 활성화 -> {UE_IP}:{UE_PORT} (Address: {OSC_ADDRESS})")
        except Exception as e:
            print(f"⚠️ OSC 클라이언트 생성 실패: {e}")

    # 평가할 모델 단계 구성
    if only_mp:
        stages = ["MP"]
    elif args.models:
        all_stages = ["MP", "RTM"]
        if freihand_model is not None:
            all_stages.append("Frei")
        stages = [m for m in args.models if m in all_stages]
    else:
        stages = ["MP", "RTM"]
        if freihand_model is not None:
            all_stages.append("Frei")
        
    target_angles = [90, 180, 270, 360]
    # 스냅샷 저장소: snapshot_data[model][target][cam]
    snapshot_data = {
        model: {
            target: {cfg['name']: np.nan for cfg in cam_configs} 
            for target in target_angles
        } for model in stages
    }

    # 카메라별 독립 RobustHandTracker 인스턴스 생성 (다중 손 간섭 완벽 방지)
    trackers = {
        t.cam_name: RobustHandTracker(
            target_handedness=args.target_hand,
            max_lost_frames=15,
            max_jump_dist=120.0
        ) for t in threads
    }

    # 듀얼 카메라 지능형 융합기 초기화 (슈미트 트리거 및 각도/핀치 융합)
    fuser = DualCamKnobFuser(
        grab_threshold=args.grab_thresh,
        release_threshold=args.release_thresh,
        invert_angle=args.invert_angle,
        angle_gain=args.angle_gain
    )

    for stage_idx, stage_model in enumerate(stages):
        print(f"\n========================================")
        print(f"[Stage {stage_idx+1}/{len(stages)}] '{stage_model}' 모델 측정을 준비합니다.")
        print(">>> 's' 키: 카메라 영점(0도) 맞추기")
        print(">>> '1', '2', '3', '4' 키: 각각 90, 180, 270, 360도에 손을 맞추고 누르면 스냅샷 저장!")
        print(">>> 'n' 키: 다음 모델로 넘어가기")
        
        # 새 스테이지 진입 시 트래커 및 상태 초기화
        for trk in trackers.values():
            trk.reset()
        fuser.reset()
        baseline_angles = {t.cam_name: 0.0 for t in threads}
        current_angles = {t.cam_name: {stage_model: np.nan} for t in threads}
        last_sent_angle = 0.0
        quit_requested = False
        
        while True:
            frames = []
            cam_data = {}
            cur_time = time.time()
            
            for t in threads:
                ret, frame = t.get_frame()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    angle = np.nan
                    pinch = PINCH_WHEN_NO_HAND
                    pixel_lms = None
                    is_tracked = False
                    
                    # ── [Stage 1: MediaPipe] ──
                    if stage_model == "MP":
                        mp_type, det = mp_detectors[t.cam_name]
                        if use_downscale:
                            infer_rgb = cv2.resize(rgb_frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
                        else:
                            infer_rgb = rgb_frame

                        if mp_type == 'tasks':
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
                            mp_result = det.detect(mp_image)
                            pixel_lms, angle, pinch, center, is_tracked = trackers[t.cam_name].update_mediapipe(
                                mp_result, w, h, timestamp=cur_time
                            )
                        else:
                            mp_result = det.process(infer_rgb)
                            pixel_lms, angle, pinch, center, is_tracked = trackers[t.cam_name].update_mediapipe_solutions(
                                mp_result, w, h, timestamp=cur_time
                            )

                        if is_tracked and pixel_lms is not None:
                            draw_detailed_hand_skeleton(frame, pixel_lms, angle, pinch, is_tracked, grab_threshold=args.grab_thresh)
                                
                    # ── [Stage 2: RTMPose] ──
                    elif stage_model == "RTM":
                        if use_downscale:
                            infer_bgr = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
                        else:
                            infer_bgr = frame

                        keypoints_all, scores_all = rtm_detector(infer_bgr)
                        if use_downscale and keypoints_all is not None and len(keypoints_all) > 0:
                            keypoints_all = np.array(keypoints_all, dtype=np.float32)
                            keypoints_all[..., 0] *= 2.0
                            keypoints_all[..., 1] *= 2.0

                        pixel_lms, angle, pinch, center, is_tracked = trackers[t.cam_name].update_rtmpose(
                            keypoints_all, scores_all, timestamp=cur_time
                        )

                        if is_tracked and pixel_lms is not None:
                            draw_detailed_hand_skeleton(frame, pixel_lms, angle, pinch, is_tracked, grab_threshold=args.grab_thresh)
                            
                    # ── [Stage 3: FreiHAND] ──
                    elif stage_model == "Frei":
                        if freihand_model is not None:
                            # 트래커의 이전 추적 중심점을 활용하여 안정적인 바운딩 박스 크롭
                            center = trackers[t.cam_name].prev_center
                            if center is None:
                                keypoints_all, scores_all = rtm_detector(frame)
                                _, _, _, center, _ = trackers[t.cam_name].update_rtmpose(keypoints_all, scores_all)
                                
                            if center is not None:
                                center_x, center_y = int(center[0]), int(center[1])
                                box_size = 110
                                x1, y1 = max(0, center_x - box_size), max(0, center_y - box_size)
                                x2, y2 = min(w, center_x + box_size), min(h, center_y + box_size)
                                if x2 - x1 > 20 and y2 - y1 > 20:
                                    cropped = rgb_frame[y1:y2, x1:x2]
                                    try:
                                        pil_img = Image.fromarray(cropped)
                                        input_tensor = freihand_transform(pil_img).unsqueeze(0).to(device)
                                        with torch.no_grad():
                                            outputs = freihand_model(input_tensor)
                                        lm_3d = outputs.view(21, 3).cpu().numpy()
                                        lm_3d[:, 1] = -lm_3d[:, 1]
                                        raw_angle = calc_angle((lm_3d[0,0], lm_3d[0,1]), (lm_3d[9,0], lm_3d[9,1]))
                                        raw_pinch = pinch_ratio_landmarks(
                                            (lm_3d[4,0], lm_3d[4,1]), (lm_3d[8,0], lm_3d[8,1]),
                                            (lm_3d[0,0], lm_3d[0,1]), (lm_3d[9,0], lm_3d[9,1])
                                        )
                                        angle = trackers[t.cam_name].angle_filter.filter(raw_angle, cur_time)
                                        pinch = trackers[t.cam_name].pinch_filter.filter(raw_pinch, cur_time)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                                        is_tracked = True
                                    except Exception:
                                        pass
                    
                    # --- 텍스트 오버레이 ---
                    y_offset = 30
                    cv2.putText(frame, f"[{t.cam_name}]", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    color = (255,255,255)
                    if stage_model == "MP": color = (255,50,50)
                    elif stage_model == "RTM": color = (50,255,50)
                    elif stage_model == "Frei": color = (50,50,255)
                    
                    cv2.putText(frame, f"{stage_model}: {angle:.1f}" if not np.isnan(angle) else f"{stage_model}: NaN", (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Pinch: {pinch:.2f}", (10, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    current_angles[t.cam_name][stage_model] = angle
                    q = trackers[t.cam_name].get_tracking_quality()
                    cam_data[t.cam_name] = {'tracked': is_tracked, 'angle': angle, 'pinch': pinch, 'quality': q}
                    frames.append(frame)
                else:
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    
            # ── [DualCamKnobFuser 센서 융합 및 언리얼 OSC 전송] ──
            if len(cam_configs) >= 2:
                name1 = cam_configs[0]['name']
                name2 = cam_configs[1]['name']
                c1 = cam_data.get(name1, {'tracked': False})
                c2 = cam_data.get(name2, {'tracked': False})
                fused_angle, fused_pinch, is_grabbed, is_fused_tracked, fuser_debug = fuser.update(
                    c1, c2, timestamp=cur_time
                )
            else:
                name1 = cam_configs[0]['name']
                c1 = cam_data.get(name1, {'tracked': False})
                fused_angle, fused_pinch, is_grabbed, is_fused_tracked, fuser_debug = fuser.update(
                    c1, {'tracked': False}, timestamp=cur_time
                )

            if osc_client is not None:
                try:
                    last_sent_angle = fused_angle
                    osc_client.send_message(OSC_ADDRESS, [float(fused_angle), float(fused_pinch)])
                except Exception:
                    pass
                    
            if frames and not args.headless:
                display = np.hstack(frames)
                if display.shape[1] > 1920:
                    display = cv2.resize(display, (display.shape[1]//2, display.shape[0]//2))
                    
                if only_mp:
                    # MediaPipe 단독 실시간 스트리밍 전용 HUD
                    cv2.putText(display, "🎯 MediaPipe Dual-Cam Knob Streamer", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    p_state = "GRABBED" if is_grabbed else "RELEASED"
                    p_color = (0, 0, 255) if is_grabbed else (0, 255, 0)
                    cv2.putText(display, f"Angle: {fused_angle:.1f} deg | Pinch: {fused_pinch:.2f} [{p_state}] {fuser_debug}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, p_color, 2)
                    cv2.putText(display, f"Streaming -> {UE_IP}:{UE_PORT} [{OSC_ADDRESS}]", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(display, "Press 's' to Zero (0 deg) | 'q' to Quit", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                else:
                    # 상단 안내 텍스트
                    cv2.putText(display, f"Stage {stage_idx+1}/{len(stages)} : {stage_model} Only", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(display, "Press 's' to Zero, 'n' to Next Stage", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display, "[Keys] 1:90  2:180  3:270  4:360", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    # 스냅샷 저장 현황 표시
                    y_pos = 160
                    for tgt in target_angles:
                        cam1_val = snapshot_data[stage_model][tgt][cam_configs[0]['name']]
                        status = "Recorded" if not np.isnan(cam1_val) else "Empty"
                        color = (0, 255, 0) if status == "Recorded" else (0, 0, 255)
                        cv2.putText(display, f"{tgt} deg: {status}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_pos += 30
                
                cv2.imshow('Multi-Cam Hand Tracker', display)

            key = cv2.waitKey(1) & 0xFF
            
            target_to_record = None
            if not only_mp:
                if key == ord('1'): target_to_record = 90
                elif key == ord('2'): target_to_record = 180
                elif key == ord('3'): target_to_record = 270
                elif key == ord('4'): target_to_record = 360
            if key == ord('q'):
                quit_requested = True
                break
            elif key == ord('n') and not only_mp:
                # 스테이지 종료 시 메모리 정리
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            elif key == ord('s'):
                fuser.calibrate_zero()
                for t in threads:
                    baseline_angles[t.cam_name] = current_angles[t.cam_name][stage_model] if not np.isnan(current_angles[t.cam_name][stage_model]) else 0.0
                print(f">>> {stage_model} 모델 및 DualCamKnobFuser의 영점(0도)이 초기화되었습니다.")
                
            if target_to_record is not None and not only_mp:
                for t in threads:
                    raw_angle = current_angles[t.cam_name][stage_model]
                    if not np.isnan(raw_angle):
                        angle = raw_angle - baseline_angles[t.cam_name]
                        snapshot_data[stage_model][target_to_record][t.cam_name] = angle
                    else:
                        snapshot_data[stage_model][target_to_record][t.cam_name] = np.nan
                print(f">>> {stage_model} 모델 - {target_to_record}도 스냅샷 측정 완료!")

        if quit_requested or only_mp:
            break

    print("\n측정을 종료합니다. 스레드를 닫습니다...")
    for t in threads:
        t.stop()
    for t in threads:
        t.join()
    if not args.headless:
        cv2.destroyAllWindows()

    if only_mp:
        print("✅ MediaPipe 실시간 스트리밍이 종료되었습니다.")
        return

    # ── CSV 데이터 생성 ──
    csv_rows = []
    for model in stages:
        for target in target_angles:
            row = {"Model": model, "TargetAngle": target}
            for cfg in cam_configs:
                cam = cfg['name']
                row[cam] = snapshot_data[model][target][cam]
            csv_rows.append(row)
            
    final_df = pd.DataFrame(csv_rows)
    final_df.to_csv("multicam_evaluation_snapshot.csv", index=False)
    print("✅ 데이터가 'multicam_evaluation_snapshot.csv'에 저장되었습니다.")

    # ── 통합 막대 그래프 그리기 ──
    try:
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 7))
        x = np.arange(len(target_angles))
        width = 0.25
        bar_colors = {"MP": "#1f77b4", "RTM": "#ff7f0e", "Frei": "#2ca02c"}
        
        multiplier = 0
        for model in stages:
            bars = []
            for target in target_angles:
                errors = []
                for cfg in cam_configs:
                    val = snapshot_data[model][target][cfg['name']]
                    if not np.isnan(val):
                        error = abs(abs(val) - target)
                        errors.append(error)
                        
                if errors:
                    bars.append(np.median(errors))
                else:
                    bars.append(np.nan)
                    
            offset = width * multiplier
            rects = plt.bar(x + offset, bars, width, 
                            label=model, 
                            color=bar_colors.get(model),
                            edgecolor='white', alpha=0.9)
            
            labels = ["Fail" if np.isnan(v) else f"{v:.1f}°" for v in bars]
            plt.bar_label(rects, labels=labels, padding=3, fontsize=10)
            multiplier += 1

        plt.axhline(0, color='black', linewidth=2)
        plt.title("측정값 오차 비교 (Measurement Error)", fontsize=16, fontweight='bold')
        plt.xlabel("Target Angle")
        plt.ylabel("Absolute Measurement Error (Degree)")
        plt.xticks(x + width, [f"{t}°" for t in target_angles])
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title="Model")
        plt.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
        
        ax = plt.gca()
        ax.set_ylim(0, max(20, ax.get_ylim()[1] * 1.1))
        
        plt.tight_layout()
        plt.savefig("multicam_evaluation_snapshot_error_bar.png", dpi=300)
        print("✅ 카메라 평균 오차 분석 그래프가 'multicam_evaluation_snapshot_error_bar.png'에 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ 그래프 생성 중 오류 (임베디드/헤드리스 환경 정상 동작 가능): {e}")


if __name__ == "__main__":
    main()
