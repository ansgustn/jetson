# Jetson Nano Real-Time Hand Tracking & Knob Interaction

> **NVIDIA Jetson Nano(ARM Cortex-A57 4코어, 4GB RAM) 전용**
> 초경량 실시간 손 추적, 지능형 센서 융합 및 언리얼 엔진(Unreal Engine BP_Knob) OSC 연동 시스템

---

## 📌 개요 (Overview)

본 프로젝트는 임베디드 엣지 디바이스(NVIDIA Jetson Nano)의 제한된 컴퓨팅 자원 환경에서 **단일 웹캠/CSI 카메라 또는 듀얼 웹캠(2대)**을 통하여 사용자의 손동작(Pinch/Grab)과 손목 회전(Rotation)을 실시간으로 추적하고, **OSC(Open Sound Control over UDP)**를 통해 언리얼 엔진(Unreal Engine 5 BP_Knob)에 저지연(<2ms) 30 FPS로 전송하는 비전 인터랙션 프레임워크입니다.

자세한 시스템 개념도 및 전체 아키텍처 명세는 [`CONCEPT_AND_ARCHITECTURE.txt`](./CONCEPT_AND_ARCHITECTURE.txt)를 참조하세요.

---

## ✨ 핵심 기능 및 기술 혁신 (Key Features)

1. **지능형 단일 합의 듀얼 카메라 융합 (`DualCamKnobFuser`)**
   - 두 대의 카메라가 독립적으로 신호를 쏘면서 발생하는 "잡음-놓음" 무한 핑퐁(채터링) 원천 차단
   - 손가락 간격이 더 좁은 카메라의 신호를 채택하는 **비대칭 핀치 융합 (`min(p1, p2)`)**
   - **슈미트 트리거(Schmitt Trigger)** 및 **양방향 2프레임 디바운스** 적용
   - 잡고 있는 동안 핀치값 `0.20` 강제 잠금(Lock), 놓았을 때 `1.50` 전송

2. **360도 연속 각도 언래핑 및 정면 왜곡 극복 (`calc_robust_hand_angle`)**
   - 손목(0)-중지뿌리(9) 종축 벡터와 검지뿌리(5)-새끼뿌리(17) 횡축 너클 벡터를 직교 정렬
   - 2D 투영 길이 제곱(SNR)에 비례하여 지능형 가중 합성
   - 카메라 정면을 향하는 단축 자세(Foreshortening)에서도 짐벌락 없이 360도 안정적인 각도 산출

3. **그랩 시작 순간 영점 앵커링 (Anchor on Grab Onset)**
   - 핀치를 쥐는 순간 손가락 근육 수축으로 인한 손목 역방향 비틀림(-5° ~ -15°) 노이즈 자동 폐기
   - 잡은 상태에서만 각도 누적, 서브픽셀 데드밴드(0.12° 미만 떨림 억제) 적용

4. **1-프레임 교차 추론 스케줄링 (`--interleave`)**
   - Jetson Nano 듀얼 카메라 동시 추론 시 프레임 드랍(12~14 FPS)을 해결
   - 홀수/짝수 프레임 교차 추론 및 캐시 재사용으로 **25~30 FPS** 고속 성능 확보

5. **초경량 순수 파이썬 OSC 클라이언트 (`SimpleUDPClient`)**
   - 외부 라이브러리 의존성 없이 표준 `socket`과 `struct`만으로 Big-Endian 바이너리 OSC 스트림 전송

---

## 🏗 시스템 개념도 (Conceptual Diagram)

```text
  [ 사용자 조작 ]                  [ Jetson Nano 임베디드 엣지 ]               [ 수신 워크스테이션 ]
 +------------------+           +-------------------------------+          +--------------------+
 |                  |           |                               |          |                    |
 |   사용자 손 동작 |           |   NVIDIA Jetson Nano (4GB)    |          |   Unreal Engine    |
 |  (Pinch / Twist) |           |                               |          |     BP_Knob        |
 |                  |           |   +-----------------------+   |          |                    |
 |  +------------+  |  시야각 1 |   |  비동기 카메라 캡처   |   |          |  +--------------+  |
 |  |  카메라 1  |=== === === =>|   | (Threaded Camera 1)   |   |          |  | OSC Receiver |  |
 |  | (Cam Index)|  |           |   +-----------+-----------+   |          |  |  (UDP:8000)  |  |
 |  +------------+  |           |               |               |          |  +-------+------+  |
 |                  |           |               v               |          |          |         |
 |                  |           |   +-----------------------+   |          |          v         |
 |                  |           |   |   MediaPipe Vision    |   |          |  +--------------+  |
 |                  |           |   |  - 21 Hand Keypoints  |   |          |  | 노브 액터    |  |
 |                  |           |   |  - 1-Frame Interleave |   |          |  | 회전 / 핀치  |  |
 |                  |           |   +-----------+-----------+   |          |  | 제어 반영    |  |
 |  +------------+  |  시야각 2 |               |               |          |  +--------------+  |
 |  |  카메라 2  |=== === === =>|               v               |          |                    |
 |  | (Cam Index)|  |           |   +-----------------------+   |  OSC/UDP |                    |
 |  +------------+  |           |   | 지능형 융합 및 필터   |   |  패킷    |                    |
 |                  |           |   | - RobustHandTracker   |=== === === =>|                    |
 |                  |           |   | - DualCamKnobFuser    |   | (/media  |                    |
 |                  |           |   | - 1-Euro Filter       |   |  pipe/   |                    |
 |                  |           |   +-----------+-----------+   |  knob)   |                    |
 |                  |           |               |               |          |                    |
 |                  |           |               v               |          |                    |
 |                  |           |   +-----------------------+   |          |                    |
 |                  |           |   |    SimpleUDPClient    |   |          |                    |
 |                  |           |   |   (No-Dependency OSC) |   |          |                    |
 |                  |           |   +-----------------------+   |          |                    |
 +------------------+           +-------------------------------+          +--------------------+
```

---

## 🚀 빠른 시작 (Getting Started)

### 1. Jetson Nano 환경 최적화 및 설치
Jetson 보드 터미널에서 다음 스크립트를 실행하여 10W 최대 성능 잠금, 4GB Swap 생성 및 패키지를 설치합니다:
```bash
chmod +x setup_jetson.sh
./setup_jetson.sh
```

### 2. 실행 명령어

#### (A) 듀얼 웹캠 실행 (추천 / 최고 성능)
```bash
python3 jetson_dual_cam_tracker.py --cam1 0 --cam2 1 --interleave --headless
```

#### (B) 단일 웹캠 실행 (테스트용)
```bash
python3 jetson_knob_tracker.py --cam 0 --target-hand Right --headless
```

#### (C) CSI 카메라(GStreamer) 실행
```bash
python3 jetson_knob_tracker.py --cam 0 --csi --headless
```

---

## 📡 OSC 통신 프로토콜 규격 (OSC Protocol)

- **기본 IP**: `192.168.137.1` (명령행 인자 `--ip`로 변경 가능)
- **기본 포트**: `8000` (명령행 인자 `--port`로 변경 가능)
- **통합 주소**: `/mediapipe/knob/angle`
- **전송 데이터 형식**: `[float angle, float pinch]`
  - `angle`: `0.0` ~ `359.9` (시계 방향 회전 시 증가, 언리얼 좌표계 일치)
  - `pinch`: 잡았을 때 `0.20` (GRAB), 놓았을 때 `1.50` (RELEASE)

---

## 📂 파일 구조 (File Structure)

```text
├── CONCEPT_AND_ARCHITECTURE.txt  # 시스템 개념도 및 전체 아키텍처 상세 문서
├── jetson_dual_cam_tracker.py    # 듀얼 웹캠 실시간 트래커 & 센서 융합 메인 스크립트
├── jetson_knob_tracker.py        # 단일 웹캠/CSI 초경량 실시간 트래커 스크립트
├── hand_tracker_utils.py         # 1€ 필터, RobustTracker, DualCamKnobFuser 등 핵심 알고리즘
├── evaluate_rotation_models.py   # 다중 회전 추론 모델 성능 평가 및 벤치마크 도구
├── hand_landmarker.task          # MediaPipe Tasks API 모델 바이너리
├── setup_jetson.sh               # Jetson 보드 환경 구성 및 최적화 쉘 스크립트
├── requirements_jetson.txt       # Jetson 환경 필수 패키지 목록
└── pythonosc/                    # 내장형 OSC 패키지 폴더
```

---

## 📄 라이선스 (License)

본 프로젝트는 자유롭게 수정 및 배포가 가능합니다.
