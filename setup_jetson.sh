#!/bin/bash
# ==============================================================================
# setup_jetson.sh
# NVIDIA Jetson Nano 최적 성능 세팅 및 필수 환경 구성 스크립트
# ==============================================================================

echo "========================================================"
echo "🚀 Jetson Nano 듀얼 웹캠 트래커 환경 최적화 설정 시작"
echo "========================================================"

# 1. Jetson Nano 전력 및 클럭 최대 성능 모드(10W MAXN) 설정
echo "\n[1/4] 전력 모드를 10W 최대 성능(MAXN) 모드로 설정합니다..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo "✅ 10W 모드 및 최대 클럭 잠금 완료!"
else
    echo "⚠️ nvpmodel 명령을 찾을 수 없습니다. (Jetson 공식 OS 여부 확인 필요)"
fi

# 2. Swap 메모리 확인 및 4GB 생성 (OOM 킬 방지 필수)
echo "\n[2/4] Swap 메모리를 확인합니다..."
SWAP_TOTAL=$(free -m | awk '/Swap/ {print $2}')
if [ "$SWAP_TOTAL" -lt 2000 ]; then
    echo "⚠️ 현재 Swap이 부족합니다 (${SWAP_TOTAL}MB). 4GB Swap 파일을 생성합니다..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    # fstab에 등록하여 재부팅 후에도 유지
    if ! grep -q "/swapfile" /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    echo "✅ 4GB Swap 메모리 생성 및 활성화 완료!"
else
    echo "✅ 이미 충분한 Swap 메모리가 활성화되어 있습니다 (${SWAP_TOTAL}MB)."
fi

# 3. 비디오 디바이스(V4L2 웹캠) 접근 권한 부여
echo "\n[3/4] 현재 사용자($USER)에게 비디오 디바이스(webcam) 권한을 부여합니다..."
sudo usermod -aG video $USER
echo "✅ video 그룹 권한 추가 완료."

# 4. 필수 패키지 및 MediaPipe(ARM64 전용 휠) 설치
echo "\n[4/4] Python 필수 패키지 및 MediaPipe를 확인/설치합니다..."
sudo apt-get install -y python3-pip python3-dev curl unzip
python3 -m pip install --upgrade "pip<21.3"
python3 -m pip install "numpy>=1.19.4,<1.20.0" dataclasses

if ! python3 -c "import mediapipe" &> /dev/null; then
    echo "⚠️ MediaPipe가 없습니다. Jetson Nano(ARM64)용 빌드 휠을 다운로드하여 설치합니다..."
    cd /tmp
    curl -OL https://github.com/PINTO0309/mediapipe-bin/releases/download/v0.8.5/v0.8.5.zip
    unzip -o v0.8.5.zip -d mediapipe_tmp
    PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    WHL_FILE=$(find mediapipe_tmp -name "*${PY_VER}*.whl" | head -n 1)
    if [ -z "$WHL_FILE" ]; then
        WHL_FILE=$(find mediapipe_tmp -name "*.whl" | head -n 1)
    fi
    if [ -n "$WHL_FILE" ]; then
        pip3 install "$WHL_FILE"
        echo "✅ MediaPipe 설치 완료!"
    else
        echo "❌ 호환되는 MediaPipe wheel 파일을 찾지 못했습니다."
    fi
    rm -rf v0.8.5.zip mediapipe_tmp
    cd - > /dev/null
else
    echo "✅ 이미 MediaPipe가 정상 설치되어 있습니다."
fi

echo "\n========================================================"
echo "🎉 모든 최적화 설정이 완료되었습니다!"
echo "========================================================"
echo "실행 명령어 예시:"
echo "1) 단일 웹캠 실행 (화면 없이 최고 성능):"
echo "   python3 jetson_knob_tracker.py --cam 0 --target-hand Right --headless"
echo ""
echo "2) 듀얼 웹캠 실행 (교차 추론 + Headless):"
echo "   python3 jetson_dual_cam_tracker.py --cam1 0 --cam2 1 --interleave --headless"
echo "========================================================"
