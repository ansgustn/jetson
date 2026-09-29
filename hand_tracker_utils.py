# -*- coding: utf-8 -*-
"""
hand_tracker_utils.py
=====================
Jetson Nano 및 임베디드 환경을 위한 고정밀 손 추적, 스무딩, 다중 손 튐(간섭) 방지 유틸리티 모듈.

주요 기능:
1. OneEuroFilter / EMA: 랜드마크 및 각도 지터(떨림) 완벽 제거 및 지연 없는 반응.
2. RobustHandTracker:
   - Handedness(오른손/왼손) 잠금으로 다른 손의 난입 차단
   - 프레임 간 속도 벡터 기반 위치 예측(Motion Prediction)
   - 물리적 한계를 벗어나는 순간이동(Teleportation) 후보 거부
   - 가림 발생 시 즉각 타겟을 잃지 않는 관성 유지(Coasting)
3. calc_angle, pinch_ratio 등 핵심 측정 함수 및 각도 언래핑 내장.
"""

import math
import socket
import struct
import time
import numpy as np


class OneEuroFilter:
    """
    1€ Filter: 저속에서는 노이즈(지터)를 강력하게 제거하고, 
    고속 회전/이동 시에는 지연(Lag) 없이 부드럽게 반응하는 적응형 저역 통과 필터.
    """
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def filter(self, x, timestamp=None):
        if timestamp is None:
            timestamp = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = timestamp
            return x

        t_e = timestamp - self.t_prev
        if t_e <= 1e-5:
            return self.x_prev

        # 속도(미분) 추정 및 필터링
        dx = (x - self.x_prev) / t_e
        alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # 속도에 따라 차단 주파수 가변 조정
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._smoothing_factor(t_e, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


def calc_angle(pt1, pt2):
    """
    손목(pt1=0) -> 중지 MCP(pt2=9) 벡터 기준 각도 (-180 ~ +180도) 계산.
    Unreal Engine BP_Knob 좌표계와 일치: 사용자가 시계방향(오른쪽)으로 돌리면 양수 각도 증가.
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    return -math.degrees(math.atan2(-dy, dx))


def calc_robust_hand_angle(lms):
    """
    손목(0)-중지뿌리(9) 종축 벡터와 검지뿌리(5)-새끼뿌리(17) 횡축 너클 벡터를
    2D 투영 길이에 따른 SNR 기반으로 지능형 가중 합성합니다.
    
    손가락이 카메라 정면을 향하고 손목이 뒤에 오는 정면 원근 단축(Foreshortening)
    자세에서도 횡축 너클 바가 최대 모멘트 레버리지(지렛대)를 유지하므로,
    특이점(Gimbal lock/각도 와해) 없이 360도 완벽하고 안정적인 회전 각도를 산출합니다.
    """
    if lms is None or len(lms) < 18:
        return 0.0

    p0 = lms[0]   # Wrist
    p5 = lms[5]   # Index MCP
    p9 = lms[9]   # Middle MCP
    p17 = lms[17] # Pinky MCP

    # 1. 종축(Longitudinal) 벡터: 손목 -> 중지 뿌리
    ux = p9[0] - p0[0]
    uy = p9[1] - p0[1]
    len_u = math.hypot(ux, uy)

    # 2. 횡축(Transverse) 너클 벡터: 검지 뿌리 -> 새끼 뿌리
    vx = p17[0] - p5[0]
    vy = p17[1] - p5[1]
    len_v = math.hypot(vx, vy)

    if len_u < 1e-4 and len_v < 1e-4:
        return 0.0

    # 너클 벡터를 90도 회전시켜 종축과 동일한 진행 방향으로 정렬
    rot1_x, rot1_y = vy, -vx
    rot2_x, rot2_y = -vy, vx

    dot1 = ux * rot1_x + uy * rot1_y
    dot2 = ux * rot2_x + uy * rot2_y

    if dot1 >= dot2:
        v_perp_x, v_perp_y = rot1_x, rot1_y
    else:
        v_perp_x, v_perp_y = rot2_x, rot2_y

    # 정규화
    inv_u = 1.0 / max(1e-4, len_u)
    u_nx = ux * inv_u
    u_ny = uy * inv_u

    inv_v = 1.0 / max(1e-4, len_v)
    v_nx = v_perp_x * inv_v
    v_ny = v_perp_y * inv_v

    # 2D 투영 길이의 제곱(SNR)에 비례하여 가중 합성
    # 손이 정면을 향해 len_u가 짧아지면 너클(len_v)이 70~80% 이상 가중치를 주도
    w_u = len_u * len_u
    w_v = len_v * len_v
    total_w = w_u + w_v

    fused_x = (w_u * u_nx + w_v * v_nx) / total_w
    fused_y = (w_u * u_ny + w_v * v_ny) / total_w

    return -math.degrees(math.atan2(-fused_y, fused_x))



def unwrap_angle(current_raw, prev_angle):
    """
    각도가 +180도에서 -180도로 넘어갈 때 튀는 불연속성을 보정.
    """
    if prev_angle is None or np.isnan(prev_angle):
        return current_raw
    diff = (current_raw - prev_angle + 180.0) % 360.0 - 180.0
    return prev_angle + diff


def pinch_ratio_landmarks(pt_thumb, pt_index, pt_wrist, pt_middle, default_val=1.5):
    """
    엄지 끝 - 검지 끝 거리를 손목 - 중지관절 기준 길이로 정규화.
    0.2 ~ 0.35 : Grab (잡음) / 0.7 이상 : Release (놓음)
    """
    dist = math.hypot(pt_index[0] - pt_thumb[0], pt_index[1] - pt_thumb[1])
    ref = math.hypot(pt_middle[0] - pt_wrist[0], pt_middle[1] - pt_wrist[1])
    if ref < 1e-6:
        return default_val
    return dist / ref


class RobustHandTracker:
    """
    다중 손 등장 시 목표 손을 유지하고, 튐 현상(Teleportation/Swapping)을 원천 차단하는 지능형 단일 손 추적기.
    
    특징:
    - Handedness(오른손/왼손) 일관성 검사
    - 속도 벡터 기반 위치 예측(Kalman/Linear Motion Model)
    - 이상 점프(Max Jump Distance) 거부
    - 순간 가림 시 직전 각도/핀치값 유지(Coasting)
    - 1€ 필터를 통한 실시간 노이즈 제거
    """
    def __init__(self, target_handedness='Auto', max_lost_frames=12, max_jump_dist=120.0, is_mirrored=True):
        """
        :param target_handedness: 'Right', 'Left', 'Auto' (최초 감지 손 자동 고정), 'Any'
        :param max_lost_frames: 손을 놓쳤을 때 이전 상태를 유지(Coasting)할 최대 프레임 수
        :param max_jump_dist: 1프레임 동안 허용되는 최대 중심점 이동 거리 (픽셀 단위, 640x480 기준)
        :param is_mirrored: 카메라가 cv2.flip 등으로 좌우 반전되었는지 여부 (반전 시 Left/Right 라벨 반전 보정)
        """
        self.target_handedness = target_handedness
        self.is_mirrored = is_mirrored
        self.locked_handedness = None
        self.max_lost_frames = max_lost_frames
        self.max_jump_dist = max_jump_dist

        # 위치 및 모션 상태
        self.prev_center = None
        self.velocity = (0.0, 0.0)
        self.prev_size = None
        self.lost_count = 0

        # 필터 (미세 회전 조작에 즉각 반응하도록 최적화: 저속 데드밴드 제거)
        self.pos_filter_x = OneEuroFilter(min_cutoff=1.5, beta=0.08)
        self.pos_filter_y = OneEuroFilter(min_cutoff=1.5, beta=0.08)
        self.angle_filter = OneEuroFilter(min_cutoff=2.5, beta=0.20)
        self.pinch_filter = OneEuroFilter(min_cutoff=1.5, beta=0.02)
        # 전체 21개 랜드마크 좌표 지터 방지 필터 (1€ Filter)
        self.lm_filters = {
            idx: (OneEuroFilter(min_cutoff=2.0, beta=0.15), OneEuroFilter(min_cutoff=2.0, beta=0.15))
            for idx in range(21)
        }


        # 출력 캐시
        self.last_valid_angle = None
        self.last_valid_pinch = 1.5
        self.last_valid_landmarks = None
        self.is_tracking = False
        self.is_coasting = False
        self.current_confidence = 0.0
        self.current_size = 0.0

    def _get_expected_mediapipe_label(self, hand_side):
        """거울 반전(cv2.flip) 상태를 고려하여 MediaPipe가 출력할 것으로 예상되는 라벨을 반환합니다."""
        if not self.is_mirrored or hand_side not in ['Right', 'Left']:
            return hand_side
        # MediaPipe는 거울 반전 시 실제 왼손(Left)을 'Right'로, 실제 오른손(Right)을 'Left'로 판정함
        return 'Right' if hand_side == 'Left' else 'Left'

    def reset(self):
        """추적 상태 초기화"""
        self.prev_center = None
        self.velocity = (0.0, 0.0)
        self.prev_size = None
        self.lost_count = 0
        self.is_coasting = False
        self.current_confidence = 0.0
        self.current_size = 0.0
        if self.target_handedness == 'Auto':
            self.locked_handedness = None
        self.pos_filter_x.reset()
        self.pos_filter_y.reset()
        self.angle_filter.reset()
        self.pinch_filter.reset()
        for fx, fy in self.lm_filters.values():
            fx.reset()
            fy.reset()
        self.last_valid_angle = None
        self.last_valid_pinch = 1.5
        self.last_valid_landmarks = None
        self.is_tracking = False

    def get_tracking_quality(self):
        """
        추적 품질 점수 (0.0 ~ 1.0)를 산출합니다.
        손 미감지 = 0.0, 가림(Coasting) = 신뢰도 30% 감쇄, 크기 및 검출 스코어 종합.
        """
        if not self.is_tracking:
            return 0.0
        if getattr(self, 'is_coasting', False):
            return float(getattr(self, 'current_confidence', 0.5) * 0.3)
        size = getattr(self, 'current_size', 50.0)
        conf = getattr(self, 'current_confidence', 0.8)
        size_factor = min(1.0, max(0.2, size / 50.0))
        return float(conf * size_factor)

    def update_mediapipe(self, mp_result, img_w, img_h, timestamp=None):
        """
        MediaPipe Task(HandLandmarker) 결과로부터 가장 적합한 손 1개를 추적 및 필터링.
        
        :param mp_result: mp_detector.detect() 반환 객체
        :param img_w: 프레임 가로 픽셀
        :param img_h: 프레임 세로 픽셀
        :return: (selected_landmarks_pixel, angle, pinch, center, is_tracked)
        """
        if timestamp is None:
            timestamp = time.time()

        candidates = []
        if mp_result and mp_result.hand_landmarks:
            num_detected = len(mp_result.hand_landmarks)
            for i in range(num_detected):
                lms = mp_result.hand_landmarks[i]
                
                # Handedness 추출
                handedness_label = None
                handedness_score = 0.5
                if hasattr(mp_result, 'handedness') and i < len(mp_result.handedness):
                    cats = mp_result.handedness[i]
                    if cats:
                        handedness_label = cats[0].category_name
                        handedness_score = cats[0].score

                # 픽셀 좌표 변환
                wrist_x, wrist_y = lms[0].x * img_w, lms[0].y * img_h
                middle_x, middle_y = lms[9].x * img_w, lms[9].y * img_h
                center_x = (wrist_x + middle_x) / 2.0
                center_y = (wrist_y + middle_y) / 2.0
                hand_size = math.hypot(middle_x - wrist_x, middle_y - wrist_y)

                pixel_lms = [(lm.x * img_w, lm.y * img_h, lm.z * img_w) for lm in lms]

                candidates.append({
                    'center': (center_x, center_y),
                    'size': hand_size,
                    'handedness': handedness_label,
                    'handedness_score': handedness_score,
                    'pixel_landmarks': pixel_lms,
                    'confidence': handedness_score
                })

        return self._process_candidates(candidates, timestamp)

    def update_mediapipe_solutions(self, results, img_w, img_h, timestamp=None):
        """
        MediaPipe Solutions(mp.solutions.hands) 결과로부터 가장 적합한 손 1개를 추적 및 필터링.
        Jetson Nano(ARM64) 환경에서 권장되는 경량 API 지원.
        
        :param results: detector.process() 반환 객체
        :param img_w: 프레임 가로 픽셀
        :param img_h: 프레임 세로 픽셀
        :return: (selected_landmarks_pixel, angle, pinch, center, is_tracked)
        """
        if timestamp is None:
            timestamp = time.time()

        candidates = []
        if results and results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                lms = hand_landmarks.landmark
                wrist_x = lms[0].x * img_w
                wrist_y = lms[0].y * img_h
                middle_x = lms[9].x * img_w
                middle_y = lms[9].y * img_h
                center_x = (wrist_x + middle_x) / 2.0
                center_y = (wrist_y + middle_y) / 2.0
                hand_size = math.hypot(middle_x - wrist_x, middle_y - wrist_y)

                handedness_label = None
                handedness_score = 0.5
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness_label = results.multi_handedness[i].classification[0].label
                    handedness_score = results.multi_handedness[i].classification[0].score

                pixel_lms = [(lm.x * img_w, lm.y * img_h, lm.z * img_w) for lm in lms]
                candidates.append({
                    'center': (center_x, center_y),
                    'size': hand_size,
                    'handedness': handedness_label,
                    'handedness_score': handedness_score,
                    'pixel_landmarks': pixel_lms,
                    'confidence': handedness_score
                })

        return self._process_candidates(candidates, timestamp)

    def update_rtmpose(self, keypoints_all, scores_all, timestamp=None):
        """
        RTMPose 검출 결과(keypoints, scores)로부터 손 1개 추적.
        """
        if timestamp is None:
            timestamp = time.time()

        candidates = []
        if keypoints_all is not None and len(keypoints_all) > 0:
            for i in range(len(keypoints_all)):
                kps = keypoints_all[i]
                scs = scores_all[i]
                if scs[0] < 0.25 or scs[9] < 0.25:
                    continue

                wrist_x, wrist_y = kps[0][0], kps[0][1]
                middle_x, middle_y = kps[9][0], kps[9][1]
                center_x = (wrist_x + middle_x) / 2.0
                center_y = (wrist_y + middle_y) / 2.0
                hand_size = math.hypot(middle_x - wrist_x, middle_y - wrist_y)
                avg_score = float(np.mean(scs))

                pixel_lms = [(kps[j][0], kps[j][1], 0.0) for j in range(len(kps))]

                candidates.append({
                    'center': (center_x, center_y),
                    'size': hand_size,
                    'handedness': None,
                    'handedness_score': 0.5,
                    'pixel_landmarks': pixel_lms,
                    'confidence': avg_score
                })

        return self._process_candidates(candidates, timestamp)

    def _process_candidates(self, candidates, timestamp):
        """
        후보군 중 최적의 손을 선별하고 상태를 갱신하는 공통 코어 로직.
        """
        expected_label = self._get_expected_mediapipe_label(self.target_handedness)

        # 1. 기존 추적 대상이 없는 경우 (초기 획득)
        if self.prev_center is None:
            if not candidates:
                return None, np.nan, 1.5, None, False

            # Handedness 필터링 (엄격한 타겟 손 선별)
            valid_candidates = []
            for c in candidates:
                if expected_label in ['Right', 'Left']:
                    if c['handedness'] and c['handedness'] != expected_label and c['handedness_score'] > 0.6:
                        continue
                valid_candidates.append(c)

            # 타겟 손이 명시된 경우, 반대 손이 들어오면 추적을 시작하지 않고 대기
            if not valid_candidates:
                if self.target_handedness in ['Right', 'Left']:
                    return None, np.nan, 1.5, None, False
                valid_candidates = candidates

            # 크기 * 신뢰도가 가장 우세한 손 선택
            valid_candidates.sort(key=lambda c: c['size'] * c['confidence'], reverse=True)
            chosen = valid_candidates[0]

            # 상태 초기화
            self.prev_center = chosen['center']
            self.velocity = (0.0, 0.0)
            self.prev_size = chosen['size']
            self.lost_count = 0
            self.is_tracking = True

            if self.target_handedness == 'Auto':
                self.locked_handedness = chosen['handedness']
            elif self.target_handedness in ['Right', 'Left']:
                self.locked_handedness = expected_label

            return self._finalize_detection(chosen, timestamp)

        # 2. 기존 추적 대상이 있는 경우 (연속 추적 및 다른 손 배제)
        pred_x = self.prev_center[0] + self.velocity[0]
        pred_y = self.prev_center[1] + self.velocity[1]

        best_candidate = None
        best_cost = float('inf')

        for c in candidates:
            # A. Handedness 불일치 시 엄격 배제 (다른 손 난입 완전 차단)
            handedness_penalty = 0.0
            if self.locked_handedness and c['handedness']:
                if c['handedness'] != self.locked_handedness:
                    if c['handedness_score'] > 0.6:
                        continue
                    handedness_penalty = 1.0

            # B. 예측 위치와의 거리 계산
            dist = math.hypot(c['center'][0] - pred_x, c['center'][1] - pred_y)

            # C. 순간 이동(Teleport) 방지: 허용 반경을 초과하면 다른 손으로 판단하여 배제
            if dist > self.max_jump_dist:
                continue

            # D. 크기 변화량 비율
            size_diff = abs(c['size'] - self.prev_size) / max(self.prev_size, 1e-4)

            # E. 종합 비용 계산 (거리 비중 60%, 크기 안정성 25%, Handedness 15%)
            cost = (dist / self.max_jump_dist) * 0.6 + min(size_diff, 1.0) * 0.25 + handedness_penalty * 0.15
            if cost < best_cost:
                best_cost = cost
                best_candidate = c

        # 적합한 후보를 찾은 경우
        if best_candidate is not None and best_cost < 0.85:
            cx, cy = best_candidate['center']
            # 속도 벡터 EMA 업데이트
            vx = cx - self.prev_center[0]
            vy = cy - self.prev_center[1]
            self.velocity = (0.7 * vx + 0.3 * self.velocity[0], 0.7 * vy + 0.3 * self.velocity[1])
            self.prev_center = (cx, cy)
            self.prev_size = best_candidate['size']
            self.lost_count = 0
            self.is_tracking = True

            return self._finalize_detection(best_candidate, timestamp)

        # 적합한 후보를 찾지 못한 경우 (일시적 가림 or 다른 손만 감지됨)
        self.lost_count += 1
        if self.lost_count <= self.max_lost_frames:
            # Coasting (직전 관성 유지)
            self.is_coasting = True
            self.prev_center = (self.prev_center[0] + self.velocity[0] * 0.5,
                                self.prev_center[1] + self.velocity[1] * 0.5)
            self.velocity = (self.velocity[0] * 0.8, self.velocity[1] * 0.8)
            # 순간 가림(Occlusion) 시 4프레임 동안은 직전 핀치값을 그대로 유지하여 잡음 풀림 방지
            prev_p = self.last_valid_pinch if self.last_valid_pinch is not None else 0.5
            if self.lost_count <= 4:
                coast_pinch = prev_p
            else:
                coast_pinch = min(1.5, prev_p + 0.1 * (self.lost_count - 4))
            return self.last_valid_landmarks, self.last_valid_angle, coast_pinch, self.prev_center, True
        else:
            # 완전히 놓침 -> 초기화
            self.reset()
            return None, np.nan, 1.5, None, False

    def _finalize_detection(self, chosen, timestamp):
        """스무딩 필터 적용 및 회전각, 핀치값 계산"""
        lms = chosen['pixel_landmarks']
        # 전체 21개 랜드마크 필터링으로 손가락 떨림 및 좌표 튐 완벽 제거
        smooth_lms = []
        for idx in range(len(lms)):
            if idx not in self.lm_filters:
                self.lm_filters[idx] = (OneEuroFilter(min_cutoff=1.5, beta=0.05), OneEuroFilter(min_cutoff=1.5, beta=0.05))
            fx, fy = self.lm_filters[idx]
            sx = fx.filter(lms[idx][0], timestamp)
            sy = fy.filter(lms[idx][1], timestamp)
            smooth_lms.append((sx, sy, lms[idx][2]))

        pt_wrist = smooth_lms[0]
        pt_thumb = smooth_lms[4]
        pt_index = smooth_lms[8]
        pt_middle = smooth_lms[9]

        self.is_coasting = False
        self.current_confidence = float(chosen.get('confidence', 0.8))
        self.current_size = float(chosen.get('size', 50.0))

        # 중심점 스무딩
        smooth_cx = self.pos_filter_x.filter(chosen['center'][0], timestamp)
        smooth_cy = self.pos_filter_y.filter(chosen['center'][1], timestamp)
        center = (smooth_cx, smooth_cy)

        # 각도 계산 및 언래핑 + 1€ 필터 스무딩 (종축+횡축 너클 지능형 합성)
        raw_angle = calc_robust_hand_angle(smooth_lms)
        unwrapped = unwrap_angle(raw_angle, self.last_valid_angle)
        smooth_angle = self.angle_filter.filter(unwrapped, timestamp)

        # 핀치 비율 계산 및 필터 스무딩
        raw_pinch = pinch_ratio_landmarks(
            (pt_thumb[0], pt_thumb[1]), (pt_index[0], pt_index[1]),
            (pt_wrist[0], pt_wrist[1]), (pt_middle[0], pt_middle[1])
        )
        smooth_pinch = self.pinch_filter.filter(raw_pinch, timestamp)

        self.last_valid_angle = smooth_angle
        self.last_valid_pinch = smooth_pinch
        self.last_valid_landmarks = smooth_lms

        return smooth_lms, smooth_angle, smooth_pinch, center, True


class DualCamKnobFuser:
    """
    듀얼 카메라(2대)의 [각도, 핀치, 신뢰도]를 지능적으로 융합하는 센서 퓨전 엔진.
    
    핵심 원리:
    1. 단일 합의(Consensus) OSC 스트리밍:
       - 2대가 각각 다른 메시지를 쏘며 생기는 잡음-놓음 무한 루프(핑퐁)를 원천 차단.
       - 2대의 시야를 하나로 통합하여 [fused_angle, fused_pinch] 1개만 전송.
    2. 후한 그랩 인식 & 비대칭 핀치 융합:
       - 한 대는 그랩(0.40)이고 다른 한 대는 시야각/가림 왜곡으로 0.72일 때,
         손가락 간격이 더 좁은(그랩에 더 가까운) 카메라의 신호를 채택(p_eval = min(p1, p2)).
    3. 슈미트 트리거 / 양방향 디바운스 (Chattering Free):
       - RELEASED -> GRABBED: p_eval <= grab_threshold(0.70) 2프레임 연속 시 진입.
       - GRABBED -> RELEASED: p_eval > release_threshold(0.70) 2프레임 연속 시 복귀.
       - 잡은 상태(Grabbed) 중에는 언리얼이 절대 놓치지 않도록 0.20 강제 락(Lock).
       - 놓은 상태(Released) 중에는 확실한 릴리즈를 위해 1.50 전송.
    4. 360도 연속 각도 융합 & 미세 조정(Fine Tuning) 반응성:
       - 일시적 1~2프레임 손가락 가림/블러 시에도 회전 각도 기억을 유지하여 '돌아가다 멈춤' 현상 원천 차단.
       - 두 카메라 시야각 차이에 의한 회전 상쇄를 방지하고, 회전 평면을 명확히 보는 카메라의 델타를 우선 반영.
       - 저속 회전 시의 필터 데드밴드를 제거하여 0.2도 단위의 미세 조작에도 즉각 반응.
    """
    def __init__(self, grab_threshold=0.70, release_threshold=0.70, invert_angle=False, angle_gain=1.0):
        self.grab_threshold = float(grab_threshold)
        self.release_threshold = float(release_threshold)
        self.invert_angle = bool(invert_angle)
        self.angle_gain = float(angle_gain)

        # 상태 머신 (슈미트 트리거 + 양방향 2프레임 디바운스)
        self.is_grabbed = False
        self.grab_confirm_frames = 0
        self.release_confirm_frames = 0

        # 각도 융합 상태
        self.fused_angle = 0.0
        self.prev_cam1_angle = None
        self.prev_cam2_angle = None
        self.lost_cam1_frames = 0
        self.lost_cam2_frames = 0
        self.pinch_filter = OneEuroFilter(min_cutoff=1.5, beta=0.04)

    def reset(self):
        self.is_grabbed = False
        self.grab_confirm_frames = 0
        self.release_confirm_frames = 0
        self.prev_cam1_angle = None
        self.prev_cam2_angle = None
        self.lost_cam1_frames = 0
        self.lost_cam2_frames = 0
        self.pinch_filter.reset()

    def calibrate_zero(self):
        """현재 회전 각도를 영점(0도)으로 캘리브레이션합니다. ('s' 키 대응)"""
        self.fused_angle = 0.0

    def update(self, c1_data, c2_data, timestamp=None):
        """
        :param c1_data: dict {'tracked': bool, 'angle': float, 'pinch': float, 'quality': float}
        :param c2_data: dict {'tracked': bool, 'angle': float, 'pinch': float, 'quality': float}
        :return: (send_angle, send_pinch, is_grabbed, is_tracked, debug_info)
        """
        if timestamp is None:
            timestamp = time.time()

        t1 = bool(c1_data.get('tracked', False))
        a1 = c1_data.get('angle', np.nan)
        p1 = c1_data.get('pinch', 1.5)
        q1 = float(c1_data.get('quality', 0.5))

        t2 = bool(c2_data.get('tracked', False))
        a2 = c2_data.get('angle', np.nan)
        p2 = c2_data.get('pinch', 1.5)
        q2 = float(c2_data.get('quality', 0.5))

        # 1. 두 카메라 모두 손을 완전히 놓친 경우 -> 즉시 완전 릴리즈(1.5)
        if not t1 and not t2:
            self.lost_cam1_frames += 1
            self.lost_cam2_frames += 1
            if self.lost_cam1_frames > 6:
                self.prev_cam1_angle = None
            if self.lost_cam2_frames > 6:
                self.prev_cam2_angle = None
            self.is_grabbed = False
            self.grab_confirm_frames = 0
            self.release_confirm_frames = 0
            return float(self.fused_angle % 360.0), 1.5, False, False, "NO_HAND"

        # 2. 핀치(Pinch) 및 그랩/놓음 지능형 융합 (각도 누적 전에 먼저 확정)
        was_grabbed = self.is_grabbed
        active_pinches = []
        if t1 and not np.isnan(p1) and p1 < 3.0:
            active_pinches.append((float(p1), q1, 'Cam1'))
        if t2 and not np.isnan(p2) and p2 < 3.0:
            active_pinches.append((float(p2), q2, 'Cam2'))

        if not active_pinches:
            candidate_pinch = 1.5
            self.is_grabbed = False
            self.grab_confirm_frames = 0
            self.release_confirm_frames = 0
            chosen_cam = "None"
        else:
            if len(active_pinches) == 1:
                p_eval, q_eval, chosen_cam = active_pinches[0]
            else:
                p_a, q_a, name_a = active_pinches[0]
                p_b, q_b, name_b = active_pinches[1]
                # 두 카메라 중 손가락 간격이 더 좁은(그랩에 더 가까운) 카메라 값을 평가 기준으로 채택
                if p_a <= p_b:
                    p_eval, chosen_cam = p_a, name_a
                else:
                    p_eval, chosen_cam = p_b, name_b

            candidate_pinch = p_eval

            # 슈미트 트리거(히스테리시스) + 양방향 2프레임 디바운스
            if not self.is_grabbed:
                self.release_confirm_frames = 0
                if p_eval <= self.grab_threshold:
                    self.grab_confirm_frames += 1
                    if self.grab_confirm_frames >= 2:
                        self.is_grabbed = True
                        self.grab_confirm_frames = 0
                else:
                    self.grab_confirm_frames = 0
            else:
                self.grab_confirm_frames = 0
                is_rel = (p_eval > self.release_threshold) if self.release_threshold <= self.grab_threshold else (p_eval >= self.release_threshold)
                if is_rel:
                    self.release_confirm_frames += 1
                    if self.release_confirm_frames >= 2:
                        self.is_grabbed = False
                        self.release_confirm_frames = 0
                else:
                    self.release_confirm_frames = 0

        # 그랩 최초 진입 시점 판별
        grab_just_started = (self.is_grabbed and not was_grabbed)

        # 3. 각도(Delta) 지능형 연속 융합
        delta1, delta2 = 0.0, 0.0
        has_delta1, has_delta2 = False, False
        w1, w2 = 0.0, 0.0

        if t1 and not np.isnan(a1):
            if self.prev_cam1_angle is not None and self.lost_cam1_frames <= 6:
                diff1 = (a1 - self.prev_cam1_angle + 180.0) % 360.0 - 180.0
                if abs(diff1) < 60.0:  # 이상 순간이동 점프 배제
                    delta1 = diff1
                    has_delta1 = True
                    w1 = max(0.02, abs(delta1)) * max(0.1, q1)
            self.prev_cam1_angle = a1
            self.lost_cam1_frames = 0
        else:
            self.lost_cam1_frames += 1
            if self.lost_cam1_frames > 6:
                self.prev_cam1_angle = None

        if t2 and not np.isnan(a2):
            if self.prev_cam2_angle is not None and self.lost_cam2_frames <= 6:
                diff2 = (a2 - self.prev_cam2_angle + 180.0) % 360.0 - 180.0
                if abs(diff2) < 60.0:
                    delta2 = diff2
                    has_delta2 = True
                    w2 = max(0.02, abs(delta2)) * max(0.1, q2)
            self.prev_cam2_angle = a2
            self.lost_cam2_frames = 0
        else:
            self.lost_cam2_frames += 1
            if self.lost_cam2_frames > 6:
                self.prev_cam2_angle = None

        # 가중 델타 합성
        fused_delta = 0.0
        if has_delta1 and has_delta2:
            if delta1 * delta2 > 0:
                # 같은 방향: 회전 평면을 더 정면에서 보고 있는(delta 절대값이 큰) 카메라를 적극 반영
                if abs(delta1) >= abs(delta2):
                    fused_delta = 0.75 * delta1 + 0.25 * delta2
                else:
                    fused_delta = 0.25 * delta1 + 0.75 * delta2
            else:
                # 방향이 상충할 경우: 한 카메라가 뚜렷한 회전(>=0.5도)이고 다른 카메라는 미세 노이즈(<0.3도)면 우세 카메라 채택
                if abs(delta1) >= 0.5 and abs(delta2) < 0.3:
                    fused_delta = delta1
                elif abs(delta2) >= 0.5 and abs(delta1) < 0.3:
                    fused_delta = delta2
                else:
                    fused_delta = (w1 * delta1 + w2 * delta2) / (w1 + w2)
        elif has_delta1:
            fused_delta = delta1
        elif has_delta2:
            fused_delta = delta2

        # 4. 회전 각도 누적 (★ 핵심: 그랩 상태에서만 회전 누적 & 그랩 진입 시점 영점 앵커링 ★)
        if grab_just_started:
            # 그랩이 체결된 바로 그 프레임: 공중 이동 및 핀치 집게 동작 시 손목 역방향 비틀림(-5~-15도) 노이즈 원천 폐기!
            self.prev_cam1_angle = a1
            self.prev_cam2_angle = a2
            fused_delta = 0.0
        elif self.is_grabbed:
            # 잡고 있는 동안에만 노브 회전 각도 누적!
            # 서브픽셀 떨림 데드밴드 (0.12도 미만의 정지 미세 떨림은 0 처리하여 반대방향 튐 차단)
            if abs(fused_delta) < 0.12:
                fused_delta = 0.0

            if self.invert_angle:
                fused_delta = -fused_delta
            fused_delta *= self.angle_gain
            self.fused_angle += fused_delta
        else:
            # 놓은 상태(공중 이동 중): 노브는 제자리에 고정(Lock)
            fused_delta = 0.0

        send_angle = float(self.fused_angle % 360.0)
        send_pinch = 0.20 if self.is_grabbed else 1.50

        debug_info = f"[{chosen_cam}|{'GRAB' if self.is_grabbed else 'REL'}|p={candidate_pinch:.2f}|d={fused_delta:+5.1f}]"
        return send_angle, float(send_pinch), self.is_grabbed, True, debug_info


class SimpleUDPClient:
    """
    Python 표준 라이브러리(socket, struct)만 사용하여
    외부 패키지 의존성 없이 Python 3.6~3.12 전 버전에서 초경량으로 구동되는 독립형 OSC UDP 클라이언트.
    """
    def __init__(self, ip: str, port: int):
        self.ip = str(ip)
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @staticmethod
    def _pad(data: bytes) -> bytes:
        pad_len = (4 - (len(data) % 4)) % 4
        return data + (b'\x00' * pad_len)

    def send_message(self, address: str, values) -> None:
        try:
            addr_bytes = self._pad(address.encode('utf-8') + b'\x00')
            if not isinstance(values, (list, tuple)):
                values = [values]

            tag_chars = []
            val_bytes_list = []
            for v in values:
                if isinstance(v, (int, float, np.floating, np.integer)):
                    tag_chars.append('f')
                    val_bytes_list.append(struct.pack('>f', float(v)))
                elif isinstance(v, str):
                    tag_chars.append('s')
                    val_bytes_list.append(self._pad(v.encode('utf-8') + b'\x00'))
                else:
                    tag_chars.append('f')
                    val_bytes_list.append(struct.pack('>f', float(v)))

            tags = ',' + ''.join(tag_chars)
            tag_bytes = self._pad(tags.encode('utf-8') + b'\x00')
            payload = addr_bytes + tag_bytes + b''.join(val_bytes_list)

            self.sock.sendto(payload, (self.ip, self.port))
        except Exception:
            pass

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
