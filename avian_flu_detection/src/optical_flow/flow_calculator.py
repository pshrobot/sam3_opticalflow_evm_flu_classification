"""
Optical Flow Calculator
마스크 영역 내에서만 Optical Flow를 계산하는 모듈
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional


class OpticalFlowCalculator:
    """Optical Flow 계산기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Optical Flow 설정
                - method: 'farneback' 또는 'lucas_kanade'
                - farneback: Farneback 알고리즘 파라미터
        """
        config = config or {}
        self.method = config.get('method', 'farneback')

        # Farneback 파라미터
        fb_config = config.get('farneback', {})
        self.pyr_scale = fb_config.get('pyr_scale', 0.5)
        self.levels = fb_config.get('levels', 3)
        self.winsize = fb_config.get('winsize', 15)
        self.iterations = fb_config.get('iterations', 3)
        self.poly_n = fb_config.get('poly_n', 5)
        self.poly_sigma = fb_config.get('poly_sigma', 1.2)
        self.flags = fb_config.get('flags', 0)

        # 이전 프레임 저장 (스트리밍 모드용)
        self.prev_gray: Optional[np.ndarray] = None

    def compute_flow(self, frame_t: np.ndarray,
                     frame_t1: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     weight_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        두 프레임 간 Optical Flow 계산

        Args:
            frame_t: 이전 프레임 (BGR 또는 Grayscale)
            frame_t1: 현재 프레임 (BGR 또는 Grayscale)
            mask: 유효 영역 마스크 (선택)
            weight_map: 거리 기반 가중치 (선택)

        Returns:
            Dict containing:
                - flow: 전체 flow 배열 (H, W, 2)
                - flow_x: x방향 flow
                - flow_y: y방향 flow
                - magnitude: flow 크기
                - angle: flow 방향 (라디안)
                - masked_magnitude: 마스크 적용된 크기
                - weighted_magnitude: 가중치 적용된 크기
                - mask: 사용된 마스크
        """
        # 그레이스케일 변환
        if len(frame_t.shape) == 3:
            gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
        else:
            gray_t = frame_t

        if len(frame_t1.shape) == 3:
            gray_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
        else:
            gray_t1 = frame_t1

        # Optical Flow 계산
        if self.method == 'farneback':
            flow = self._compute_farneback(gray_t, gray_t1)
        else:
            flow = self._compute_lucas_kanade(gray_t, gray_t1)

        # Flow 분해
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        # 크기와 방향 계산
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)

        # 마스크 적용
        if mask is not None:
            masked_magnitude = magnitude * mask
        else:
            masked_magnitude = magnitude.copy()

        # 가중치 적용
        if weight_map is not None:
            weighted_magnitude = masked_magnitude * weight_map
        else:
            weighted_magnitude = masked_magnitude.copy()

        return {
            'flow': flow,
            'flow_x': flow_x,
            'flow_y': flow_y,
            'magnitude': magnitude,
            'angle': angle,
            'masked_magnitude': masked_magnitude,
            'weighted_magnitude': weighted_magnitude,
            'mask': mask
        }

    def _compute_farneback(self, gray_t: np.ndarray,
                           gray_t1: np.ndarray) -> np.ndarray:
        """
        Farneback 방식 Optical Flow 계산

        Args:
            gray_t: 이전 프레임 (Grayscale)
            gray_t1: 현재 프레임 (Grayscale)

        Returns:
            flow: Optical Flow 배열 (H, W, 2)
        """
        flow = cv2.calcOpticalFlowFarneback(
            gray_t, gray_t1, None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags
        )
        return flow

    def _compute_lucas_kanade(self, gray_t: np.ndarray,
                              gray_t1: np.ndarray) -> np.ndarray:
        """
        Lucas-Kanade 방식 (Dense) Optical Flow 계산
        실제로는 Dense LK가 아닌 Farneback으로 대체

        Args:
            gray_t: 이전 프레임
            gray_t1: 현재 프레임

        Returns:
            flow: Optical Flow 배열
        """
        # Dense Lucas-Kanade를 위해 기본 Farneback 사용
        flow = cv2.calcOpticalFlowFarneback(
            gray_t, gray_t1, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def compute_flow_incremental(self, frame: np.ndarray,
                                 mask: Optional[np.ndarray] = None,
                                 weight_map: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        스트리밍 방식으로 Flow 계산 (이전 프레임 자동 관리)

        Args:
            frame: 현재 프레임 (BGR)
            mask: 유효 영역 마스크
            weight_map: 가중치 맵

        Returns:
            Flow 결과 또는 None (첫 프레임인 경우)
        """
        # 그레이스케일 변환
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 첫 프레임인 경우
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Flow 계산
        result = self.compute_flow(
            self.prev_gray,
            gray,
            mask,
            weight_map
        )

        # 이전 프레임 업데이트
        self.prev_gray = gray

        return result

    def reset(self):
        """이전 프레임 초기화"""
        self.prev_gray = None

    def visualize_flow(self, flow: np.ndarray,
                       frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optical Flow를 HSV 컬러로 시각화

        Args:
            flow: Optical Flow 배열 (H, W, 2)
            frame: 배경 프레임 (선택, BGR)

        Returns:
            시각화된 이미지 (BGR)
        """
        # 크기와 방향 계산
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # HSV 이미지 생성
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: 방향
        hsv[..., 1] = 255  # Saturation: 최대
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: 크기

        # BGR로 변환
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 배경 프레임과 합성
        if frame is not None:
            result = cv2.addWeighted(frame, 0.5, flow_bgr, 0.5, 0)
        else:
            result = flow_bgr

        return result

    def draw_flow_arrows(self, frame: np.ndarray,
                         flow: np.ndarray,
                         mask: Optional[np.ndarray] = None,
                         step: int = 20,
                         color: tuple = (0, 255, 0),
                         thickness: int = 1) -> np.ndarray:
        """
        Optical Flow를 화살표로 시각화

        Args:
            frame: 배경 프레임 (BGR)
            flow: Optical Flow 배열
            mask: 마스크 (선택)
            step: 화살표 간격
            color: 화살표 색상 (BGR)
            thickness: 화살표 두께

        Returns:
            화살표가 그려진 프레임
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        for y in range(0, h, step):
            for x in range(0, w, step):
                # 마스크 체크
                if mask is not None and mask[y, x] == 0:
                    continue

                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx**2 + fy**2)

                # 최소 크기 필터
                if magnitude < 0.5:
                    continue

                # 화살표 그리기
                end_x = int(x + fx * 2)
                end_y = int(y + fy * 2)
                cv2.arrowedLine(result, (x, y), (end_x, end_y),
                                color, thickness, tipLength=0.3)

        return result
