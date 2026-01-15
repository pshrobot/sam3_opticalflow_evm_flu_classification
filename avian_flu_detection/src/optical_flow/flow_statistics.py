"""
Flow Statistics Extractor
Optical Flow에서 통계를 추출하는 모듈
"""

import numpy as np
from typing import Dict, Any, Optional, List


class FlowStatisticsExtractor:
    """Optical Flow 통계 추출기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 통계 설정
                - global: 전역 통계 설정
                - regional: 영역별 통계 설정
        """
        config = config or {}

        # 전역 통계 설정
        global_config = config.get('global', {})
        self.compute_mean = global_config.get('compute_mean', True)
        self.compute_variance = global_config.get('compute_variance', True)
        self.compute_skewness = global_config.get('compute_skewness', True)
        self.compute_kurtosis = global_config.get('compute_kurtosis', True)

        # 영역별 통계 설정
        regional_config = config.get('regional', {})
        self.regional_enabled = regional_config.get('enabled', True)
        self.grid_rows = regional_config.get('grid_rows', 4)
        self.grid_cols = regional_config.get('grid_cols', 4)

    def extract_statistics(self, flow_result: Dict[str, Any],
                           mask_stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Flow 결과에서 통계 추출

        Args:
            flow_result: OpticalFlowCalculator의 출력
            mask_stats: 마스크 형태 통계

        Returns:
            Dict containing all statistics
        """
        stats = {}

        # 마스크 적용된 magnitude 가져오기
        magnitude = flow_result.get('weighted_magnitude', flow_result['magnitude'])
        mask = flow_result.get('mask')

        # 유효 픽셀 추출
        if mask is not None:
            valid_pixels = magnitude[mask > 0]
        else:
            valid_pixels = magnitude.flatten()

        # 전역 통계 계산
        stats['global'] = self._compute_global_statistics(valid_pixels)

        # 면적 정규화된 활동 지수
        mask_area = mask_stats.get('area', 1)
        if mask_area > 0:
            stats['activity_index'] = float(np.sum(magnitude)) / mask_area
        else:
            stats['activity_index'] = 0.0

        # 영역별 통계 계산
        if self.regional_enabled:
            stats['regional'] = self._compute_regional_statistics(
                flow_result['magnitude'], mask
            )

        # 마스크 형태 통계 추가
        stats['mask_shape'] = mask_stats

        # Flow 방향 분포
        angle = flow_result.get('angle')
        if angle is not None:
            if mask is not None:
                valid_angles = angle[mask > 0]
            else:
                valid_angles = angle.flatten()
            stats['direction'] = self._compute_direction_statistics(valid_angles)

        return stats

    def _compute_global_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """
        전역 통계 계산

        Args:
            values: 유효 픽셀 값들

        Returns:
            전역 통계 딕셔너리
        """
        if len(values) == 0:
            return {
                'mean': 0.0,
                'variance': 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'median': 0.0,
                'max': 0.0,
                'min': 0.0,
                'count': 0,
                'sum': 0.0
            }

        stats = {}

        # 기본 통계
        stats['mean'] = float(np.mean(values))
        stats['variance'] = float(np.var(values))
        stats['std'] = float(np.std(values))
        stats['median'] = float(np.median(values))
        stats['max'] = float(np.max(values))
        stats['min'] = float(np.min(values))
        stats['count'] = int(len(values))
        stats['sum'] = float(np.sum(values))

        # 왜도 (Skewness)
        if self.compute_skewness and stats['std'] > 0:
            stats['skewness'] = float(
                np.mean(((values - stats['mean']) / stats['std']) ** 3)
            )
        else:
            stats['skewness'] = 0.0

        # 첨도 (Kurtosis) - excess kurtosis (정규분포 = 0)
        if self.compute_kurtosis and stats['std'] > 0:
            stats['kurtosis'] = float(
                np.mean(((values - stats['mean']) / stats['std']) ** 4) - 3
            )
        else:
            stats['kurtosis'] = 0.0

        return stats

    def _compute_regional_statistics(self, magnitude: np.ndarray,
                                     mask: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        영역별 통계 계산 (그리드 기반)

        Args:
            magnitude: Flow magnitude 배열
            mask: 마스크 (선택)

        Returns:
            영역별 통계 딕셔너리
        """
        h, w = magnitude.shape
        row_step = h // self.grid_rows
        col_step = w // self.grid_cols

        regional_stats = {
            'grid_means': [],
            'grid_activities': [],
            'low_activity_regions': [],
            'high_activity_regions': []
        }

        all_activities: List[tuple] = []

        for i in range(self.grid_rows):
            row_means = []
            row_activities = []

            for j in range(self.grid_cols):
                # 그리드 영역 추출
                r_start = i * row_step
                r_end = (i + 1) * row_step if i < self.grid_rows - 1 else h
                c_start = j * col_step
                c_end = (j + 1) * col_step if j < self.grid_cols - 1 else w

                region_mag = magnitude[r_start:r_end, c_start:c_end]

                if mask is not None:
                    region_mask = mask[r_start:r_end, c_start:c_end]
                    valid_pixels = region_mag[region_mask > 0]
                    mask_area = np.sum(region_mask > 0)
                else:
                    valid_pixels = region_mag.flatten()
                    mask_area = region_mag.size

                # 영역 통계
                if len(valid_pixels) > 0:
                    region_mean = float(np.mean(valid_pixels))
                    region_activity = float(np.sum(valid_pixels)) / max(mask_area, 1)
                else:
                    region_mean = 0.0
                    region_activity = 0.0

                row_means.append(region_mean)
                row_activities.append(region_activity)
                all_activities.append((i, j, region_activity))

            regional_stats['grid_means'].append(row_means)
            regional_stats['grid_activities'].append(row_activities)

        # 이상 영역 식별
        if all_activities:
            activities = [a[2] for a in all_activities]
            mean_activity = np.mean(activities)
            std_activity = np.std(activities)

            if std_activity > 0:
                threshold_low = mean_activity - 1.5 * std_activity
                threshold_high = mean_activity + 1.5 * std_activity

                for i, j, activity in all_activities:
                    if activity < threshold_low:
                        regional_stats['low_activity_regions'].append({
                            'row': i, 'col': j, 'activity': activity
                        })
                    elif activity > threshold_high:
                        regional_stats['high_activity_regions'].append({
                            'row': i, 'col': j, 'activity': activity
                        })

        # 영역 통계 요약
        regional_stats['mean_activity'] = float(np.mean(activities)) if activities else 0.0
        regional_stats['std_activity'] = float(np.std(activities)) if activities else 0.0
        regional_stats['num_low_activity'] = len(regional_stats['low_activity_regions'])
        regional_stats['num_high_activity'] = len(regional_stats['high_activity_regions'])

        return regional_stats

    def _compute_direction_statistics(self, angles: np.ndarray) -> Dict[str, float]:
        """
        Flow 방향 분포 통계 (Circular statistics)

        Args:
            angles: 방향 각도 배열 (라디안)

        Returns:
            방향 통계 딕셔너리
        """
        if len(angles) == 0:
            return {
                'mean_direction': 0.0,
                'direction_variance': 0.0,
                'direction_uniformity': 0.0,
                'resultant_length': 0.0
            }

        # Circular mean (방향 평균)
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        mean_direction = np.arctan2(sin_sum, cos_sum)

        # Resultant length (방향 일관성)
        r = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)

        # Circular variance (방향 분산)
        # V = 1 - R (0: 완전 집중, 1: 완전 분산)
        direction_variance = 1 - r

        return {
            'mean_direction': float(mean_direction),
            'direction_variance': float(direction_variance),
            'direction_uniformity': float(r),
            'resultant_length': float(r)
        }

    def compute_activity_change(self, current_stats: Dict[str, Any],
                                previous_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        이전 통계와 현재 통계 간 변화량 계산

        Args:
            current_stats: 현재 프레임 통계
            previous_stats: 이전 프레임 통계

        Returns:
            변화량 딕셔너리
        """
        current_activity = current_stats.get('activity_index', 0)
        previous_activity = previous_stats.get('activity_index', 0)

        # 활동 변화량
        if previous_activity > 0:
            activity_change_ratio = (current_activity - previous_activity) / previous_activity
        else:
            activity_change_ratio = 0.0

        activity_change_absolute = current_activity - previous_activity

        # 전역 통계 변화
        current_global = current_stats.get('global', {})
        previous_global = previous_stats.get('global', {})

        mean_change = current_global.get('mean', 0) - previous_global.get('mean', 0)
        variance_change = current_global.get('variance', 0) - previous_global.get('variance', 0)

        return {
            'activity_change_ratio': float(activity_change_ratio),
            'activity_change_absolute': float(activity_change_absolute),
            'mean_change': float(mean_change),
            'variance_change': float(variance_change)
        }
