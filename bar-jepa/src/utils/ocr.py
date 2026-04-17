import numpy as np
import torch

from typing import List

from src.utils.numeric import extract_numeric_value
from src.utils.postprocessing import hungarian_match


class NumericOCR:

    def __init__(
        self,
        language: str = 'en',
        rec_model: str = 'latin_PP-OCRv5_mobile_rec'
    ):
        """
        Initializes PaddleOCR.

        :param language: PaddleOCR language code for latin-script recognition
        :param rec_model: PaddleOCR recognition model name
        """
        self.language = language
        self.rec_model = rec_model
        self.available = False
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                text_recognition_model_name=self.rec_model,
                show_log=False
            )
            self.available = True
        except ImportError:
            print('PaddleOCR not available, numeric OCR will be disabled')

    @staticmethod
    def _normalize_polygon(
        polygon_xy: List[List[float]],
        width: int,
        height: int
    ) -> List[List[float]]:
        """
        Normalizes polygon points from [x, y] pixels to [y, x] in [0, 1].

        :param polygon_xy: polygon points in pixel coordinates [x, y]
        :param width: image width
        :param height: image height
        :return: normalized polygon points [[y, x], ...]
        """
        norm = []
        w = float(width)
        h = float(height)

        for x, y in polygon_xy:
            x = float(x)
            y = float(y)
            x_n = min(max(x / w, 0.0), 1.0)
            y_n = min(max(y / h, 0.0), 1.0)
            norm.append([y_n, x_n])

        return norm

    @staticmethod
    def _ransac_predict_bar_values(
        p_bars: torch.Tensor,
        matched_tick_centers: torch.Tensor,
        matched_values: torch.Tensor
    ) -> torch.Tensor:
        """
        RANSAC fitting and bar value prediction from bar y-coordinates.

        :param p_bars: predicted bars [y, x, conf], shape [N, 3]
        :param matched_tick_centers: matched tick positions [y, x], shape [M, 2]
        :param matched_values: numeric values at matched tick positions, shape [M]
        :return: predicted bars with values [B, 3] as [y, x, value], sorted left -> right
        """
        device = p_bars.device
        if matched_tick_centers.size(0) < 2:
            return torch.empty((0, 3), device=device)

        # Sort matched ticks bottom -> top before fitting
        order = torch.argsort(matched_tick_centers[:, 0], descending=True)
        matched_tick_centers = matched_tick_centers[order]
        matched_values = matched_values[order]

        from sklearn.linear_model import RANSACRegressor
        ys_np = matched_tick_centers[:, 0].detach().cpu().numpy().reshape(-1, 1)
        values_np = matched_values.detach().cpu().numpy()
        ransac = RANSACRegressor(random_state=0)
        ransac.fit(ys_np, values_np)
        slope = torch.tensor(float(ransac.estimator_.coef_[0]), device=device)
        intercept = torch.tensor(float(ransac.estimator_.intercept_), device=device)

        # Predict values at bar y-coordinates and sort left -> right
        bar_centers = p_bars[:, :2]
        pred_values = slope * bar_centers[:, 0] + intercept
        order = torch.argsort(bar_centers[:, 1])
        sorted_centers = bar_centers[order]
        sorted_values = pred_values[order].unsqueeze(1)
        return torch.cat((sorted_centers, sorted_values), dim=1)

    def infer_bar_values(
        self,
        bars: torch.Tensor,
        ticks: torch.Tensor,
        image: np.ndarray,
        dist_thresh: float,
        gt_tick_yxv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Infers bar values via OCR and oracle (GT tick labels).

        :param bars: predicted bars [y, x, conf], shape [N, 3]
        :param ticks: predicted ticks [y, x, conf], shape [M, 3]
        :param image: full-resolution image array, shape [H, W, 3], dtype uint8
        :param dist_thresh: max. spatial distance for tick matching
        :param gt_tick_yxv: ground-truth ticks [y, x, value], shape [K, 3]
        :return: tuple containing:
        
            - ocr bars, [B, 3] as [y, x, value], sorted left -> right
            - oracle bars, [B, 3] as [y, x, value], sorted left -> right
        """
        device = bars.device
        empty = torch.empty((0, 3), device=device)

        tick_centers = ticks[:, :2]

        # Oracle path
        if bars.size(0) == 0 or ticks.size(0) == 0 or gt_tick_yxv.size(0) == 0:
            oracle = empty
        else:
            tick_idx, gt_idx = hungarian_match(tick_centers, gt_tick_yxv[:, :2], dist_thresh)
            if tick_idx.numel() == 0:
                oracle = empty
            else:
                oracle = self._ransac_predict_bar_values(
                    bars,
                    tick_centers[tick_idx],
                    gt_tick_yxv[:, 2][gt_idx]
                )

        # OCR path
        if bars.size(0) == 0 or ticks.size(0) == 0 or not self.available:
            return empty, oracle

        h, w = image.shape[:2]
        raw = self._engine.ocr(image, cls=True)
        pages = raw or []
        centers = []
        values = []

        for page in pages:
            for det in page:
                polygon = det[0]
                text_score = det[1]

                text = str(text_score[0]).strip()
                value = extract_numeric_value(text)
                if value is None:
                    continue

                poly_xy = [[float(x), float(y)] for x, y in polygon]
                poly_yx = self._normalize_polygon(poly_xy, w, h)
                center = np.mean(np.array(poly_yx, dtype=np.float32), axis=0).tolist()
                centers.append([float(center[0]), float(center[1])])
                values.append(float(value))

        if len(values) == 0:
            return empty, oracle

        text_centers = torch.tensor(centers, device=device)
        text_values = torch.tensor(values, device=device)

        tick_idx, text_idx = hungarian_match(tick_centers, text_centers, dist_thresh)
        if tick_idx.numel() == 0:
            return empty, oracle

        ocr = self._ransac_predict_bar_values(
            bars,
            tick_centers[tick_idx],
            text_values[text_idx]
        )

        return ocr, oracle
