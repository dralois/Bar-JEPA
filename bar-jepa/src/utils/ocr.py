import numpy as np
import torch

from typing import List, Tuple

from src.utils.numeric import extract_numeric_value


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


    def infer_bar_values(
        self,
        bars: torch.Tensor,
        ticks: torch.Tensor,
        image: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts numeric OCR text, matches ticks to OCR numbers, and infers bar values.

        :param bars: predicted bars [y, x, conf], shape [N, 3]
        :param ticks: predicted ticks [y, x, conf], shape [N, 3]
        :param image: full-resolution image array, shape [H, W, 3], dtype uint8
        :return: tuple containing:

            - predicted bar centers [B, 2] (sorted left -> right)
            - predicted bar values [B]
        """
        device = bars.device
        h, w = image.shape[:2]

        if bars.size(0) == 0 or ticks.size(0) == 0:
            empty_centers = torch.empty((0, 2), device=device)
            empty_values = torch.empty((0,), device=device)
            return empty_centers, empty_values

        raw = self._engine.ocr(image, cls=True)
        pages = raw or []
        centers = []
        values = []

        # Process OCR detections
        for page in pages:
            for det in page:
                polygon = det[0]
                text_score = det[1]

                text = str(text_score[0]).strip()
                value = extract_numeric_value(text)
                if value is None:
                    continue

                # For numeric text, calculate normalized center coordinates
                poly_xy = [[float(x), float(y)] for x, y in polygon]
                poly_yx = _normalize_polygon(poly_xy, w, h)
                center = np.mean(np.array(poly_yx, dtype=np.float32), axis=0).tolist()
                centers.append([float(center[0]), float(center[1])])
                values.append(float(value))

        if len(values) == 0:
            text_centers = torch.empty((0, 2), device=device)
            text_values = torch.empty((0,), device=device)
        else:
            text_centers = torch.tensor(centers, device=device)
            text_values = torch.tensor(values, device=device)

        if text_centers.size(0) == 0:
            empty_centers = torch.empty((0, 2), device=device)
            empty_values = torch.empty((0,), device=device)
            return empty_centers, empty_values

        # Match each tick to closest OCR number and deduplicate text assignments
        tick_centers = ticks[:, :2]
        diffs = tick_centers.unsqueeze(1) - text_centers.unsqueeze(0)
        distances = torch.sqrt((diffs * diffs).sum(dim=2))
        best_dist, best_text_idx = distances.min(dim=1)

        kept_tick_idx = []
        kept_text_idx = []
        unique_text_idx = torch.unique(best_text_idx)

        # For each unique OCR text, select the closest tick mark
        for text_idx in unique_text_idx.tolist():
            candidate_tick_idx = torch.nonzero(best_text_idx == text_idx, as_tuple=False).flatten()
            local_best = torch.argmin(best_dist[candidate_tick_idx])
            selected_tick_idx = candidate_tick_idx[local_best]
            kept_tick_idx.append(int(selected_tick_idx.item()))
            kept_text_idx.append(int(text_idx))

        tick_idx = torch.tensor(kept_tick_idx, device=device, dtype=torch.long)
        text_idx = torch.tensor(kept_text_idx, device=device, dtype=torch.long)
        matched_tick_centers = tick_centers[tick_idx]
        matched_values = text_values[text_idx]

        # Sort matched ticks bottom -> top
        order = torch.argsort(matched_tick_centers[:, 0], descending=True)
        matched_tick_centers = matched_tick_centers[order]
        matched_values = matched_values[order]

        # Need at least two matched ticks to fit value(y) line
        if matched_tick_centers.size(0) < 2:
            empty_centers = torch.empty((0, 2), device=device)
            empty_values = torch.empty((0,), device=device)
            return empty_centers, empty_values

        # Fit value = slope * y + intercept from matched tick pairs using RANSAC
        ys_np = matched_tick_centers[:, 0].detach().cpu().numpy().reshape(-1, 1)
        values_np = matched_values.detach().cpu().numpy()
        from sklearn.linear_model import RANSACRegressor
        ransac = RANSACRegressor(random_state=0)
        ransac.fit(ys_np, values_np)
        slope = torch.tensor(float(ransac.estimator_.coef_[0]), device=device)
        intercept = torch.tensor(float(ransac.estimator_.intercept_), device=device)

        # Predict values at bar corner y-coordinates and sort left -> right
        bar_centers = bars[:, :2]
        pred_values = slope * bar_centers[:, 0] + intercept
        order = torch.argsort(bar_centers[:, 1])
        return bar_centers[order], pred_values[order]
