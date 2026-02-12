from typing import Dict, List, Optional

import numpy as np
import torch
import re

from paddleocr import PaddleOCR


_NUMERIC_PATTERN = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?')


def _tensor_to_numpy_rgb(image: torch.Tensor) -> np.ndarray:
    """
    Converts an image tensor to uint8 RGB numpy array.

    :param image: image tensor, shape [C, H, W] or [H, W]
    :return: image as numpy array, shape [H, W, 3], dtype uint8
    """
    img = image.detach().cpu()

    if img.ndim != 3:
        raise ValueError(f'Expected image tensor with 3 dims, got shape {tuple(img.shape)}')

    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    else:
        img = img[:3]

    np_img = img.permute(1, 2, 0).numpy()

    if np_img.dtype != np.uint8:
        if np_img.max() <= 1.0:
            np_img = np_img * 255.0
        np_img = np.clip(np_img, 0.0, 255.0).astype(np.uint8)

    return np_img


def _extract_numeric_value(text: str) -> Optional[float]:
    """
    Extracts the first numeric value from OCR text.

    :param text: OCR output text
    :return: parsed float value or None if not numeric
    """
    cleaned = text.replace('−', '-')
    match = _NUMERIC_PATTERN.search(cleaned)
    if match is None:
        return None

    token = match.group(0).replace(',', '')
    try:
        return float(token)
    except ValueError:
        return None


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
    w = float(max(width, 1))
    h = float(max(height, 1))

    for x, y in polygon_xy:
        x = float(x)
        y = float(y)
        x_n = min(max(x / w, 0.0), 1.0)
        y_n = min(max(y / h, 0.0), 1.0)
        norm.append([y_n, x_n])

    return norm


class NumericOCR:
    """
    PaddleOCR wrapper for extracting numeric text with normalized coordinates.

    Returned items follow the format:

    - value: parsed float value
    - center: normalized [y, x]
    """

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
        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=self.language,
            text_recognition_model_name=self.rec_model,
            show_log=False
        )
        self.available = True

    def extract_numeric_text(
        self,
        image: torch.Tensor
    ) -> List[Dict[str, float | List[float]]]:
        """
        Extracts numeric OCR results from one full-resolution image.

        :param image: full-resolution image tensor, shape [C, H, W]
        :return: list of numeric OCR results containing value + center
        """
        np_img = _tensor_to_numpy_rgb(image)
        h, w = np_img.shape[:2]

        raw = self._engine.ocr(np_img, cls=True)  # type: ignore
        pages = raw or []
        results: List[Dict[str, float | List[float]]] = []

        for page in pages:
            for det in page:
                polygon = det[0]
                text_score = det[1]

                text = str(text_score[0]).strip()
                value = _extract_numeric_value(text)
                if value is None:
                    continue

                poly_xy = [[float(x), float(y)] for x, y in polygon]
                poly_yx = _normalize_polygon(poly_xy, w, h)

                center = np.mean(np.array(poly_yx, dtype=np.float32), axis=0).tolist()
                results.append({
                    'value': value,
                    'center': [float(center[0]), float(center[1])]
                })

        return results

    def match_ticks_to_numeric_text(
        self,
        ticks: torch.Tensor,
        numeric_text: List[Dict[str, float | List[float]]]
    ) -> List[Dict[str, float | List[float]]]:
        """
        Matches each predicted tick to the closest numeric OCR text.

        :param ticks: predicted ticks [y, x, conf], shape [N, 3]
        :param numeric_text: OCR numeric text entries for one image
        :return: sorted unique matches with tick center and mapped numeric value
        """
        if ticks.size(0) == 0 or len(numeric_text) == 0:
            return []

        text_centers = np.array([entry['center'] for entry in numeric_text], dtype=np.float32)
        tick_centers = ticks[:, :2].detach().cpu().numpy().astype(np.float32)

        best_for_text: Dict[int, Dict[str, float | List[float]]] = {}

        for tick_center in tick_centers:
            diffs = text_centers - tick_center[None, :]
            distances = np.sqrt((diffs * diffs).sum(axis=1))
            best_text_idx = int(np.argmin(distances))
            best_distance = float(distances[best_text_idx])
            best_entry = numeric_text[best_text_idx]

            prev = best_for_text.get(best_text_idx)
            if prev is not None and best_distance >= float(prev['distance']):
                continue

            best_for_text[best_text_idx] = {
                'tick_center': [float(tick_center[0]), float(tick_center[1])],
                'value': float(best_entry['value']),
                'distance': best_distance
            }

        deduped = list(best_for_text.values())
        # Sort bottom -> top, then left -> right.
        deduped.sort(
            key=lambda m: (
                -float(m['tick_center'][0]),  # type: ignore
                float(m['tick_center'][1])    # type: ignore
            )
        )

        return [{'tick_center': m['tick_center'], 'value': m['value']} for m in deduped]

    def infer_bar_values(
        self,
        bars: torch.Tensor,
        tick_text_matches: List[Dict[str, float | List[float]]]
    ) -> tuple[List[Dict[str, float | List[float]]], Optional[float]]:
        """
        Infers bar values from predicted bar corners and matched OCR ticks.

        :param bars: predicted bars [y, x, conf], shape [N, 3]
        :param tick_text_matches: matched tick/value pairs for one image
        :return: tuple with:

            - predicted bars with values (sorted left -> right)
            - mean normalized y-distance per value unit
        """
        if bars.size(0) == 0 or len(tick_text_matches) < 2:
            return [], None

        ys = np.array(
            [float(m['tick_center'][0]) for m in tick_text_matches],  # type: ignore[index]
            dtype=np.float64
        )
        vals = np.array([float(m['value']) for m in tick_text_matches], dtype=np.float64)

        tri_u = np.triu_indices(len(ys), k=1)
        dy = np.abs(ys[tri_u[0]] - ys[tri_u[1]])
        dv = np.abs(vals[tri_u[0]] - vals[tri_u[1]])

        eps = 1e-8
        valid = dv > eps
        if not np.any(valid):
            return [], None

        dist_per_value = float((dy[valid] / dv[valid]).mean())
        if not np.isfinite(dist_per_value) or dist_per_value <= eps:
            return [], None

        slope, intercept = np.polyfit(ys, vals, 1)
        if not np.isfinite(slope) or abs(slope) <= eps:
            return [], None

        bar_centers = bars[:, :2].detach().cpu().numpy().astype(np.float32)
        pred_values = float(slope) * bar_centers[:, 0].astype(np.float64) + float(intercept)

        bar_value_pairs: List[Dict[str, float | List[float]]] = []
        for center, value in zip(bar_centers, pred_values):
            bar_value_pairs.append({
                'bar_center': [float(center[0]), float(center[1])],
                'value': float(value)
            })

        bar_value_pairs.sort(key=lambda m: float(m['bar_center'][1]))  # type: ignore[index]
        return bar_value_pairs, dist_per_value
