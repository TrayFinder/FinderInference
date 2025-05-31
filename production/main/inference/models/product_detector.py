import os
import time
from typing import List, Optional, Tuple, Union
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import main.utils.config as constants
from main.utils.logger_class import LoggerClass

class ProductDetector:
    def __init__(self, model_name: str = "2025-04-09/best.pt") -> None:
        self.model_path: str = os.path.join(constants.MODELS_DIR, model_name)
        self.model: YOLO = YOLO(self.model_path)

    def predict(
        self,
        image_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        show: bool = False
    ) -> Tuple[Results, float]:
        if image_path is None:
            image_path = os.path.join(constants.ASSETS_DIR, 'images/agua_cristal.jpeg')

        start_time = time.perf_counter()

        results: List[Results] = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            max_det=1,
            # verbose=False,
            # show=show
        )
        elapsed_time = time.perf_counter() - start_time
        return self._get_boxes(results[0]), elapsed_time

    def _get_boxes(self, result: Results) -> Union[np.ndarray, List]:
        try:
            predictions = result.boxes

            if not hasattr(predictions, "data"):
                LoggerClass.info("No predictions found.")
                return []

            tensor_boxes: np.ndarray = predictions.data.cpu().numpy()

            if tensor_boxes.ndim != 2 or tensor_boxes.shape[1] < 6 or tensor_boxes.shape[0] == 0:
                LoggerClass.info("No valid boxes.")
                return []
            LoggerClass.debug("Confidence:", tensor_boxes[4])
            return tensor_boxes[:4]

        except Exception as e:
            LoggerClass.error(f"Error extracting boxes: {e}")
            return []
