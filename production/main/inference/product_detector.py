import os
from time import perf_counter, sleep
from typing import List, Optional, Tuple, Union
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import main.utils.config as constants
from main.utils.logger_class import LoggerClass

class ProductDetector:
    def __init__(self, model_path: str = constants.PRODUCT_DETECTOR_MODEL_PATH) -> None:
        self.model: YOLO = YOLO(model_path)

    def predict(
        self,
        image: np.array,
        conf_threshold: float = 0.25,
        show: bool = False
    ) -> Tuple[Results, float]:

        start_time = perf_counter()


        results: List[Results] = self.model.predict(
            source=image,
            conf=conf_threshold,
            max_det=1,
            verbose=False,
            show=show
        )
        elapsed_time = perf_counter() - start_time
        return self._get_boxes(results[0]), elapsed_time

    def _get_boxes(self, result: Results) -> Union[np.ndarray, List]:
        try:
            predictions = result.boxes

            if not hasattr(predictions, "data"):
                LoggerClass.debug("No predictions found.")
                return []

            tensor_boxes: np.ndarray = predictions.data.cpu().numpy()

            if tensor_boxes.ndim != 2 or tensor_boxes.shape[1] < 6 or tensor_boxes.shape[0] == 0:
                LoggerClass.info("No valid boxes.")
                return []

            LoggerClass.debug("Box found with confidence(s): %s", tensor_boxes[:, 4])
            print("Tensor boxes:", tensor_boxes)

            # Return only [x1, y1, x2, y2] for each box
            return tensor_boxes[:, :4]

        except Exception as e:
            LoggerClass.error(f"Error extracting boxes: {e}")
            return []
