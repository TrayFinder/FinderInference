from time import perf_counter

import cv2
import main.utils.config as constants
import numpy as np
from main.inference.embedding_search import EmbeddingSearch
from main.inference.product_detector import ProductDetector
from main.utils.image_processor import ImageProcessor
from main.utils.logger_class import LoggerClass


class AiPipeline:
    """
    Class responsible for combining all models, doing the
    preprocessing and postprocessing Methods should return only the final decision
    """

    def __init__(self):
        LoggerClass.configure('test', debug=True)
        LoggerClass.debug('Initializing AiPipeline')
        self.embedding_searcher = EmbeddingSearch()

    def detector_selector(self, image):
        detector_selector = ProductDetector()
        box_coordinates, inference_time = detector_selector.predict(image)
        LoggerClass.debug(f'Box coordinates: {box_coordinates}')
        if len(box_coordinates) == 0:
            LoggerClass.log_json(
                category='debug',
                json_dict={
                    'mode': 'product_detector',
                    'box_found': 'no',
                    'inference_time': round(inference_time, 4),
                },
            )
            return []

        LoggerClass.log_json(
            category='debug',
            json_dict={
                'mode': 'product_detector',
                'box_found': 'yes',
                'inference_time': round(inference_time, 4),
            },
        )

        cropped_image = ImageProcessor.crop_original_image(
            box_coordinates[0], image
        )

        # image_288 = ImageProcessor.preprocess(cropped_image, constants.TARGET_SIZE)

        padded_img = ImageProcessor.resize_and_pad(
            cropped_image, constants.TARGET_SIZE
        )

        product_image = ImageProcessor.preprocess_for_embedding_model(
            padded_img
        )

        return product_image

    def embedding_search(self, product_image):
        """
        Wraper function for the embedding searcher model
        """
        start_time = perf_counter()
        embedding = self.embedding_searcher.predict(product_image)
        generation_time = perf_counter() - start_time

        start_time = perf_counter()
        barcodes = self.embedding_searcher.process_model_outputs(embedding)
        search_time = perf_counter() - start_time

        if barcodes is None:
            LoggerClass.log_json(
                category='debug',
                json_dict={
                    'mode': 'embedding_search',
                    'barcodes_found': 'no',
                    'embedding_generation_time': round(generation_time, 4),
                    'search_time': round(search_time, 4),
                },
            )
            return []

        LoggerClass.log_json(
            category='debug',
            json_dict={
                'mode': 'embedding_search',
                'barcodes_found': 'yes',
                'barcodes_predicted': barcodes,
                'embedding_generation_time': round(generation_time, 4),
                'search_time': round(search_time, 4),
            },
        )
        return barcodes

    def __call__(self, image_path):
        image = cv2.imread(image_path)
        processed_image = self.detector_selector(image)
        return self.embedding_search(processed_image)


if __name__ == '__main__':
    ai = AiPipeline()
    print(ai(constants.IMAGES_DIR + 'agua_cristal.jpeg'))
