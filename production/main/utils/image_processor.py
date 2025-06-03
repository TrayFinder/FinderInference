"""
Class responsible for preprocessing images
"""
import cv2
import numpy as np


class ImageProcessor:
    """
    Class responsible for preprocessing images
    """

    @staticmethod
    def resize_and_pad(img, target_size):
        """
        Resize and pad an image to a target size
        The pad is needed since the original image is not a square
        """
        # Calculate the target aspect ratio and determine dimensions for resizing
        aspect_ratio = img.shape[1] / img.shape[0]
        target_aspect_ratio = target_size[1] / target_size[0]
        if aspect_ratio > target_aspect_ratio:
            new_width = target_size[1]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[0]
            new_width = int(target_size[0] * aspect_ratio)
        # Resize the image directly to new dimensions
        img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        # Initialize the output image with padding and then place the resized image
        delta_w = target_size[1] - new_width
        delta_h = target_size[0] - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [114, 114, 114]  # This color can be changed as needed
        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return padded_img

    @staticmethod
    def preprocess(frame, target_size):
        """Preprocesses the frame for the YOLO models"""
        frame = ImageProcessor.resize_and_pad(frame, target_size)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        preprocessed_frame = np.expand_dims(frame, axis=0)
        return preprocessed_frame

    @staticmethod
    def preprocess_for_embedding_model(image):
        """Preprocesses the image for the embedding model"""
        image = image.astype(np.float32) / 255.0
        preprocessed_image = np.expand_dims(image, axis=0)
        return preprocessed_image

    @staticmethod
    def crop_original_image(adjusted_box, frame):
        """Crops the original image based on the adjusted bounding box"""
        # adjusted_box = [int(i) for i in adjusted_box]
        x1, y1, x2, y2 = map(lambda x: int(round(x)), adjusted_box)
        cropped_image = frame[y1:y2, x1:x2]
        return cropped_image
