import os

import h5py
import FinderInference.production.main.utils.config as constants
import onnxruntime
import scann
from FinderInference.production.main.utils.logger_class import LoggerClass


class EmbeddingSearch:
    def __init__(self):
        # Load ONNX model with optimized session options
        self.ort_session = onnxruntime.InferenceSession(
            constants.EMBEDDING_MODEL_PATH,
            session_options=self.opt_session_setup(),
            providers=['CPUExecutionProvider'],
        )
        self.labels = self.load_labels()  # Load labels only here
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        self.neighburs_to_look = (
            50  # Increased to get more candidates for unique filtering
        )

        if os.path.exists(
            os.path.join(constants.INDEX_DIR, 'scann_config.pb')
        ):
            self.loaded_searcher = scann.scann_ops_pybind.load_searcher(
                constants.INDEX_DIR
            )
            print('✅ Loaded ScaNN index from file.')
        else:
            self.index_construction()
            self.loaded_searcher = scann.scann_ops_pybind.load_searcher(
                constants.INDEX_DIR
            )
            print('✅ Constructed and saved ScaNN index.')

    def load_labels(self):
        """
        Loads only the labels from the HDF5 file.
        """
        with h5py.File(constants.H5_PATH, 'r') as f:
            labels = f['labels'][:]
        # Decode bytes to strings if necessary
        labels = [
            label.decode('utf-8') if isinstance(label, bytes) else label
            for label in labels
        ]
        LoggerClass.info(
            f'Loaded {len(labels)} labels from {constants.H5_PATH}'
        )
        return labels

    def load_embeddings(self):
        """
        Loads embeddings and labels from the HDF5 file (used in index construction).
        """
        with h5py.File(constants.H5_PATH, 'r') as f:
            embeddings = f['embeddings'][:]
        LoggerClass.info(
            f'Loaded {len(embeddings)} embeddings and labels from {constants.H5_PATH}'
        )
        return embeddings

    def index_construction(self):
        """
        Builds the ScaNN index from stored image embeddings.
        """
        embeddings = self.load_embeddings()
        searcher = (
            scann.scann_ops_pybind.builder(
                embeddings, num_neighbors=10, distance_measure='dot_product'
            )
            .tree(
                num_leaves=35,  # max ~ (number of vectors)**(1/2): number of clusters
                num_leaves_to_search=10,  # 10%-50% of the total amount
                training_sample_size=len(
                    embeddings
                ),  # n of vectors used to train the internal quantization
            )
            .score_ah(
                dimensions_per_block=2,  # n of splits for processing a single vector (trade of between acc and speed)
                anisotropic_quantization_threshold=0.2,
            )
            .reorder(30)
            .build()
        )   # n of neighbors to precisely order

        searcher.serialize(constants.INDEX_DIR)

    def predict(self, image) -> list:
        embedding = self.ort_session.run(
            [self.output_name], {self.input_name: image}
        )[0][0]
        return embedding

    def process_model_outputs(self, embeddings) -> list:
        neighbors, distances = self.nearest_neighbor_search(embeddings)

        # Create a list to store top 10 unique barcodes in order of similarity
        top_barcodes = []
        seen_barcodes = set()

        # Process neighbors in order of similarity
        for i, idx in enumerate(neighbors):
            if idx < len(self.labels):
                barcode = self.labels[idx]
                # Only add if we haven't seen this barcode before
                if barcode not in seen_barcodes:
                    top_barcodes.append(barcode)
                    seen_barcodes.add(barcode)
                    # Stop once we have 10 unique barcodes
                    if len(top_barcodes) >= 10:
                        break
            else:
                LoggerClass.error(
                    f'Index {idx} out of range for labels with length {len(self.labels)}'
                )

        LoggerClass.info(
            f'Found {len(top_barcodes)} unique barcodes from {len(neighbors)} neighbors'
        )
        return top_barcodes

    def nearest_neighbor_search(self, embeddings):
        neighbors, distances = self.loaded_searcher.search(
            embeddings, self.neighburs_to_look
        )
        return neighbors, distances

    @staticmethod
    def opt_session_setup():
        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        return opt_session
