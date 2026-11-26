"""Execute the authors' pretrained TensorFlow 1 ViSiL inference graph on CPU."""

import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

from .dataset import sha256
from .resources import TF_DIGESTS, TF_REVISION


class ViSiL:
    """Feed RGB frames and regional descriptors to the published ResNet graph."""

    def __init__(self, checkpoint, batch_size=16):
        self.batch_size = batch_size
        self.graph = tf.Graph()
        digests = {name: sha256(checkpoint / name) for name in TF_DIGESTS}
        if digests != TF_DIGESTS:
            raise ValueError(f"Unexpected ViSiL checkpoint or graph bytes in {checkpoint}")
        self.provenance = {
            "name": "ViSiL (authors' published TensorFlow ResNet-50 inference graph)",
            "upstream_revision": TF_REVISION,
            "checkpoint": "ckpt/resnet/model",
            "checkpoint_sha256": digests,
            "tensorflow": tf.__version__,
            "batch_size": batch_size,
            "regional_dim": 3840,
            "frame_regions": 9,
            "preprocessing": "Embedded RGB preprocessing, 256px short edge, 224px center crop",
            "similarity": "Directional Chamfer, learned temporal comparator, directional Chamfer",
        }
        self.cache_key = hashlib.sha256(
            (str(self.provenance) + sha256(Path(__file__))).encode()
        ).hexdigest()
        prefix = str(checkpoint / "model")
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(prefix + ".meta", clear_devices=True)
        tensor = self.graph.get_tensor_by_name
        self.frames = tensor("input:0")
        self.descriptors = tensor("Mul:0")
        self.query = tensor("query:0")
        self.candidate = tensor("candidate:0")
        self.frame_matrix = tensor("Squeeze:0")
        self.temporal_matrix = tensor("video_comparator/Squeeze:0")
        self.similarity = tensor("Squeeze_1:0")
        config = tf.compat.v1.ConfigProto(device_count={"GPU": 0},
                                         intra_op_parallelism_threads=4,
                                         inter_op_parallelism_threads=2)
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        try:
            with self.graph.as_default():
                saver.restore(self.session, prefix)
        except Exception:
            self.session.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.session.close()

    def extract_features(self, frames):
        batches = [
            self.session.run(self.descriptors, {self.frames: frames[start:start+self.batch_size]})
            for start in range(0, len(frames), self.batch_size)
        ]
        return np.concatenate(batches, axis=0)

    def compare(self, query, candidate):
        frame, temporal, score = self.session.run(
            (self.frame_matrix, self.temporal_matrix, self.similarity),
            {self.query: query, self.candidate: candidate},
        )
        return frame, temporal, float(score)
