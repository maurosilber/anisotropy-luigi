import luigi
import numpy as np
from cellment import tracking
from donkeykong.target import LocalNpy
from luigi.util import delegates

from .image import CorrectedImage
from .parameters import CorrectedImageParams
from .segmentation import Labels


@delegates
class TrackedLabels(CorrectedImageParams, luigi.Task):
    """Tracks cells and generates a new label mask."""
    area_threshold = luigi.IntParameter()
    edge_threshold = luigi.FloatParameter()

    def subtasks(self):
        return CorrectedImage(**self.corrected_image_params)

    def requires(self):
        return Labels(**self.corrected_image_params)

    def output(self):
        return LocalNpy(self.to_results_file('.tracked_labels.npy'))

    def run(self):
        labels = self.input().open().copy()  # Loads labeled segmentation
        graph = tracking.Labels_graph.from_labels_stack(labels)  # Computes graph of intersections

        # Split merged labels
        with self.subtasks() as images:
            tracking.split_nodes(labels, graph, images, self.area_threshold, self.edge_threshold)

        subgraphs = tracking.decompose(graph)  # Decomposes in disconnected subgraphs
        subgraphs = list(filter(lambda x: x.is_timelike_chain(), subgraphs))  # Keeps only time-like chains

        # Generates new labeled mask with tracking data
        tracked_labels = np.zeros_like(labels, dtype=np.min_scalar_type(len(subgraphs) + 1))
        for new_label, subgraph in enumerate(subgraphs, 1):
            for node in subgraph:
                tracked_labels[node.time][labels[node.time] == node.label] = new_label

        self.output().save(tracked_labels)
