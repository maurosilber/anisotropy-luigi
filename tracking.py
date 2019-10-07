import itertools

import luigi
import numpy as np
from cellment import tracking

from image import CorrectedImageParams
from segmentation import Labels
from utils import LocalNpy


class TrackedLabels(CorrectedImageParams, luigi.Task):
    """Tracks cells and generates a new label mask."""

    def requires(self):
        return Labels(path=self.rel_path,
                      rel_path=self.rel_path,
                      normalization_path=self.normalization_path)

    def output(self):
        return LocalNpy(self.results_file('.tracked_labels.npy'))

    def run(self):
        labels = self.input().open()  # Loads labeled segmentation

        graph = tracking.labels_graph(labels)  # Computes graph of intersections
        subgraphs = tracking.decompose(graph)  # Decomposes in disconnected subgraphs
        subgraphs = map(tracking.get_timelike_chains, subgraphs)  # Decomposes in timelike chains
        subgraphs = itertools.chain.from_iterable(subgraphs)
        subgraphs = list(filter(lambda x: len(x) > 10, subgraphs))  # Filters short chains

        # Generates new labeled mask with tracking data
        tracked_labels = np.zeros_like(labels, dtype=np.min_scalar_type(len(subgraphs) + 1))
        for new_label, subgraph in enumerate(subgraphs, 1):
            for t, label in subgraph.nodes:
                tracked_labels[t][labels[t] == label] = new_label

        self.output().save(tracked_labels)
