import itertools

import luigi
import numpy as np
from cellment import tracking

from image import CorrectedImageParams
from segmentation import Labels
from utils import LocalNpy


class TrackedLabels(CorrectedImageParams, luigi.Task):
    def requires(self):
        return Labels(path=self.rel_path,
                      rel_path=self.rel_path,
                      g_factor_path=self.g_factor_path)

    def output(self):
        return LocalNpy(self.path.with_suffix('.tracked_labels.npy'))

    def run(self):
        labels = self.input().open()
        graph = tracking.labels_graph(labels)
        subgraphs = map(tracking.get_timelike_chains, tracking.decompose(graph))
        subgraphs = itertools.chain.from_iterable(subgraphs)
        subgraphs = list(filter(lambda x: len(x) > 10, subgraphs))
        tracked_labels = np.zeros_like(labels, dtype=np.min_scalar_type(len(subgraphs) + 1))
        for new_label, subgraph in enumerate(subgraphs, 1):
            for t, label in subgraph.nodes:
                tracked_labels[t][labels[t] == label] = new_label
        self.output().save(tracked_labels)
