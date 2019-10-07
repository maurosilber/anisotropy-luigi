import luigi
import numpy as np
from luigi.util import delegates
from scipy import ndimage

from image import CorrectedImage
from tracking import TrackedLabels
from utils import CorrectedPairParams, LocalNpz


@delegates
class Intensity(CorrectedPairParams, luigi.Task):
    def subtasks(self):
        return CorrectedImage(path=self.path,
                              rel_path=self.rel_path,
                              g_factor_path=self.g_factor_path)

    def requires(self):
        return TrackedLabels(path=self.rel_path,
                             rel_path=self.rel_path,
                             g_factor_path=self.g_factor_rel_path)

    def output(self):
        return LocalNpz(self.results_file('.intensity.npz'))

    def run(self):
        tracked_labels = self.input().open()
        labels = np.arange(1, tracked_labels.max() + 1)
        intensities = np.empty((len(tracked_labels), labels.size))

        with self.subtasks() as ims:
            for i, (im, tracked_label) in enumerate(zip(ims.corrected_images(), tracked_labels)):
                # im has a mask where it's saturated.
                intensities[i] = ndimage.sum(im.filled(np.nan), tracked_label, labels)

        data = {}
        for label, intensity in zip(labels, intensities.T):
            cond = np.where(intensity > 0)[0]
            data[f'{label}_index'] = cond
            data[f'{label}_intensity'] = intensity[cond]

        self.output().save(**data)
