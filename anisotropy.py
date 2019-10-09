import luigi
import numpy as np
from luigi.util import delegates
from scipy import ndimage

from image import CorrectedImage
from parameters import CorrectedPairParams
from tracking import TrackedLabels
from utils import LocalNpz


@delegates
class Intensity(CorrectedPairParams, luigi.Task):
    """Calculates total intensity per cell."""

    def subtasks(self):
        return CorrectedImage(**self.corrected_image_params)

    def requires(self):
        return TrackedLabels(**self.corrected_relative_image_params)

    def output(self):
        return LocalNpz(self.to_results_file('.intensity.npz'))

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
