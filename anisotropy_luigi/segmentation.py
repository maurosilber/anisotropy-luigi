import luigi
import numpy as np
from cellment import functions, multi_threshold_segmentation
from donkeykong.target import LocalNpy
from luigi.util import delegates
from skimage.segmentation import relabel_from_one

from .image import CorrectedImage, CorrectedBackground
from .parameters import CorrectedImageParams


@delegates
class Labels(CorrectedImageParams, luigi.Task):
    """Segmentation of image."""
    binary_opening_size = luigi.IntParameter()

    def subtasks(self):
        return {'image': CorrectedImage(**self.corrected_image_params)}

    def requires(self):
        return {'background': CorrectedBackground(**self.corrected_image_params)}

    def output(self):
        return LocalNpy(self.to_results_file('.labels.npy'))

    def run(self):
        with self.subtasks()['image'] as ims, self.input()['background'] as bg_rvs:
            labels = np.empty(ims.shape, dtype=int)
            for i, im in enumerate(ims):
                bg_rv = functions.HistogramRV(bg_rvs[str(i)])  # loads background distribution
                labels[i] = multi_threshold_segmentation(im.data,  # using underlying image data without mask
                                                         (0.7, 0.9, 0.99, 0.999), bg_rv=bg_rv,
                                                         size=self.binary_opening_size)
                labels[i] = relabel_from_one(labels[i])
        dtype = np.min_scalar_type(labels.max())
        self.output().save(labels.astype(dtype, copy=False))
