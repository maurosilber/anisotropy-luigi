import luigi
import numpy as np
from cellment import background, multi_threshold_segmentation
from donkeykong.target import LocalNpy
from luigi.util import delegates

from .image import CorrectedImage, CorrectedBackground
from .parameters import CorrectedImageParams


@delegates
class Labels(CorrectedImageParams, luigi.Task):
    """Segmentation of image."""
    bg_threshold = luigi.FloatParameter()
    binary_opening_size = luigi.IntParameter()
    remove_small_objects = luigi.IntParameter()

    def subtasks(self):
        return {'image': CorrectedImage(**self.corrected_image_params)}

    def requires(self):
        return {'background': CorrectedBackground(**self.corrected_image_params)}

    def output(self):
        return LocalNpy(self.to_results_file('.labels.npy'))

    def run(self):
        with self.subtasks()['image'] as ims, self.input()['background'] as bg_rvs:
            labels = np.empty(ims.shape, dtype=int)
            max_label = 0
            for i, im in enumerate(ims):
                bg_rv = background.HistogramRV(bg_rvs[str(i)])  # loads background distribution
                mask = (im.data > bg_rv.ppf(self.bg_threshold))  # using underlying image data without mask
                mask = morphology.binary_opening(mask, morphology.disk(self.binary_opening_size))
                label, max_label_i = morphology.label(mask, return_num=True)
                max_label = max(max_label, max_label_i)
                labels[i] = morphology.remove_small_objects(label, self.remove_small_objects)
        dtype = np.min_scalar_type(max_label)
        self.output().save(labels.astype(dtype))
