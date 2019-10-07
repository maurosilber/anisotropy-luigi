import luigi
import numpy as np
from cellment import background
from luigi.util import delegates
from skimage import morphology

from image import CorrectedImage, CorrectedBackground
from utils import CorrectedImageParams, LocalNpy


@delegates
class Labels(CorrectedImageParams, luigi.Task):
    bg_threshold = luigi.FloatParameter()
    binary_opening_size = luigi.IntParameter()
    remove_small_objects = luigi.IntParameter()

    def subtasks(self):
        params = {'path': self.rel_path,
                  'rel_path': self.rel_path,
                  'g_factor_path': self.g_factor_path}
        return {'image': CorrectedImage(**params)}

    def requires(self):
        params = {'path': self.rel_path,
                  'rel_path': self.rel_path,
                  'g_factor_path': self.g_factor_path}
        return {'background': CorrectedBackground(**params)}

    def output(self):
        return LocalNpy(self.results_file('.labels.npy'))

    def run(self):
        with self.subtasks()['image'] as ims, self.input()['background'] as bg_rvs:
            labels = np.empty(ims.shape, dtype=int)
            max_label = 0
            for i, im in enumerate(ims.corrected_images()):
                bg_rv = background.HistogramRV(bg_rvs[str(i)])
                mask = (im.data > bg_rv.ppf(self.bg_threshold))  # using underlying image data without mask
                mask = morphology.binary_opening(mask, morphology.disk(self.binary_opening_size))
                label, max_label_i = morphology.label(mask, return_num=True)
                max_label = max(max_label, max_label_i)
                labels[i] = morphology.remove_small_objects(label, self.remove_small_objects)
        dtype = np.min_scalar_type(max_label)
        self.output().save(labels.astype(dtype))
