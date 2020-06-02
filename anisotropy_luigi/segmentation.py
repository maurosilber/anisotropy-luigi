import luigi
import numpy as np
from donkeykong.target import LocalNpy
from luigi.util import delegates
from skimage.segmentation import relabel_sequential

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
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        with self.subtasks()['image'] as ims:
            labels = np.empty(ims.shape, dtype=np.uint16)
            for i, im in enumerate(ims):
                labels[i] = model.predict_instances(normalize(im.data))[0]
        dtype = np.min_scalar_type(labels.max())
        self.output().save(labels.astype(dtype, copy=False))
