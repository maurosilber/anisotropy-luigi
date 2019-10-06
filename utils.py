import luigi
import numpy as np
from skimage.external.tifffile import TiffFile


class FileParam(luigi.Config):
    path = luigi.Parameter(description='Path to image.')


class RelFileParam(luigi.Config):
    rel_path = luigi.Parameter(description='Path to image from which shift is calculated relative to.')


class CorrectedImageParams(FileParam, RelFileParam):
    g_factor_path = luigi.Parameter()


class CorrectedPairParams(CorrectedImageParams):
    g_factor_rel_path = luigi.Parameter()


class LocalTiff(luigi.LocalTarget):
    tif = None

    def open(self, pages=None):
        self.tif = TiffFile(self.path, pages=pages)
        return self.tif

    def close(self):
        if self.tif is not None:
            self.tif.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(self):
        return NotImplementedError


class LocalNpy(luigi.LocalTarget):
    def open(self, mode='r'):
        return np.load(self.path, mmap_mode='r')

    def save(self, array):
        np.save(self.path, array)


class LocalNpz(luigi.LocalTarget):
    npz = None

    def open(self, mode='r'):
        self.npz = np.load(self.path, allow_pickle=True)
        return self.npz

    def close(self):
        self.npz.close()

    def save(self, *args, **kwargs):
        np.savez(self.path, *args, **kwargs)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.npz.close()
