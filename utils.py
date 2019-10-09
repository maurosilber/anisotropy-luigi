import pathlib

import luigi
import numpy as np
import pandas as pd
from skimage.external.tifffile import TiffFile


class LocalTarget(luigi.LocalTarget):
    """LocalTarget extends luigi.LocalTarget for invalidation operations.

    Adds a remove method to remove a file.
    Adds a protected keyword argument to prevent file removal.
    """

    def __init__(self, *args, **kwargs):
        self.protected = kwargs.pop('protected', False)
        super().__init__(*args, **kwargs)

    def remove(self):
        if not self.protected:
            pathlib.Path(self.path).unlink()


class LocalTiff(LocalTarget):
    tif = None

    def open(self, pages=None):
        self.tif = TiffFile(self.path, pages=pages)
        return self

    def close(self):
        if self.tif is not None:
            self.tif.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(self):
        return NotImplementedError

    def __len__(self):
        return len(self.tif)

    def image(self, item=0):
        return self.tif.asarray(key=item)

    def images(self):
        for item in range(len(self.tif)):
            yield self.image(item)

    @property
    def shape(self):
        return (len(self), *self.tif[0].shape)


class LocalNpy(LocalTarget):
    def open(self, mode='r'):
        return np.load(self.path, mmap_mode=mode)

    def save(self, array):
        np.save(self.path, array)


class LocalNpz(LocalTarget):
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


class LocalPandas(LocalTarget):
    def open(self):
        return pd.read_pickle(self.path)

    def save(self, df):
        pd.to_pickle(df, self.path)
