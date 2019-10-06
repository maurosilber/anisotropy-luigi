import json

import luigi
import numpy as np
from cellment import background
from luigi.util import delegates

from utils import FileParam, RelFileParam, CorrectedImageParams, LocalTiff, LocalNpy, LocalNpz


class Image(FileParam, luigi.ExternalTask):
    def output(self):
        return LocalTiff(self.path)


class MaskedImage(FileParam, luigi.WrapperTask):
    saturation = luigi.IntParameter()
    tif = None

    def requires(self):
        return Image(self.path)

    def open(self):
        self.tif = self.input().open()
        return self

    def close(self):
        if self.tif is not None:
            self.tif.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def masked_image(self, item):
        return np.ma.masked_greater_equal(self.tif.asarray(item), self.saturation)

    def masked_images(self):
        for i in range(len(self.tif)):
            yield self.masked_image(i)


@delegates
class Background(FileParam, luigi.Task):
    sigma = luigi.FloatParameter()
    window = luigi.IntParameter()
    threshold = luigi.FloatParameter()

    def subtasks(self):
        return MaskedImage(self.path)

    def output(self):
        return LocalNpy(self.path.with_suffix('.bg.npy'))

    def run(self):
        with self.subtasks() as ims:
            bg = np.empty(len(ims.tif))
            for i, im in enumerate(ims.masked_images()):
                bg_rv = background.bg_rv(im, self.sigma, self.window, threshold=self.threshold)
                bg[i] = bg_rv.median()
        self.output().save(bg)


@delegates
class Shift(FileParam, RelFileParam, luigi.Task):
    # Do we want a corrected image to compute shift?
    def subtasks(self):
        return MaskedImage(self.path), MaskedImage(self.rel_path)

    def output(self):
        return LocalNpy(self.path.with_suffix('.shift.npy'))

    def run(self):
        ims, rel_ims = self.subtasks()
        if ims.path == rel_ims.path:
            self.output().save((0., 0.))
        else:
            # TODO: compute shift
            self.output().save((0., 0.))


class G_Factor(FileParam, luigi.ExternalTask):
    def output(self):
        return LocalTiff(self.path)


class Metadata(FileParam, luigi.Task):
    def requires(self):
        return Image(self.path)

    def output(self):
        return luigi.LocalTarget(self.path.with_suffix('.exposure.txt'))

    def run(self):
        with self.input().open() as tif:
            metadata = self.parse_metadata(self.path)
        with self.output().open('w') as f:
            json.dump(metadata, f, indent=0)

    @staticmethod
    def parse_metadata(file):
        with open(file, 'rb') as f:
            metadata = []
            append = False
            for row in f:
                try:
                    row = row.decode('utf-8')
                except:
                    continue

                if not append and 'Band' in row:
                    append = True
                elif append:
                    metadata.append(row)
                    if 'TimePos1' in row:
                        append = False
        metadata = {k: v for k, v in (s[:-1].split('=') for s in metadata if '=' in s)}
        return metadata


@delegates
class CorrectedImage(CorrectedImageParams, luigi.WrapperTask):
    def subtasks(self):
        return MaskedImage(self.path)

    def requires(self):
        return {'background': Background(self.path),
                'shift': Shift(self.path, self.rel_path),
                'g_factor': G_Factor(self.g_factor_path),
                'image_metadata': Metadata(self.path),
                'g_factor_metadata': Metadata(self.g_factor_path)}

    def __enter__(self):
        self.ims = self.subtasks().open()
        self.bg = self.input()['background'].open()
        self.shift = self.input()['shift'].open()
        self.axis = tuple(range(-len(self.shift), 0))
        self.g_factor = self.input()['g_factor'].open()
        with self.input()['image_metadata'].open() as f:
            self.image_exposure = float(json.load(f)['ExposureTime1'])
        with self.input()['g_factor_metadata'].open() as f:
            self.g_factor_exposure = float(json.load(f)['ExposureTime'])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ims.close()
        self.g_factor.close()

    def corrected_image(self, item):
        im = self.ims.masked_image(item) / self.image_exposure
        bg = self.bg[item] / self.image_exposure
        g_factor = self.g_factor.asarray() / self.g_factor_exposure
        return np.roll((im - bg) / (g_factor - bg), self.shift, axis=self.axis)

    def corrected_images(self):
        for i in range(len(self.ims.tif)):
            yield self.corrected_image(i)

    @property
    def shape(self):
        return (len(self.ims.tif), *self.ims.tif[0].shape)


@delegates
class CorrectedBackground(CorrectedImageParams, luigi.Task):
    sigma = luigi.FloatParameter()
    window = luigi.IntParameter()
    threshold = luigi.FloatParameter()

    def subtasks(self):
        return CorrectedImage(path=self.path,
                              rel_path=self.rel_path,
                              g_factor_path=self.g_factor_path)

    def output(self):
        return LocalNpz(self.path.with_suffix('.corrected_bg_rv.npz'))

    def run(self):
        with self.subtasks() as ims:
            bg_rvs = {}
            for i, im in enumerate(ims.corrected_images()):
                bg_rv = background.bg_rv(im, self.sigma, self.window, threshold=self.threshold)
                bg_rvs[str(i)] = bg_rv._histogram
        self.output().save(**bg_rvs)
