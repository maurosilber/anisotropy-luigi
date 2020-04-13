from contextlib import ExitStack

import luigi
import numpy as np
from cellment import functions, background
from donkeykong.target import LocalNpy, LocalNpz, LocalTiff, LocalJSON
from luigi.util import delegates
from scipy import interpolate
from skimage import feature
from sklearn.svm import SVR

from .files import Files
from .parameters import FileParam, ExperimentParam, ChannelParams, RelativeChannelParams, \
    CorrectedImageParams


class Image(luigi.ExternalTask, FileParam):
    def output(self):
        return LocalTiff(self.path, protected=True)


class MaskedImage(luigi.WrapperTask, FileParam):
    """Image with saturated areas masked.

    Implements methods to iterate through the masked images.
    """
    saturation = luigi.IntParameter()
    ims = None

    def requires(self):
        return Image(path=self.path)

    def open(self):
        self.ims = self.input().open()
        return self

    def close(self):
        self.ims.close()

    def __getitem__(self, item):
        return np.ma.masked_greater_equal(self.ims[item], self.saturation)

    def __len__(self):
        return len(self.ims)

    @property
    def shape(self):
        return self.ims.shape


@delegates
class Background(FileParam, luigi.Task):
    sigma = luigi.FloatParameter()
    window = luigi.IntParameter()
    threshold = luigi.FloatParameter()

    def subtasks(self):
        return MaskedImage(path=self.path)

    def output(self):
        return LocalNpz(self.to_results_file('.bg.npz'))

    def run(self):
        with self.subtasks() as ims:
            bgs = {}
            for i, im in enumerate(ims):
                bgs[str(i)] = self.calc_background(im)
        self.output().save(**bgs)

    def calc_background(self, image, d=15):
        smo = functions.smo(image, self.sigma, self.window)
        smo_rv = functions.smo_rv(image.ndim, self.sigma, self.window)
        smo_mask = feature.peak_local_max(-smo, min_distance=5, indices=False)
        smo_mask &= smo < smo_rv.ppf(self.threshold)
        smo_mask &= smo > smo_rv.ppf(0.01)
        med, *iqr = np.percentile(image[smo_mask], (50, 25, 75))
        smo_mask &= image < med + 3 * np.diff(iqr)

        svr = SVR(gamma='scale')
        X, y = np.array(np.nonzero(smo_mask)).T, image[smo_mask]
        svr.fit(X, y)

        ymax, xmax = image.shape
        grid = np.mgrid[0:ymax:d, 0:xmax:d]
        y, x = grid[0, :, 0], grid[1, 0]
        z = svr.predict(grid.reshape(2, -1).T).reshape(grid[0].shape)
        return x, y, z

    def interpolate_background(self, x, y, z, shape):
        inter = interpolate.RectBivariateSpline(y, x, z)
        y, x = map(np.arange, shape)
        z = inter(y, x)
        return z

    def get_background(self, i, shape):
        npz = self.npz
        return self.interpolate_background(*npz[str(i)], shape)

    def open(self):
        self.npz = self.output().open()
        return self

    def close(self):
        self.npz.close()


class Shift(ExperimentParam, ChannelParams, RelativeChannelParams, luigi.Task):
    def requires(self):
        return Files()

    def output(self):
        return LocalNpy(self.to_experiment_path((f'{self.fluorophore}_{self.polarization}_wrt_'
                                                 f'{self.relative_fluorophore}_{self.relative_polarization}'
                                                 f'.shift.npy')))

    def run(self):
        if (self.fluorophore == self.relative_fluorophore) and (self.polarization == self.relative_polarization):
            self.output().save((0., 0.))
        else:
            df = self.input().open()
            df = df[df['experiment_path'] == self.experiment_path]
            shifts = []
            for _, dg in df.groupby('position'):
                dg = dg.set_index(['fluorophore', 'polarization'])
                d = dg.loc[self.fluorophore, self.polarization]
                d_ref = dg.loc[self.relative_fluorophore, self.relative_polarization]
                with Image(path=d.path).output() as ims, Image(path=d_ref.path).output() as rel_ims:
                    for im, rel_im in zip(ims, rel_ims):
                        shift = feature.register_translation(im, rel_im, 100)[0]
                        shifts.append(shift)
            shifts = np.median(shifts, axis=0)
            self.output().save(shifts)


class Normalization(luigi.ExternalTask, FileParam):
    def output(self):
        return LocalTiff(self.path, protected=True)


class Metadata(FileParam, luigi.Task):
    """Extracts metadata from file and saves as json."""

    def requires(self):
        return Image(path=self.path)

    def output(self):
        return LocalJSON(self.to_results_file('.metadata.json'), protected=True)

    def run(self):
        metadata = self.parse_metadata(self.path)
        self.output().save(metadata, indent=0)

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
class CorrectedImage(luigi.WrapperTask, CorrectedImageParams):
    """Calculates a corrected image.

    Implements methods to get a corrected image when using as a subtask.
    """

    def subtasks(self):
        return {'image': MaskedImage(path=self.path)}

    def requires(self):
        return {'background': Background(path=self.path),
                'shift': Shift(experiment_path=self.experiment_path,
                               fluorophore=self.fluorophore,
                               polarization=self.polarization),
                # 'normalization': Normalization(path=self.normalization_path),
                'image_metadata': Metadata(path=self.path),
                # 'normalization_metadata': Metadata(path=self.normalization_path)
                }

    def open(self):
        with ExitStack() as cm:
            self.ims = cm.enter_context(self.subtasks()['image'])
            self.bgs = cm.enter_context(self.requires()['background'])
            self.shift = cm.enter_context(self.input()['shift'])
            self.normalization = cm.enter_context(self.input()['normalization'].open())
            self._stack = cm.pop_all()

        self.axis = tuple(range(-len(self.shift), 0))
        self.image_exposure = float(self.input()['image_metadata'].open()['ExposureTime1'])
        normalization_metadata = self.input()['normalization_metadata'].open()
        self.normalization_exposure = float(normalization_metadata['ExposureTime'])
        self.normalization_background = float(normalization_metadata['Background'])
        return self

    def close(self):
        self._stack.close()

    def corrected_image(self, item):
        im = self.ims[item]
        bg = self.bgs.get_background(item, im.shape)
        im = (im - bg) / self.image_exposure
        normalization = (self.normalization[0] - self.normalization_background) / self.normalization_exposure
        shift = -self.shift.astype(int)
        return np.roll(im / normalization, shift, axis=self.axis)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item, xy = item[0], item[1:]
        else:
            xy = slice(None)
        return self.corrected_image(item)[xy]

    def __len__(self):
        return len(self.ims)

    @property
    def shape(self):
        return self.ims.shape


@delegates
class CorrectedBackground(CorrectedImageParams, luigi.Task):
    """Calculates the background distribution from a corrected image."""
    sigma = luigi.FloatParameter()
    window = luigi.IntParameter()
    threshold = luigi.FloatParameter()

    def subtasks(self):
        return CorrectedImage(**self.corrected_image_params)

    def output(self):
        return LocalNpz(self.to_results_file('.corrected_bg_rv.npz'))

    def run(self):
        with self.subtasks() as ims:
            bg_rvs = {}
            smo_rv = functions.smo_rv(2, self.sigma, self.window)
            for i, im in enumerate(ims):
                smo_mask = background.smo_mask(im, self.sigma, self.window, threshold=self.threshold, smo_rv=smo_rv)
                bg = im[smo_mask]
                med, *iqr = np.percentile(bg, (50, 25, 75))
                bg = bg[bg < med + 10 * np.diff(iqr)]
                bg_rv = functions.HistogramRV.from_data(bg)
                bg_rvs[str(i)] = bg_rv._histogram
        self.output().save(**bg_rvs)
