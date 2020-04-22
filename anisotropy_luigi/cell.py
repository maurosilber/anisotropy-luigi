import pathlib
from functools import reduce

import luigi
import numpy as np
import pandas as pd
from anisotropy import functions as anifuncs
from donkeykong.target import LocalNpz, LocalPandasPickle
from luigi.util import delegates
from scipy import ndimage
from scipy import stats
from scipy.ndimage.measurements import _stats

from .files import Files
from .image import CorrectedImage
from .parameters import DirectoryParams, RelativeChannelParams
from .tracking import TrackedLabels


@delegates
class Intensities(DirectoryParams, RelativeChannelParams, luigi.Task):
    """Calculates intensities curves for each cell."""
    dg = luigi.DictParameter()

    @property
    def dgs(self):
        """Indexed by channels."""
        return pd.DataFrame.from_dict(self.dg, 'index')

    def subtasks(self):
        subtasks = {}
        for fp, dg in self.dgs.groupby('fluorophore'):
            subtasks[fp] = dg.set_index('polarization', drop=False).apply(CorrectedImage.from_dict, axis=1).to_dict()
        return subtasks

    def requires(self):
        d_relative = (self.dgs
            .set_index(['fluorophore', 'polarization'], drop=False)
            .loc[(self.relative_fluorophore, self.relative_polarization)])
        return TrackedLabels.from_dict(d_relative)

    def output(self):
        key = next(iter(self.dg))
        d = self.dg[key]
        path, position = pathlib.Path(d['path']), d['position']
        return LocalNpz(self.to_results_path(path.with_name(f'{position}.cell.npz')))

    def run(self):
        tracked_labels = self.input().open()
        labels = np.arange(1, tracked_labels.max() + 1)

        # Count number of pixels per label
        cell_sizes = np.empty((len(tracked_labels), labels.size), dtype=np.int32)
        cell_on_border = np.zeros((len(tracked_labels), labels.size), dtype=bool)
        for i, tracked_label in enumerate(tracked_labels):
            count, _ = _stats(tracked_label, tracked_label, labels)
            cell_sizes[i] = count

            # # If labeled object is next to the image borders, don't consider its intensity.
            edges = (tracked_label[0], tracked_label[:, 0], tracked_label[-1], tracked_label[:, -1])
            labels_in_edges = reduce(np.union1d, edges)
            labels_in_edges = labels_in_edges[np.nonzero(labels_in_edges)]
            cell_on_border[i][labels_in_edges - 1] = True

        intensities = {}
        for fp, v in self.subtasks().items():
            intensities[(fp, 'non-saturated')] = np.empty((len(tracked_labels), labels.size))
            intensities[(fp, 'parallel')] = np.empty((len(tracked_labels), labels.size))
            intensities[(fp, 'perpendicular')] = np.empty((len(tracked_labels), labels.size))
            ims_par, ims_per = v['parallel'], v['perpendicular']
            with ims_par, ims_per:
                for i, (im_par, im_per, tracked_label) in enumerate(zip(ims_par, ims_per, tracked_labels)):
                    # im has a mask where it's saturated.
                    mask = im_par.mask | im_per.mask  # If any polarization is saturated
                    tracked_label = np.ma.masked_array(tracked_label, mask).filled(0)  # Set as background (0)

                    # Count non saturated pixels
                    intensities[(fp, 'non-saturated')][i] = _stats(tracked_label, tracked_label, labels)[0]
                    # Sum non saturated intensities
                    intensities[(fp, 'parallel')][i] = ndimage.sum(im_par.filled(0), tracked_label, labels)
                    intensities[(fp, 'perpendicular')][i] = ndimage.sum(im_per.filled(0), tracked_label, labels)

        data = {'labels': labels}
        for i, (label, cell_size) in enumerate(zip(labels, cell_sizes.T)):
            cond = np.where(cell_size > 0)[0]
            data[f'{label}_index'] = cond
            data[f'{label}_cell_size'] = cell_size[cond]
            data[f'{label}_cell_on_border'] = cell_on_border[:, i][cond]
            for (fluorophore, polarization), intensity in intensities.items():
                data[f'{label}_{fluorophore}_{polarization}_intensity'] = intensity[:, i][cond]
        data['fluorophores'] = self.dgs.fluorophore.unique()
        data['channels'] = list(data.keys())

        self.output().save(**data)


class Anisotropy(luigi.WrapperTask, DirectoryParams, RelativeChannelParams):
    """Calculates anisotropy curves for each cell."""
    dg = luigi.DictParameter()
    npz = None

    def requires(self):
        return Intensities(dg=self.dg)

    def open(self):
        self.npz = self.input().open()
        return self

    def close(self):
        self.npz.close()

    def time(self, label):
        return self.npz[f'{label}_index']

    def cell_size(self, label):
        return self.npz[f'{label}_cell_size']

    def cell_on_border(self, label):
        return self.npz[f'{label}_cell_on_border']

    def intensity(self, polarization, fluorophore, label):
        return self.npz[f'{label}_{fluorophore}_{polarization}_intensity']

    def total_intensity(self, fluorophore, label):
        return anifuncs.total_intensity(self.intensity('parallel', fluorophore, label),
                                        self.intensity('perpendicular', fluorophore, label))

    def mean_intensity(self, fluorophore, label):
        return self.total_intensity(fluorophore, label) / self.cell_size(label)

    def anisotropy(self, fluorophore, label):
        return anifuncs.anisotropy_from_intensity(self.intensity('parallel', fluorophore, label),
                                                  self.intensity('perpendicular', fluorophore, label))

    @property
    def labels(self):
        return self.npz['labels']

    @property
    def max_label(self):
        return self.labels.max()

    @property
    def channels(self):
        return self.npz['channels']

    @property
    def fluorophores(self):
        return self.npz['fluorophores']


class CurvesSummary(DirectoryParams, RelativeChannelParams, luigi.Task):
    """DataFrame with a summary for each cell."""

    def requires(self):
        # Manual handling of task
        files = Files()
        if not files.complete():
            files.run()
        df = files.get_files()

        # Yielding all tasks
        for (date, position), dg in df.groupby(['date', 'position']):
            yield Anisotropy(dg=dg.to_dict('index'))

    def output(self):
        return LocalPandasPickle(self.results_path / 'curves_summary.pandas')

    def run(self):
        df = []
        for anisotropy in self.requires():
            row = next(iter(anisotropy.dg.values()))
            group_data = {key: row[key] for key in ('date', 'position')}
            with anisotropy as ani:
                for label in ani.labels:
                    data = {**group_data, 'label': label,
                            'cell_size': np.median(ani.cell_size(label)),
                            'cell_border': np.any(ani.cell_on_border(label)),
                            'length': min((~np.isnan(ani.anisotropy(fp, label))).sum() for fp in ani.fluorophores)
                            }

                    for fp in ani.fluorophores:
                        data[f'{fp}_total_intensity'] = np.median(ani.total_intensity(fp, label))
                        data[f'{fp}_mean_intensity'] = np.median(ani.mean_intensity(fp, label))
                        data[f'{fp}_anisotropy_iqr'] = stats.iqr(np.diff(ani.anisotropy(fp, label)))
                    df.append(data)

        df = pd.DataFrame(df)
        self.output().save(df)


class AnisotropyJumps(DirectoryParams, RelativeChannelParams, luigi.Task):
    filter_size = luigi.IntParameter()
    jump_threshold = luigi.FloatParameter()
    z_score_threshold = luigi.FloatParameter()
    jump_window_dilation = luigi.IntParameter()

    def requires(self):
        return {'files': Files(), 'curve_summary': CurvesSummary()}

    def output(self):
        return LocalPandasPickle(self.results_path / 'anisotropy_jumps.pandas')

    def run(self):
        files = self.requires()['files'].get_files().set_index(['date', 'position']).sort_index()

        curves_summary = self.input()['curve_summary'].open()
        curves_summary = (curves_summary.query('length > 50')
                          .query('BFP_anisotropy_iqr < 0.2')
                          .query('Kate_anisotropy_iqr < 0.2'))

        df = []
        for row in curves_summary.itertuples():
            dg = files.loc[(row.date, row.position)]
            with Anisotropy(dg=dg.reset_index().to_dict('index')) as ani:
                data = {}
                time = ani.time(row.label)

                for fp in ani.fluorophores:
                    a = ani.anisotropy(fp, row.label)
                    data[fp] = self.jump_filter(a, self.filter_size)

            masks = (self.discriminator(d['diff'], d['z_score'], self.jump_threshold, self.z_score_threshold) for d in
                     data.values())
            full_mask = self.full_mask(masks, dilation=self.jump_window_dilation)
            jumps, _ = ndimage.label(full_mask)
            full_mask = np.invert(full_mask, out=full_mask)  # np.ma.masked_array needs the inverse
            for jump_slice in ndimage.find_objects(jumps):
                jump_row = dict(row._asdict())
                for fp, d in data.items():
                    try:
                        jump = np.ma.masked_array(d['median'][jump_slice], full_mask[jump_slice])
                        jump_row[f'{fp}_jump_max'] = np.nanmax(jump)
                        jump_row[f'{fp}_jump_min'] = np.nanmin(jump)

                        ix = np.nanargmax(np.ma.masked_array(d['diff'][jump_slice], full_mask[jump_slice]))
                        jump_row[f'{fp}_jump_diff'] = d['diff'][jump_slice][ix]
                        jump_row[f'{fp}_jump_z_score'] = d['z_score'][jump_slice][ix]
                        jump_row[f'{fp}_jump_time'] = time[jump_slice][ix]
                    except ValueError:  # All-NaN slices
                        for key in ('max', 'min', 'diff', 'z_score', 'time'):
                            jump_row[f'{fp}_jump_{key}'] = np.nan

                df.append(jump_row)

        df = pd.DataFrame(df)
        self.output().save(df)

    @staticmethod
    def jump_filter(anisotropy, size):
        median = ndimage.filters.median_filter(anisotropy, size=size)
        mad = 1.4826 * ndimage.filters.median_filter(np.abs(anisotropy - median), size=size)
        diff = median[size:] - median[:-size]
        mad_sum = np.sqrt(mad[size:] ** 2 + mad[:-size] ** 2)
        z_score = diff / mad_sum
        return {'median': median[size:], 'diff': diff, 'z_score': z_score}

    @staticmethod
    def discriminator(diff, z_score, jump_threshold, z_score_threshold):
        return (diff > jump_threshold) & (np.abs(z_score) > z_score_threshold)

    @staticmethod
    def full_mask(masks, dilation):
        masks = (ndimage.binary_dilation(mask, np.ones(dilation)) for mask in masks)
        return reduce(np.logical_or, masks)


class JumpCurves(DirectoryParams, luigi.Task):
    def requires(self):
        return {'files': Files(), 'jumps': AnisotropyJumps()}

    def output(self):
        return LocalNpz(self.results_path / 'jump_curves.npz')

    def run(self):
        files = self.requires()['files'].get_files().set_index(['date', 'position']).sort_index()
        results = self.input()['jumps'].open()

        keys = ('index', 'cell_size',
                'BFP_parallel_intensity', 'BFP_perpendicular_intensity',
                'Kate_parallel_intensity', 'Kate_perpendicular_intensity',
                'Cit_parallel_intensity', 'Cit_perpendicular_intensity')
        data = {}
        for g, dg in results.groupby(['date', 'position']):
            with Anisotropy(dg=files.loc[g].reset_index().to_dict('index')) as ani:
                for label in dg.label:
                    for key in keys:
                        data[f'{g[0]}_{g[1]}_{label}_{key}'] = ani.npz[f'{label}_{key}']
        self.output().save(**data)
