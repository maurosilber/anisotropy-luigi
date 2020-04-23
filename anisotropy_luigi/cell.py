import pathlib
from functools import reduce

import luigi
import numpy as np
import pandas as pd
from anisotropy import functions as anifuncs
from donkeykong.target import LocalNpz, LocalPandasPickle
from luigi.util import delegates
from scipy import ndimage, stats
from scipy.ndimage.filters import median_filter
from scipy.ndimage.measurements import _stats

from .files import Files
from .image import CorrectedImage
from .parameters import DirectoryParams, RelativeChannelParams
from .tracking import TrackedLabels


@delegates
class Intensities(DirectoryParams, RelativeChannelParams, luigi.Task):
    """Calculate intensities curves for each cell."""
    dg = luigi.DictParameter()

    @property
    def dgs(self):
        return pd.DataFrame.from_dict(self.dg, 'index')

    def subtasks(self):
        subtasks = {}
        for fp, dg in self.dgs.groupby('fluorophore'):
            subtasks[fp] = (dg
                            .set_index('polarization', drop=False)
                            .apply(CorrectedImage.from_dict, axis=1)
                            .to_dict())
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

        self.output().save(**data)


class CellCurves(luigi.WrapperTask, DirectoryParams, RelativeChannelParams):
    """Calculate curves for each cell.

    Intensity, anisotropy, cell size.
    """
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

    def non_saturated_size(self, fluorophore, label):
        return self.npz[f'{label}_{fluorophore}_non-saturated_intensity']

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
    def fluorophores(self):
        return self.npz['fluorophores']


class CellsSummary(DirectoryParams, RelativeChannelParams, luigi.Task):
    """DataFrame with a summary for each cell."""

    def requires(self):
        # Manual handling of task
        files = Files()
        if not files.complete():
            files.run()
        df = files.get_files()

        # Yielding all tasks
        for (date, position), dg in df.groupby(['date', 'position']):
            yield CellCurves(dg=dg.to_dict('index'))

    def output(self):
        return LocalPandasPickle(self.results_path / 'curves_summary.pandas')

    def run(self):
        df = []
        for curves in self.requires():
            row = next(iter(curves.dg.values()))
            group_data = {key: row[key] for key in ('date', 'position')}
            with curves:
                for label in curves.labels:
                    data = {**group_data, 'label': label,
                            'cell_size': np.median(curves.cell_size(label)),
                            'cell_border': np.any(curves.cell_on_border(label)),
                            }

                    for fp in curves.fluorophores:
                        data[f'{fp}_length'] = np.sum(curves.non_saturated_size(fp, label) > 0)
                        data[f'{fp}_total_intensity'] = np.median(curves.total_intensity(fp, label))
                        data[f'{fp}_mean_intensity'] = np.median(curves.mean_intensity(fp, label))
                        anisotropy = curves.anisotropy(fp, label)
                        data[f'{fp}_anisotropy_min'] = np.min(anisotropy)
                        data[f'{fp}_anisotropy_max'] = np.max(anisotropy)
                        data[f'{fp}_anisotropy_median'] = np.median(anisotropy)
                        data[f'{fp}_anisotropy_iqr'] = stats.iqr(np.diff(anisotropy))
                    data['length'] = max(data[f'{fp}_length'] for fp in curves.fluorophores)
                    df.append(data)

        df = pd.DataFrame(df)
        self.output().save(df)


class AnisotropyJumps(DirectoryParams, RelativeChannelParams, luigi.Task):
    # Summary parameters
    min_length = luigi.IntParameter()
    anisotropy_min = luigi.FloatParameter()
    anisotropy_max = luigi.FloatParameter()
    # Jump detection parameters
    filter_size = luigi.IntParameter()
    size_change_threshold = luigi.FloatParameter()
    jump_threshold = luigi.FloatParameter()
    z_score_threshold = luigi.FloatParameter()
    jump_window_dilation = luigi.IntParameter()

    def requires(self):
        return {'files': Files(), 'cells_summary': CellsSummary()}

    def output(self):
        return LocalPandasPickle(self.results_path / 'anisotropy_jumps.pandas')

    def run(self):
        files = self.requires()['files'].get_files().set_index(['date', 'position']).sort_index()

        # Prefilter cells from summary
        df_sum = self.input()['cells_summary'].open()
        # Filter cells with too few time points
        df_sum = df_sum.query(f'length > {self.min_length}')
        # Filter cells where no anisotropy channel is in the expected range
        cond = False
        for fp in ['BFP', 'Cit', 'Kate']:
            cond |= ((df_sum[f'{fp}_anisotropy_min'] < self.anisotropy_max)
                     & (df_sum[f'{fp}_anisotropy_max'] > self.anisotropy_min))
        df_sum = df_sum[cond]

        df = []
        for row in df_sum.itertuples():
            dg = files.loc[(row.date, row.position)]

            # Calculate parameter per channel to detect jump
            with CellCurves(dg=dg.reset_index().to_dict('index')) as c:
                time = c.time(row.label)
                border = c.cell_on_border(row.label)
                cell_size = c.cell_size(row.label)
                data = {fp: self.jump_filter(cell_size, c.anisotropy(fp, row.label), self.filter_size)
                        for fp in c.fluorophores}

            # Find a global mask of jump regions
            masks = (self.discriminator(d['diff'], d['z_score'], d['size_change'], border) for d in data.values())
            full_mask = self.full_mask(masks, dilation=self.jump_window_dilation)
            jumps, _ = ndimage.label(full_mask)

            # For each region, calculates location of jump (if any)
            full_mask = np.invert(full_mask, out=full_mask)  # np.ma.masked_array needs the inverse
            for jump_slice in ndimage.find_objects(jumps):
                jump_row = dict(row._asdict())
                for fp, d in data.items():
                    d = d.iloc[jump_slice]
                    try:
                        jump_row[f'{fp}_jump_max'] = d['median'].max()
                        jump_row[f'{fp}_jump_min'] = d['median'].min()

                        ix = d['diff'].idxmax()
                        jump_row[f'{fp}_jump_diff'] = d['diff'].loc[ix]
                        jump_row[f'{fp}_jump_z_score'] = d['z_score'][ix]
                        jump_row[f'{fp}_jump_time'] = time[ix]
                    except ValueError:  # All-NaN slices
                        for key in ('max', 'min', 'diff', 'z_score', 'time'):
                            jump_row[f'{fp}_jump_{key}'] = np.nan

                jump_row['size_max'] = d['size_median'].max()
                jump_row['size_min'] = d['size_median'].min()
                jump_row['size_change'] = d['size_change'].min()

                df.append(jump_row)

        df = pd.DataFrame(df)
        self.output().save(df)

    @staticmethod
    def jump_filter(cell_size, anisotropy, size):
        data = pd.DataFrame()
        data['size_median'] = median_filter(cell_size, size=size)
        data['size_change'] = np.exp(-np.log(data.size_median).rolling(size, center=True).apply(stats.iqr, raw=True))
        data['median'] = median_filter(anisotropy, size=size)
        data['mad'] = 1.4826 * median_filter(np.abs(anisotropy - data['median']), size=size)
        data['diff'] = data['median'].rolling(size, center=True).apply(stats.iqr, raw=True)
        data['z_score'] = data['diff'] / data['mad'].rolling(size, center=True).median()
        return data

    def discriminator(self, diff, z_score, size_change, border):
        # Excludes if area does not diminish when not in border
        cond = (size_change < self.size_change_threshold) | border
        cond &= diff > self.jump_threshold
        cond &= z_score > self.z_score_threshold
        return cond

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

        keys = ('index', 'cell_size', 'cell_on_border',
                'BFP_non-saturated_intensity', 'BFP_parallel_intensity', 'BFP_perpendicular_intensity',
                'Kate_non-saturated_intensity', 'Kate_parallel_intensity', 'Kate_perpendicular_intensity',
                'Cit_non-saturated_intensity', 'Cit_parallel_intensity', 'Cit_perpendicular_intensity')
        data = {}
        for g, dg in results.groupby(['date', 'position']):
            with CellCurves(dg=files.loc[g].reset_index().to_dict('index')) as ani:
                for label in dg.label:
                    for key in keys:
                        data[f'{g[0]}_{g[1]}_{label}_{key}'] = ani.npz[f'{label}_{key}']
        self.output().save(**data)


class CellJumpCurves:
    def __init__(self, npz, date, position, label):
        self.npz = npz
        self.prefix = f'{date}_{position}_{label}'

    def time(self):
        return self.npz[f'{self.prefix}_index']

    def cell_size(self):
        return self.npz[f'{self.prefix}_cell_size']

    def cell_on_border(self):
        return self.npz[f'{self.prefix}_cell_on_border']

    def intensity(self, polarization, fluorophore):
        return self.npz[f'{self.prefix}_{fluorophore}_{polarization}_intensity']

    def non_saturated_size(self, fluorophore):
        return self.npz[f'{self.prefix}_{fluorophore}_non-saturated_intensity']

    def total_intensity(self, fluorophore):
        return anifuncs.total_intensity(self.intensity('parallel', fluorophore),
                                        self.intensity('perpendicular', fluorophore))

    def mean_intensity(self, fluorophore):
        return self.total_intensity(fluorophore) / self.cell_size()

    def anisotropy(self, fluorophore):
        return anifuncs.anisotropy_from_intensity(self.intensity('parallel', fluorophore),
                                                  self.intensity('perpendicular', fluorophore))
