import pathlib
from collections import defaultdict
from functools import reduce

import luigi
import numpy as np
import pandas as pd
from anisotropy import anisotropy as anifuncs
from donkeykong.target import LocalNpz, LocalPandasPickle
from luigi.util import delegates
from scipy import ndimage
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
    def indexed_dg(self):
        """Indexed by channels."""
        return pd.DataFrame.from_dict(self.dg, 'index').set_index(['fluorophore', 'polarization'], drop=False)

    def subtasks(self):
        return {k: CorrectedImage.from_dict(d) for k, d in self.indexed_dg.iterrows()}

    def requires(self):
        d_relative = self.indexed_dg.loc[(self.relative_fluorophore, self.relative_polarization)]
        return TrackedLabels.from_dict(d_relative)

    def output(self):
        d = self.indexed_dg.iloc[0]
        path, position = pathlib.Path(d.path), d.position
        return LocalNpz(self.to_results_path(path.with_name(f'{position}.cell.npz')))

    def run(self):
        tracked_labels = self.input().open()
        labels = np.arange(1, tracked_labels.max() + 1)
        channels = self.subtasks().keys()

        cell_sizes = np.empty((len(tracked_labels), labels.size))
        intensities = {k: np.empty((len(tracked_labels), labels.size)) for k in channels}

        for i, tracked_label in enumerate(tracked_labels):
            count, _ = _stats(tracked_label, tracked_label, labels)
            cell_sizes[i] = count

        for k, v in self.subtasks().items():
            with v as ims:
                for i, (im, tracked_label) in enumerate(zip(ims, tracked_labels)):
                    # im has a mask where it's saturated.
                    intensities[k][i] = ndimage.sum(im.filled(np.nan), tracked_label, labels)

                    # If labeled object is next to the image borders, don't consider its intensity.
                    edges = (tracked_label[0], tracked_label[:, 0], tracked_label[-1], tracked_label[:, -1])
                    labels_in_edges = reduce(np.union1d, edges)
                    labels_in_edges = labels_in_edges[np.nonzero(labels_in_edges)]
                    intensities[k][i][labels_in_edges - 1] = np.nan

        data = {'labels': labels, 'channels': list(channels)}
        for i, (label, cell_size) in enumerate(zip(labels, cell_sizes.T)):
            cond = np.where(cell_size > 0)[0]
            data[f'{label}_index'] = cond
            data[f'{label}_cell_size'] = cell_size[cond]
            for (fluorophore, polarization), intensity in intensities.items():
                data[f'{label}_{fluorophore}_{polarization}_intensity'] = intensity[:, i][cond]

        self.output().save(**data)


class Anisotropy(luigi.WrapperTask, DirectoryParams, RelativeChannelParams):
    """Calculates anisotropy curves for each cell."""
    dg = luigi.DictParameter()

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

    def intensity(self, polarization, fluorophore, label):
        return self.npz[f'{label}_{fluorophore}_{polarization}_intensity']

    def total_intensity(self, fluorophore, label):
        return sum(self.intensity(p, fluorophore, label) for p in ('parallel', 'perpendicular'))

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
        return np.unique(self.npz['channels'][:, 0])


class AnisotropyJump(DirectoryParams, RelativeChannelParams, luigi.Task):
    filter_size = luigi.IntParameter()
    jump_threshold = luigi.FloatParameter()
    zscore_threshold = luigi.FloatParameter()

    def requires(self):
        # Manual handling of task
        files = Files()
        if not files.complete():
            files.run()
        df = files.output().open()

        # Yielding all tasks
        for (date, position), dg in df.groupby(['date', 'position']):
            yield Anisotropy(dg=dg.to_dict('index'))

    def output(self):
        return LocalPandasPickle(self.results_path / 'anisotropy.pandas')

    def run(self):
        df = []
        for anisotropy in self.requires():
            group_data = {key: anisotropy.dg[key] for key in ('date', 'position')}
            with anisotropy:
                for label in anisotropy.labels:
                    calcs = defaultdict(dict)

                    # Filter each channel
                    for fp in ('Cit', 'Kate', 'BFP'):
                        calcs[fp] = self.jump_filter(anisotropy.anisotropy(fp, label), self.filter_size)
                        calcs[fp]['mask'] = self.discriminator(calcs[fp]['median_diff'], calcs[fp]['median_zscore'],
                                                               self.jump_threshold, self.zscore_threshold)

                    # Join masks
                    mask = reduce(np.logical_or, (x['mask'] for x in calcs.values()))
                    if mask.any():
                        # Get jump index
                        tmp = []
                        for calc in calcs.values():
                            median_diff = calc['median_diff']
                            median_zscore = calc['median_zscore']

                            jump_max = np.nanmax(median_diff[mask])
                            if np.isnan(jump_max):
                                continue
                            jump_ix = np.where(median_diff == jump_max)[0]
                            jump_ix = int(np.median(jump_ix))  # If multiple indexes, keep median
                            jump_zscore = median_zscore[mask].min()

                            tmp.append((jump_ix, jump_max, jump_zscore))
                        if len(tmp) > 0:
                            has_jump = True
                            jump_ix, jump_max, jump_zscore = max(tmp, key=lambda x: x[1])
                            jump_ix += self.filter_size // 2
                        else:
                            has_jump = False
                            jump_ix = jump_max = jump_zscore = np.nan
                    else:
                        has_jump = False
                        jump_ix = jump_max = jump_zscore = np.nan

                    # Save row
                    row = {**group_data, 'label': label,
                           'jump_ix': jump_ix, 'jump_max': jump_max, 'jump_zscore': jump_zscore}
                    df.append(row)

        df = pd.DataFrame(df)
        self.output().save(df)

    @staticmethod
    def jump_filter(anisotropy, size):
        median = ndimage.filters.median_filter(anisotropy, size=size)
        mad = (ndimage.filters.percentile_filter(anisotropy, 75, size=size)
               - ndimage.filters.percentile_filter(anisotropy, 25, size=size))
        median_diff = median[size:] - median[:-size]
        mad_sum = mad[size:] + mad[:-size]
        median_zscore = median_diff / mad_sum
        return {'median': median, 'median_diff': median_diff, 'median_zscore': median_zscore}

    @staticmethod
    def discriminator(median_diff, median_zscore, jump_threshold, zscore_threshold):
        return (median_diff > jump_threshold) & (np.abs(median_zscore) > zscore_threshold)
