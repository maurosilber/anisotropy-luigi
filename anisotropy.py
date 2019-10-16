import pathlib

import luigi
import numpy as np
from luigi.util import delegates
from scipy import ndimage
from scipy.ndimage.measurements import _stats

from image import CorrectedImage
from parameters import DirectoryParams, RelativeChannelParams, DataFrameParameter
from tracking import TrackedLabels
from utils import LocalNpz


@delegates
class Intensities(DirectoryParams, RelativeChannelParams, luigi.Task):
    """Calculates intensities curves for each cell."""
    dg = DataFrameParameter()

    @property
    def indexed_dg(self):
        """Indexed by channels."""
        return self.dg.set_index(['fluorophore', 'polarization'], drop=False)

    def subtasks(self):
        return {k: CorrectedImage.from_dict(d) for k, d in self.indexed_dg.iterrows()}

    def requires(self):
        d_relative = self.indexed_dg.loc[(self.relative_fluorophore, self.relative_polarization)]
        return TrackedLabels.from_dict(d_relative)

    def output(self):
        d = self.dg.iloc[0]
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
                for i, (im, tracked_label) in enumerate(zip(ims.corrected_images(), tracked_labels)):
                    # im has a mask where it's saturated.
                    intensities[k][i] = ndimage.sum(im.filled(np.nan), tracked_label, labels)

        data = {}
        for i, (label, cell_size) in enumerate(zip(labels, cell_sizes.T)):
            cond = np.where(cell_size > 0)[0]
            data[f'{label}_index'] = cond
            data[f'{label}_cell_size'] = cell_size[cond]
            for (fluorophore, polarization), intensity in intensities.items():
                data[f'{label}_{fluorophore}_{polarization}_intensity'] = intensity[:, i][cond]

        self.output().save(**data)
