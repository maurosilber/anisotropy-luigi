from donkeykong.target import LocalPandasPickle
import numpy as np
import pandas as pd
from scipy import stats

from .cell import CellsSummary


class AgusDF(CellsSummary):
    def output(self):
        return LocalPandasPickle(self.results_path / "Agus_shaped.pandas")

    def run(self):
        df = []
        for curves in self.requires():
            row = next(iter(curves.dg.values()))
            group_data = {key: row[key] for key in ("date", "position", "time_steps")}
            with curves:
                for label in curves.labels:
                    data = {
                        **group_data,
                        "label": label,
                        "cell_size": np.median(curves.cell_size(label)),
                        "cell_border": np.any(curves.cell_on_border(label)),
                    }

                    for fp in curves.fluorophores:
                        data[f"{fp}_length"] = np.sum(
                            curves.non_saturated_size(fp, label) > 0
                        )
                        data[f"{fp}_total_intensity"] = np.median(
                            curves.total_intensity(fp, label)
                        )
                        data[f"{fp}_mean_intensity"] = np.median(
                            curves.mean_intensity(fp, label)
                        )
                        data[f"{fp}_total_intensity_variance"] = np.median(
                            curves.total_intensity(fp, label, variance=True)
                        )
                        data[f"{fp}_mean_intensity_variance"] = np.median(
                            curves.mean_intensity(fp, label, variance=True)
                        )

                        anisotropy = curves.anisotropy(fp, label)
                        data[f"{fp}_anisotropy_min"] = np.min(anisotropy)
                        data[f"{fp}_anisotropy_max"] = np.max(anisotropy)
                        data[f"{fp}_anisotropy_median"] = np.median(anisotropy)
                        data[f"{fp}_anisotropy_iqr"] = stats.iqr(np.diff(anisotropy))

                        # Agus
                        data[f"{fp}_anisotropy"] = anisotropy
                    cell_size = curves.cell_size(label)
                    data["area"] = cell_size
                    data["frame"] = np.arange(cell_size.size)
                    data["time"] = data["frame"] * data["time_steps"]

                    data["length"] = max(
                        data[f"{fp}_length"] for fp in curves.fluorophores
                    )
                    df.append(data)

        df = pd.DataFrame(df)
        self.output().save(df)
