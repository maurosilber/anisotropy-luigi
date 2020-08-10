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
                    row = {
                        **group_data,
                        "label": label,
                        "cell_size": np.median(curves.cell_size(label)),
                        "cell_border": np.any(curves.cell_on_border(label)),
                    }

                    for fp in curves.fluorophores:
                        row[f"{fp}_non_saturated_area"] = curves.non_saturated_size(
                            fp, label
                        )
                        row[f"{fp}_total_intensity"] = curves.total_intensity(fp, label)
                        row[f"{fp}_mean_intensity"] = curves.mean_intensity(fp, label)

                        anisotropy = curves.anisotropy(fp, label)
                        row[f"{fp}_anisotropy"] = anisotropy
                    cell_size = curves.cell_size(label)
                    row["area"] = cell_size
                    row["frame"] = np.arange(cell_size.size)
                    row["time"] = row["frame"] * row["time_steps"]

                    row["length"] = max(
                        np.sum(row[f"{fp}_non_saturated_area"] > 0) for fp in curves.fluorophores
                    )
                    df.append(row)

        df = pd.DataFrame(df)
        self.output().save(df)
