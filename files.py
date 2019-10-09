import pathlib

import luigi
import pandas as pd
import parse

from parameters import DirectoryParams
from utils import LocalPandas

martin_to_agus = {'CFP405': 'BFP_parallel',
                  'CFP4052': 'BFP_perpendicular',
                  'YFPPar': 'Cit_parallel',
                  'YFPPerp': 'Cit_perpendicular',
                  'mKatePar': 'Kate_parallel',
                  'mKatePerp': 'Kate_perpendicular'}

time_steps = {'20181024': 12,
              '20190327': 215492 / 60000,
              '20190329': 5,
              '20190403': 4,
              '20190405': 268800 / 60000}

normalizations = {'20181024': {'parallel': 'normalizations/parallel_fluorescein_000_1024x1344.tif',
                               'perpendicular': 'normalizations/perpendicular_fluorescein_000_1024x1344.tif'},
                  '20190327': {'parallel': 'normalizations/parallel_fluorescein_000_1024x1344.tif',
                               'perpendicular': 'normalizations/perpendicular_fluorescein_000_1024x1344.tif'},
                  '20190329': {'parallel': 'normalizations/parallel_fluorescein_000_1024x1344.tif',
                               'perpendicular': 'normalizations/perpendicular_fluorescein_000_1024x1344.tif'},
                  '20190403': {'parallel': 'normalizations/parallel_fluorescein_000_512x672.tif',
                               'perpendicular': 'normalizations/perpendicular_fluorescein_000_512x672.tif'}
                  }


class Files(DirectoryParams, luigi.Task):
    """Generates a Pandas DataFrame with image file paths and associated metadata such as date, position, etc."""

    def output(self):
        return LocalPandas(pathlib.Path(self.results_path) / 'files.pandas')

    def run(self):
        data_path = pathlib.Path(self.data_path)
        results_path = pathlib.Path(self.results_path)

        df = []
        for file in data_path.rglob('*.TIF'):
            filename = file.stem
            parsed = parse.parse('{:d} {}', filename)
            if parsed is None:
                continue
            else:
                position, rest = parsed
                if rest in martin_to_agus:
                    rest = martin_to_agus[rest]
                fluorophore, polarization = parse.parse('{}_{}', rest)

            date = file.parent.stem
            experiment_path = file.parent
            normalization_file = results_path / normalizations[date][polarization]
            df.append((date, position, fluorophore, polarization, experiment_path, file, normalization_file))

        columns = ('date', 'position', 'fluorophore', 'polarization', 'experiment_path', 'file', 'normalization')
        df = pd.DataFrame(data=df, columns=columns)
        self.output().save(df)
