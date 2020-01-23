import pathlib

import luigi
import pandas as pd
import parse
from donkeykong.target import LocalPandasPickle

from .parameters import DirectoryParams

martin_to_agus = {'CFP405': 'BFP_parallel',
                  'CFP4052': 'BFP_perpendicular',
                  'YFPPar': 'Cit_parallel',
                  'YFPPerp': 'Cit_perpendicular',
                  'mKatePar': 'Kate_parallel',
                  'mKatePerp': 'Kate_perpendicular'}

drugs = {'20181024': {'position < 21': 'DMSO',
                      'position > 20': 'STS'},
         '20190327': {'position < 11': 'H2O2_low',
                      'position > 10 and position < 17': 'DMSO',
                      'position > 16 and position < 27': 'STS',
                      'position > 26': 'H2O2'},
         '20190329': {'position < 11': 'DMSO',
                      'position > 10 and position < 21': 'STS',
                      'position > 20 and position < 31': 'Nocodazole',
                      'position > 30': 'Nocodazole_low'},
         '20190403': {'position < 16': 'DMSO',
                      'position > 15 and position < 31': 'STS',
                      'position > 30 and position < 51': 'Nocodazole',
                      'position > 50': 'Nocodazole_low'},
         '20190405': {'position < 21': 'DMSO',
                      'position > 20 and position < 41': 'Nocodazole',
                      'position > 40 and position < 61': 'Cisplatin',
                      'position > 60': 'Cisplatin_low'}}

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
        return LocalPandasPickle(pathlib.Path(self.results_path) / 'files.pandas')

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
            try:
                normalization_file = results_path / normalizations[date][polarization]
            except KeyError:  # Some normalizations haven't been calculated yet
                continue
            df.append(
                (date, position, fluorophore, polarization, str(experiment_path), str(file), str(normalization_file)))

        columns = ('date', 'position', 'fluorophore', 'polarization', 'experiment_path', 'path', 'normalization_path')
        df = pd.DataFrame(data=df, columns=columns)

        # Drug
        df['drug'] = 'N/A'
        for date in drugs.keys():
            for condition, drug in drugs[date].items():
                inds = df.query('date == "' + date + '" and ' + condition).index
                df.at[inds, 'drug'] = drug

        # Plasmid
        df['plasmid'] = 'Triple'
        inds = df.query('date == "20181024" and position <= 10').index
        df.at[inds, 'plasmid'] = '3x1'
        inds = df.query('date == "20181024" and position > 20 and position <= 30').index
        df.at[inds, 'plasmid'] = '3x1'

        # Time steps
        df['time_steps'] = df.date.apply(lambda x: time_steps[x])

        self.output().save(df)
