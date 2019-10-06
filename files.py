import pathlib

import pandas as pd
import parse

path = pathlib.Path('/home/mauro/Documents/anisotropia/')
data_path = path / 'data'

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

g_factor = {'20181024': {'parallel': path / 'params/parallel_fluorescein_000_1024x1344.tif',
                         'perpendicular': path / 'params/perpendicular_fluorescein_000_1024x1344.tif'},
            '20190327': {'parallel': path / 'params/parallel_fluorescein_000_1024x1344.tif',
                         'perpendicular': path / 'params/perpendicular_fluorescein_000_1024x1344.tif'},
            '20190329': {'parallel': path / 'params/parallel_fluorescein_000_1024x1344.tif',
                         'perpendicular': path / 'params/perpendicular_fluorescein_000_1024x1344.tif'},
            '20190403': {'parallel': path / 'params/parallel_fluorescein_000_512x672.tif',
                         'perpendicular': path / 'params/perpendicular_fluorescein_000_512x672.tif'},
            '20190405': {'parallel': None,
                         'perpendicular': None}}

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
        fp, polarization = parse.parse('{}_{}', rest)

    date = file.parent.stem
    df.append((date, position, fp, polarization, file, g_factor[date][polarization]))

df = pd.DataFrame(data=df, columns=('date', 'position', 'fp', 'polarization', 'file', 'g_factor'))
df = df.query('position == 18')
