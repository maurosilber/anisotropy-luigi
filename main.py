import luigi

from anisotropy import Intensity
from files import df


class RunAll(luigi.WrapperTask):
    def requires(self):
        for (date, position), dg in df.groupby(['date', 'position']):
            dg = dg.set_index(['fp', 'polarization'])
            d_ref = dg.loc['Cit', 'parallel']
            for d in dg.itertuples():
                yield Intensity(path=d.file,
                                rel_path=d_ref.file,
                                g_factor_path=d.g_factor,
                                g_factor_rel_path=d_ref.g_factor)


if __name__ == '__main__':
    result = luigi.build([RunAll()], local_scheduler=False, detailed_summary=True)
    print(result.summary_text)
