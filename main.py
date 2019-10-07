import luigi

from anisotropy import Intensity
from files import Files
from utils import invalidate_downstream


class RunAll(luigi.WrapperTask):
    def requires(self):
        # Manual handling of task
        files = Files()
        if not files.complete():
            files.run()
        df = files.output().open()

        # Filter
        df = df.query('position == 18')

        # Yielding all tasks
        for (date, position), dg in df.groupby(['date', 'position']):
            dg = dg.set_index(['fp', 'polarization'])
            d_ref = dg.loc['Cit', 'parallel']
            for d in dg.itertuples():
                yield Intensity(path=d.file,
                                rel_path=d_ref.file,
                                normalization_path=d.normalization,
                                normalization_rel_path=d_ref.normalization)


if __name__ == '__main__':
    tasks_to_invalidate = []
    tasks = [RunAll()]
    invalidate_downstream(tasks, tasks_to_invalidate)

    result = luigi.build(tasks, workers=1, local_scheduler=False, detailed_summary=True)
    print(result.summary_text)
