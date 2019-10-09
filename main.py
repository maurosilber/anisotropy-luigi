import luigi

from anisotropy import Intensity
from files import Files
from invalidation import invalidate_downstream


class RunAll(luigi.WrapperTask):
    """Dummy task to preprocess and call all tasks."""

    def requires(self):
        # Manual handling of task
        files = Files()
        if not files.complete():
            files.run()
        df = files.output().open()

        # Yielding all tasks
        for (date, position), dg in df.groupby(['date', 'position']):
            dg = dg.set_index(['fluorophore', 'polarization'], drop=False)
            d_ref = dg.loc['Cit', 'parallel']
            relative_params = {'relative_path': d_ref.file,
                               'relative_fluorophore': d_ref.fluorophore,
                               'relative_polarization': d_ref.polarization,
                               'relative_normalization_path': d_ref.normalization}
            for d in dg.itertuples():
                yield Intensity(path=d.file,
                                experiment_path=d.experiment_path,
                                fluorophore=d.fluorophore,
                                polarization=d.polarization,
                                normalization_path=d.normalization,
                                **relative_params)


if __name__ == '__main__':
    # Invalidation
    tasks = [RunAll()]
    tasks_to_invalidate = []
    invalidate_downstream(tasks, tasks_to_invalidate)

    # Running
    result = luigi.build(tasks, workers=1, local_scheduler=False, detailed_summary=True)
    print(result.summary_text)
