import luigi

from anisotropy import Intensities
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
            yield Intensities(dg=dg)


if __name__ == '__main__':
    # Invalidation
    tasks = [RunAll()]
    tasks_to_invalidate = []
    invalidate_downstream(tasks, tasks_to_invalidate)

    # Running
    result = luigi.build(tasks, workers=1, local_scheduler=False, detailed_summary=True)
    print(result.summary_text)
