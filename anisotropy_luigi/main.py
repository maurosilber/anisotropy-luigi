import luigi

from anisotropy_luigi.cell import AnisotropyJump
from anisotropy_luigi.invalidation import invalidate_downstream


class RunAll(luigi.WrapperTask):
    """Dummy task to preprocess and call all tasks."""

    def requires(self):
        return AnisotropyJump()


if __name__ == '__main__':
    # Invalidation
    tasks = [RunAll()]
    tasks_to_invalidate = []
    invalidate_downstream(tasks, tasks_to_invalidate)

    # Running
    result = luigi.build(tasks, workers=1, local_scheduler=False, detailed_summary=True)
    print(result.summary_text)
