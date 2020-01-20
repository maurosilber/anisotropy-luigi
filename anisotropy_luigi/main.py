import luigi

from .cell import AnisotropyJump


class RunAll(luigi.WrapperTask):
    """Dummy task to preprocess and call all tasks."""

    def __init__(self, *args, **kwargs):
        import mkl
        mkl.set_num_threads(1)
        super().__init__(*args, **kwargs)

    def requires(self):
        return AnisotropyJump()
