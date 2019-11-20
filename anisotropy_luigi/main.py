import luigi

from anisotropy_luigi.cell import AnisotropyJump


class RunAll(luigi.WrapperTask):
    """Dummy task to preprocess and call all tasks."""

    def requires(self):
        return AnisotropyJump()
