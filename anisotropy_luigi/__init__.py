import luigi
from luigi import configuration
from pkg_resources import resource_filename

package_config = resource_filename('anisotropy_luigi', 'luigi.cfg')
configuration.add_config_path(package_config)
configuration.add_config_path('luigi.cfg')  # Re-adds luigi.cfg as first place to check for a config file

from .files import Files
from .image import CorrectedImage
from .tracking import TrackedLabels
from .cell import AnisotropyJumps


class RunAll(luigi.WrapperTask):
    """Dummy task to preprocess and call all tasks."""

    def __init__(self, *args, **kwargs):
        import mkl
        mkl.set_num_threads(1)
        super().__init__(*args, **kwargs)

    def requires(self):
        return AnisotropyJumps()
