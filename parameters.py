import pathlib

import luigi


def set_default_from_config(cls):
    """Decorator to add default values to parameters from config file.

    By default, if no value is passed to a parameter on Task instantiation, Luigi searches for a specified value in the
    configuration file. But it uses the class name to search in the config file, hence it doesn't work with subclasses.
    This solves that issue.

    @set_default_from_config
    class TaskA(luigi.Config):
       param_a = luigi.Parameter()  # no default specified

    it equivalent to

    class TaskA(luigi.Config):
       param_a = luigi.Parameter(default=<default_value_from_config>)
    """
    params = cls.get_params()
    for param_name, param_value in params:
        default = param_value._get_value_from_config(cls.__name__, param_name)
        param_value._default = default
    return cls


@set_default_from_config
class DirectoryParams(luigi.Config):
    data_path = luigi.Parameter(description='Path to base data folder.', significant=False)
    results_path = luigi.Parameter(description='Path to base result folder.')

    def to_results_path(self, path):
        return self.change_root_path(path, self.data_path, self.results_path)

    @staticmethod
    def change_root_path(file, root, new_root):
        """Changes the root of file to new_root.

        :param root: /path/to/ROOT
        :param new_root: /path/to/NEW_ROOT
        :param file: /path/to/ROOT/path/to/FILE

        :return /path/to/NEW_ROOT/path/to/FILE
        """
        file = pathlib.Path(file)
        root = pathlib.Path(root)
        new_root = pathlib.Path(new_root)
        new_file = new_root / '/'.join(file.parts[len(root.parts):])
        return new_file


class FileParam(DirectoryParams):
    path = luigi.Parameter(description='Path to image.')

    def to_results_file(self, extension):
        """Generates a path to save a results file in the results directory with the given extension.

        Replicates the folder structure from data directory in the results directory.

        :param str extension: must start with a dot. Example: ".background.tif"
        """
        return super().to_results_path(self.path).with_suffix(extension)


class ExperimentParam(DirectoryParams):
    experiment_path = luigi.Parameter()

    def to_experiment_path(self, filename):
        """Generates a path to save a results file in the results directory with the given extension.

        Replicates the folder structure from data directory in the results directory.

        :param str extension: must start with a dot. Example: ".background.tif"
        """
        return super().to_results_path(self.experiment_path) / filename


class ChannelParams(luigi.Config):
    fluorophore = luigi.Parameter()
    polarization = luigi.Parameter()


@set_default_from_config
class RelativeChannelParams(luigi.Config):
    relative_fluorophore = luigi.Parameter()
    relative_polarization = luigi.Parameter()


class CorrectedImageParams(FileParam, ExperimentParam, ChannelParams):
    normalization_path = luigi.Parameter()

    @property
    def corrected_image_params(self):
        return {'path': self.path,
                'experiment_path': self.experiment_path,
                'fluorophore': self.fluorophore,
                'polarization': self.polarization,
                'normalization_path': self.normalization_path}


class CorrectedPairParams(CorrectedImageParams, RelativeChannelParams):
    relative_path = luigi.Parameter()
    relative_normalization_path = luigi.Parameter()

    @property
    def corrected_relative_image_params(self):
        return {'path': self.relative_path,
                'experiment_path': self.experiment_path,
                'fluorophore': self.relative_fluorophore,
                'polarization': self.relative_polarization,
                'normalization_path': self.relative_normalization_path}
