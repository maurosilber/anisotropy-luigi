import pathlib

import luigi
import numpy as np
import pandas as pd
from luigi.task import flatten
from luigi.tools import deps
from skimage.external.tifffile import TiffFile


def get_task_requires(task):
    if hasattr(task, 'subtasks'):
        return set(flatten(task.requires()) + flatten(task.subtasks()))
    else:
        return set(flatten(task._requires()))


# Monkey patching
deps.get_task_requires = get_task_requires


def invalidate(task):
    outputs = flatten(task.output())
    for output in outputs:
        if output.exists():
            output.remove()


def invalidate_downstream(tasks, tasks_to_invalidate):
    for task_to_invalidate in tasks_to_invalidate:
        for task in tasks:
            for dep in deps.find_deps(task, task_to_invalidate):
                invalidate(dep)


class LocalTarget(luigi.LocalTarget):
    def __init__(self, *args, **kwargs):
        self.protected = kwargs.pop('protected', False)
        super().__init__(*args, **kwargs)

    def remove(self):
        if not self.protected:
            pathlib.Path(self.path).unlink()


def set_default_from_config(cls):
    params = cls.get_params()
    for param_name, param_value in params:
        default = param_value._get_value_from_config(cls.__name__, param_name)
        param_value._default = default
    return cls


@set_default_from_config
class DirectoryParams(luigi.Config):
    data_path = luigi.Parameter(description='Path to base data folder.')
    results_path = luigi.Parameter(description='Path to base result folder.')


class FileParam(DirectoryParams):
    path = luigi.Parameter(description='Path to image.')

    def results_file(self, extension):
        return self.relative_path(self.path, self.data_path, self.results_path).with_suffix(extension)

    @staticmethod
    def relative_path(file, root, new_root):
        file = pathlib.Path(file)
        root = pathlib.Path(root)
        new_root = pathlib.Path(new_root)
        new_file = new_root / '/'.join(file.parts[len(root.parts):])
        return new_file


class RelFileParam(luigi.Config):
    rel_path = luigi.Parameter(description='Path to image from which shift is calculated relative to.')


class CorrectedImageParams(FileParam, RelFileParam):
    normalization_path = luigi.Parameter()


class CorrectedPairParams(CorrectedImageParams):
    normalization_rel_path = luigi.Parameter()


class LocalTiff(LocalTarget):
    tif = None

    def open(self, pages=None):
        self.tif = TiffFile(self.path, pages=pages)
        return self.tif

    def close(self):
        if self.tif is not None:
            self.tif.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(self):
        return NotImplementedError


class LocalNpy(LocalTarget):
    def open(self, mode='r'):
        return np.load(self.path, mmap_mode='r')

    def save(self, array):
        np.save(self.path, array)


class LocalNpz(LocalTarget):
    npz = None

    def open(self, mode='r'):
        self.npz = np.load(self.path, allow_pickle=True)
        return self.npz

    def close(self):
        self.npz.close()

    def save(self, *args, **kwargs):
        np.savez(self.path, *args, **kwargs)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.npz.close()


class LocalPandas(LocalTarget):
    def open(self):
        return pd.read_pickle(self.path)

    def save(self, df):
        pd.to_pickle(df, self.path)
