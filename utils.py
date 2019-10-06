import pathlib

import luigi
import numpy as np
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


class FileParam(luigi.Config):
    path = luigi.Parameter(description='Path to image.')


class RelFileParam(luigi.Config):
    rel_path = luigi.Parameter(description='Path to image from which shift is calculated relative to.')


class CorrectedImageParams(FileParam, RelFileParam):
    g_factor_path = luigi.Parameter()


class CorrectedPairParams(CorrectedImageParams):
    g_factor_rel_path = luigi.Parameter()


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
