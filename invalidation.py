from luigi.task import flatten
from luigi.tools import deps


def get_task_requires(task):
    """Returns task requirements.

    Standard implementation in luigi.tools.deps doesn't include subtasks,
    as they are not considered a requirement.
    """
    if hasattr(task, 'subtasks'):
        return set(flatten(task.requires()) + flatten(task.subtasks()))
    else:
        return set(flatten(task.requires()))


# Monkey patching
deps.get_task_requires = get_task_requires


def invalidate(task):
    """Invalidates a task output.

    Outputs must implement an exists and remove method."""
    outputs = flatten(task.output())
    for output in outputs:
        if output.exists():
            output.remove()


def invalidate_downstream(tasks, tasks_to_invalidate):
    """Invalidates all downstream task.

    :param tasks: Iterable of task objects.
    :param tasks_to_invalidate: Iterable of task family name strings.
    """
    for task_to_invalidate in tasks_to_invalidate:
        for task in tasks:
            for dep in deps.find_deps(task, task_to_invalidate):
                invalidate(dep)
