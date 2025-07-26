from functools import wraps
from typing import Callable

from rich.progress import Progress, SpinnerColumn, TextColumn


def render_progress_task(description: str) -> Callable:
    """A decorator to show a progress spinner for a function call."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task_id = progress.add_task(description=description, total=None)
                try:
                    return func(*args, **kwargs)
                finally:
                    progress.remove_task(task_id)

        return wrapper

    return decorator
