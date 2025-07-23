from pathlib import Path


def find_project_root() -> Path:
    """
    Finds the project root by searching for a pyproject.toml file
    """
    current_dir = Path(__file__).resolve()

    # Traverse up the directory tree
    for parent in current_dir.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Project root not found.")


def get_data_path(filename: str) -> Path:
    """
    Constructs the absolute path to a file in the top-level 'data/' directory.
    """
    project_root = find_project_root()
    data_file_path = project_root / "data" / filename
    return data_file_path
