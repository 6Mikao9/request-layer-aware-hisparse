import os
from pathlib import Path
import stat


def is_private_path(path):
    """Deny reserved directory components and any deployment-specific names.

    Set HISPARSE_PRIVATE_PATH_COMPONENTS to the additional components,
    separated by the platform's path-list separator.
    """
    names = {".private", "private-data"}
    names.update(name.casefold() for name in os.environ.get(
        "HISPARSE_PRIVATE_PATH_COMPONENTS", "").split(os.pathsep) if name)
    return any(part.casefold() in names for part in path.parts)


def _named_path(path, *, writing=False):
    path = Path(path)
    if ".." in path.parts or is_private_path(path):
        raise ValueError("Private area/traversal cannot be an experiment artifact")
    path = path.absolute()
    if is_private_path(path):
        raise ValueError("Private area cannot be an experiment artifact")
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            if writing and part == path:
                continue
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse artifacts are unsupported")
        if part != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError("Artifact parent must be a directory")
        if part == path:
            if writing:
                raise FileExistsError("Use a new report file")
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Artifact must be a regular file")
    return path
