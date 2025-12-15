import contextlib as cl
import os
import re
import shutil as su


def get_files(
    folder: str,
    sort: bool = True,
    fullpath: bool = True,
    extensions: list[str] = None,
    recursive: bool = False,
) -> list[str]:
    """Return the list of files found in the input folder.

    :param folder: input folder
    :param sort: sort list by name
    :param fullpath: return full paths of files
    :param extensions: filter files by extensions
    :param recursive: recursive folder walk

    :return: list of found files (empty if an error occurred)
    """
    files = []
    if extensions is not None:
        extensions = [e.lower() for e in extensions]
    with cl.suppress(OSError):
        items = os.listdir(folder)
        for i in items:
            j = os.path.join(folder, i)
            ext = os.path.splitext(i)[1].lower()[1:]
            if os.path.isfile(j) and (extensions is None or ext in extensions):
                files.append(os.path.abspath(j) if fullpath else i)
            elif recursive and os.path.isdir(j):
                files += get_files(j, sort, fullpath, extensions, recursive)
    return sorted(files) if sort else files


def is_empty(folder: str) -> bool:
    """Check if a folder is empty.

    :param folder: folder to check

    :return: True if folder is empty, False otherwise
    """
    return not os.listdir(folder)


def get_folders(
    folder: str,
    hidden: bool = False,
    sort: bool = True,
    recursive: bool = False,
    fullpath: bool = True,
) -> list[str]:
    """Return the list of subfolders found in the input folder.

    :param folder: input folder
    :param hidden: include hidden folders
    :param sort: sort list by name
    :param recursive: recursive folder walk
    :param fullpath: return full paths of folders

    :return: list of found folder (empty if an error occurred)
    """
    print("Scanning folders...")

    folders = []
    try:
        for root, dirs, _ in os.walk(folder):
            if not hidden:
                dirs = [d for d in dirs if not d.startswith(".")]
            if fullpath:
                folders.extend([os.path.join(root, d) for d in dirs])
            else:
                folders.extend(dirs)
            if not recursive:
                break
        return sorted(folders) if sort else folders
    except StopIteration:
        return []


def count_lines(filename: str) -> int:
    """Return the number of lines of a text file.

    :param filename: input file

    :return: number of text lines
    """
    with open(filename, "r") as file:
        return len(file.readlines())


def get_basename(filename: str) -> str:
    """Extract file basename.

    :param filename: input filename

    :return: file basename
    """
    if filename.endswith("/"):
        filename = filename[:-1]
    return os.path.basename(filename)


def get_namefile(filename: str) -> str:
    """Extract file name without extension.

    :param filename: input filename

    :return: file name
    """
    return os.path.splitext(get_basename(filename))[0]


def get_parent(filename: str) -> str:
    """Get containing folder.

    :param filename: input filename

    :return: file folder
    """
    return os.path.dirname(filename)


def get_relpath(path: str) -> str:
    """Return the relative path to the input file.

    :param path: input file

    :return: relative path
    """
    return os.path.relpath(path)


def get_abspath(path: str) -> str:
    """Return the absolute path to the input file.

    :param path: input file

    :return: absolute path
    """
    return os.path.abspath(path)


def count_files(folder: str, recursive: bool = False) -> int:
    """Return the number of files found in the input folder.

    :param folder: input folder
    :param recursive: recursive search

    :return: number of files found
    """
    if recursive:
        return sum(len(files) for _, _, files in os.walk(folder))

    files = [f for r, d, f in os.walk(folder)]
    return len([item for sublist in files for item in sublist])


def check_folder(folder: str) -> bool:
    """Check if a folder exists in the provided path.

    :param folder: folder to check

    :return: True if folder exists, False otherwise
    """
    return folder and os.path.exists(folder) and os.path.isdir(folder)


def check_folders(path: str, folders: list[str]) -> bool:
    """Check if all folders exist in the provided paths.

    :param path: base path
    :param folders: list of folders to check

    :return: True if all folders exist, False otherwise
    """
    return all(check_folder(join_paths(path, f)) for f in folders)


def check_file(filename: str) -> bool:
    """Check if a file exists in the provided path.

    :param filename: file to check

    :return: True if file exists, False otherwise
    """
    return filename and os.path.exists(filename) and os.path.isfile(filename)


def copy_file(src: str, dst: str) -> None:
    """Copy a file from source to destination.

    :param src: source file
    :param dst: destination file
    """
    su.copyfile(src, dst)


def join_paths(path1: str, path2: str) -> str:
    """Join multiple path components into a single path.

    :param path1: first path
    :param path2: second path

    :return: joined path
    """
    return os.path.join(path1, path2)


def rename_file(old_name: str, new_name: str) -> None:
    """Rename a file or directory.

    :param old_name: old name
    :param new_name: new name
    """
    old_ext = get_ext(old_name)
    new_name = f"{remove_ext(new_name)}.{old_ext}"
    os.rename(old_name, new_name)


def contains_files(folder: str, files: list) -> bool:
    """Check if a folder contains files.

    :param folder: folder to check
    :param files: list of files to check

    :return: True if folder contains files, False otherwise
    """
    return all(os.path.exists(os.path.join(folder, f)) for f in files)


def file_size(filename: str) -> int:
    """Compute the size in bytes of a file on disk.

    :param filename: file to analyze

    :return: number of bytes on disk
    """
    try:
        return os.path.getsize(filename)
    except OSError:
        return 0


def folder_size(folder: str, recursive: bool = False) -> int:
    """Compute the size in bytes of a folder on disk.

    :param folder: folder to analyze
    :param recursive: descend into subfolders

    :return: total folder bytes
    """
    try:
        if not recursive:
            size = sum(
                file_size(os.path.join(folder, f))
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            )
        else:
            size = 0
            for root, _, filenames in os.walk(folder):
                for filename in filenames:
                    size += file_size(os.path.join(root, filename))
        return size
    except OSError:
        return 0


def clean_folder(folder: str, recursive: bool = False) -> None:
    """Remove all items contained in a folder.

    :param folder: folder to clean
    :param recursive: descend into subfolders
    """
    with cl.suppress(OSError):
        for i in os.listdir(folder):
            item = os.path.join(folder, i)
            if os.path.isfile(item):
                os.remove(item)
            elif recursive and os.path.isdir(item):
                su.rmtree(item, ignore_errors=True)


def init_folder(folder: str, clean: bool = True) -> None:
    """Prepare an empty folder for output.

    :param folder: folder to be initialized
    :param clean: remove existing files
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif clean:
        clean_folder(folder, recursive=True)


def human_size(total: int, binary: bool = False) -> str:
    """Convert total bytes into human-readable format.

    :param total: number of bytes
    :param binary: use binary suffixes

    :return: human-readable string
    """
    units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    if binary:
        units = [f"{unit}i" for unit in units]
        factor = 1024
    else:
        factor = 1000
    suffix = "B"
    for unit in units:
        if abs(total) < factor:
            return f"{total:3.1f} {unit}{suffix}"
        total /= factor
    return f"{total:.1f} {units[-1]}{suffix}"


def get_ext(filename: str) -> str | None:
    """Return the file extension (alphanumeric only) from the filename, or None if none found."""
    # 1) Just the basename, so "/foo/bar.txt" → "bar.txt"
    base = os.path.basename(filename)

    # 2) Skip plain dotfiles (".bashrc", ".env", etc.)
    if base.startswith(".") and base.count(".") == 1:
        return None

    # 3) Find the last dot
    idx = base.rfind(".")
    #    - idx <= 0: no dot at all (idx == -1), or dot is the first char (hidden file)
    if idx <= 0 or idx == len(base) - 1:
        return None

    # 4) Grab the candidate extension
    ext = base[idx + 1 :]

    # 5) Validate: must be letters and/or digits only
    if re.fullmatch(r"[A-Za-z0-9]+", ext):
        return ext.lower()

    return None


def set_ext(filename: str, extension: str) -> str:
    """Ensure filename ends with the given alphanumeric extension."""
    # Normalize desired extension
    new_ext = extension.lstrip(".").lower()

    # If it already has that ext, return as-is
    cur = get_ext(filename)
    if cur == new_ext:
        return filename

    # Otherwise strip any existing ext, then append the new one
    filename = remove_ext(filename)
    # Guard against a trailing dot
    if filename.endswith("."):
        filename = filename[:-1]

    return f"{filename}.{new_ext}"


def remove_ext(filename: str) -> str:
    """Remove the (alphanumeric) extension from filename, if any."""
    ext = get_ext(filename)
    if ext is None:
        return filename

    # Split off dir + basename
    dirpath = os.path.dirname(filename)
    base = os.path.basename(filename)

    # Chop off last ".ext"
    idx = base.rfind(".")
    new_base = base[:idx]

    # Re-join
    return os.path.join(dirpath, new_base) if dirpath else new_base


def add_suffix(filename: str, suffix: str) -> str:
    """Add suffix to filename.

    :param filename: input filename
    :param suffix: suffix to add

    :return: filename with suffix added
    """
    ext = get_ext(filename)
    result = remove_ext(filename) + suffix
    return set_ext(result, ext)


def add_prefix(filename: str, prefix: str) -> str:
    """Add prefix to filename.

    :param filename: input filename
    :param prefix: prefix to add

    :return: filename with prefix added
    """
    folder, original = os.path.split(filename)
    name, ext = os.path.splitext(original)
    return os.path.join(folder, f"{prefix}{name}{ext}")
