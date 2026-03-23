"""File discovery utilities for H5AD preprocessing."""

import ctypes
import fnmatch
import gc
import re
from pathlib import Path


def force_release_memory() -> None:
    """Force Python GC and tell glibc to return memory to OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a glob pattern to a regex, supporting ** for recursive matching."""
    parts = pattern.split("**")
    regex_parts = [fnmatch.translate(p).rstrip(r"\Z") for p in parts]
    regex = r".*".join(regex_parts) + r"\Z"
    return re.compile(regex)


def matches_pattern(rel_path: str, pattern: str) -> bool:
    """Check if a relative path matches a glob pattern."""
    if "**" in pattern:
        regex = glob_to_regex(pattern)
        return regex.match(rel_path) is not None
    return fnmatch.fnmatch(rel_path, pattern)


def discover_h5ad_files_with_exclusions(
    input_pattern: str,
    exclude_patterns: list[str] | None = None,
    verbose: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Discover H5AD files and return both matched and excluded files.

    Args:
        input_pattern: Glob pattern for input files (e.g., "/data/**/*.h5ad").
        exclude_patterns: List of glob patterns to exclude.
        verbose: If True, print discovery details.

    Returns:
        Tuple of (matching_files, excluded_files), both sorted.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    exclude_patterns = exclude_patterns or []

    input_path = Path(input_pattern)

    # Find the first part that contains glob characters
    parts = input_path.parts
    base_parts: list[str] = []
    glob_parts: list[str] = []
    found_glob = False

    for part in parts:
        if found_glob or "*" in part or "?" in part or "[" in part:
            found_glob = True
            glob_parts.append(part)
        else:
            base_parts.append(part)

    if not base_parts:
        base_dir = Path(".")
    else:
        base_dir = Path(*base_parts)

    if not glob_parts:
        # No glob pattern, treat as literal path
        if input_path.exists() and input_path.suffix == ".h5ad":
            if verbose:
                print(f"\nFiles discovered (1):\n  {input_path}")
                print(f"\nFiles used (1):\n  {input_path}")
            return ([input_path], [])
        raise FileNotFoundError(f"No H5AD files found matching: {input_pattern}")

    glob_pattern = str(Path(*glob_parts))

    # Discover files
    all_files = sorted(base_dir.glob(glob_pattern))

    if verbose:
        print(f"\nFiles discovered ({len(all_files)}):")
        for f in all_files:
            print(f"  {f}")

    def is_excluded(file_path: Path) -> bool:
        try:
            rel_path = str(file_path.relative_to(base_dir))
        except ValueError:
            rel_path = str(file_path)
        return any(matches_pattern(rel_path, pattern) for pattern in exclude_patterns)

    matching_files: list[Path] = []
    excluded_files: list[Path] = []

    for f in all_files:
        if is_excluded(f):
            excluded_files.append(f)
        else:
            matching_files.append(f)

    matching_files = sorted(matching_files)
    excluded_files = sorted(excluded_files)

    if verbose:
        if excluded_files:
            print(f"\nFiles excluded ({len(excluded_files)}):")
            for f in excluded_files:
                print(f"  {f}")
        print(f"\nFiles used ({len(matching_files)}):")
        for f in matching_files:
            print(f"  {f}")

    if not matching_files:
        raise FileNotFoundError(f"No H5AD files found matching: {input_pattern}")

    return (matching_files, excluded_files)


def compute_output_path(input_file: Path, output_dir: Path) -> Path:
    """Compute output path in the output directory."""
    return output_dir / input_file.name
