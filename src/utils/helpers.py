"""
Helper utilities for RAG pipeline
"""

import pickle
import time
from pathlib import Path
from typing import Any
import hashlib
import json


def save_pickle(obj: Any, filepath: str | Path) -> None:
    """
    Save object to pickle file

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str | Path) -> Any:
    """
    Load object from pickle file

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str | Path) -> None:
    """
    Save object to JSON file

    Args:
        obj: Object to save (must be JSON serializable)
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(filepath: str | Path) -> Any:
    """
    Load object from JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(filepath: str | Path) -> Path:
    """
    Ensure directory exists, create if not

    Args:
        filepath: Directory path

    Returns:
        Path object
    """
    path = Path(filepath)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(filepath: str | Path, unit: str = 'MB') -> float:
    """
    Get file size in specified unit

    Args:
        filepath: Path to file
        unit: Unit for size ('B', 'KB', 'MB', 'GB')

    Returns:
        File size in specified unit
    """
    path = Path(filepath)
    if not path.exists():
        return 0.0

    size_bytes = path.stat().st_size

    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }

    return size_bytes / units.get(unit, 1)


def get_file_hash(filepath: str | Path) -> str:
    """
    Get MD5 hash of file

    Args:
        filepath: Path to file

    Returns:
        MD5 hash string
    """
    path = Path(filepath)
    if not path.exists():
        return ""

    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(num: int, suffix: str = "") -> str:
    """
    Format number with commas

    Args:
        num: Number to format
        suffix: Optional suffix to add

    Returns:
        Formatted number string
    """
    return f"{num:,}{suffix}"


class Timer:
    """
    Context manager for timing code execution
    """

    def __init__(self, name: str = "Operation", logger=None):
        """
        Initialize timer

        Args:
            name: Name of operation being timed
            logger: Optional logger instance
        """
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        message = f"{self.name} completed in {format_time(elapsed)}"
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    @property
    def elapsed(self) -> float:
        """Get elapsed time if timer is running"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """
    Split list into chunks

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries recursively

    Args:
        dict1: First dictionary
        dict2: Second dictionary (overrides dict1)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to max length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


if __name__ == "__main__":
    # Test utilities
    print("Testing helper utilities...")

    # Test file operations
    test_file = Path("test_temp.pkl")
    test_data = {"key": "value", "number": 42}

    save_pickle(test_data, test_file)
    loaded = load_pickle(test_file)
    assert loaded == test_data
    print("✅ Pickle save/load works")

    # Test timer
    with Timer("Test operation"):
        time.sleep(0.1)
    print("✅ Timer works")

    # Test formatting
    print(f"File size: {get_file_size(test_file, unit='KB'):.2f} KB")
    print(f"Time: {format_time(3665)}")
    print(f"Number: {format_number(1000000)}")

    # Cleanup
    test_file.unlink()

    print("\n✅ All utilities working!")
