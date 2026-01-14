"""
File utilities for AI Toolkit.

This module provides convenient file operations.

Functions:
    read_file(file_path, encoding): Read file content as string
    write_file(file_path, content, encoding, mode): Write content to file
    read_lines(file_path, encoding): Read file content as list of lines
    write_lines(file_path, lines, encoding): Write lines to file
    read_json(file_path, encoding): Read JSON file
    write_json(file_path, data, encoding, indent): Write data to JSON file
    ensure_dir(dir_path): Ensure directory exists, create if it doesn't
    delete_file(file_path): Delete a file
    delete_dir(dir_path, recursive): Delete a directory
    copy_file(src, dst): Copy a file
    copy_dir(src, dst): Copy a directory recursively
    move_file(src, dst): Move a file
    file_exists(file_path): Check if file exists
    dir_exists(dir_path): Check if directory exists
    get_file_size(file_path): Get file size in bytes
    list_files(dir_path, pattern, recursive): List files in directory
    list_dirs(dir_path): List subdirectories in directory
    get_file_extension(file_path): Get file extension
    get_file_name(file_path, with_extension): Get file name
    join_paths(*paths): Join multiple paths
    get_absolute_path(file_path): Get absolute path
    get_relative_path(file_path, base_path): Get relative path
    create_temp_file(suffix, prefix, dir): Create a temporary file
    create_temp_dir(suffix, prefix, dir): Create a temporary directory
"""

import os
import json
import shutil
from typing import Any, List, Optional, Union
from pathlib import Path


def read_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read file content as string.
    
    Args:
        file_path: Path to file
        encoding: File encoding
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_file(file_path: Union[str, Path], content: str, 
               encoding: str = 'utf-8', mode: str = 'w') -> None:
    """
    Write content to file.
    
    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding
        mode: Write mode ('w' for write, 'a' for append)
    """
    file_path = Path(file_path)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, mode, encoding=encoding) as f:
        f.write(content)


def read_lines(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[str]:
    """
    Read file content as list of lines.
    
    Args:
        file_path: Path to file
        encoding: File encoding
        
    Returns:
        List of lines
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding=encoding) as f:
        return f.readlines()


def write_lines(file_path: Union[str, Path], lines: List[str], 
                encoding: str = 'utf-8') -> None:
    """
    Write lines to file.
    
    Args:
        file_path: Path to file
        lines: List of lines to write
        encoding: File encoding
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.writelines(lines)


def read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Any:
    """
    Read JSON file.
    
    Args:
        file_path: Path to JSON file
        encoding: File encoding
        
    Returns:
        Parsed JSON data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)


def write_json(file_path: Union[str, Path], data: Any, 
               encoding: str = 'utf-8', indent: int = 2) -> None:
    """
    Write data to JSON file.
    
    Args:
        file_path: Path to JSON file
        data: Data to write
        encoding: File encoding
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file was deleted, False if it didn't exist
    """
    file_path = Path(file_path)
    
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def delete_dir(dir_path: Union[str, Path], recursive: bool = False) -> bool:
    """
    Delete a directory.
    
    Args:
        dir_path: Path to directory
        recursive: Whether to delete recursively
        
    Returns:
        True if directory was deleted, False if it didn't exist
    """
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        return False
    
    if recursive:
        shutil.rmtree(dir_path)
    else:
        dir_path.rmdir()
    
    return True


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy a file.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src = Path(src)
    dst = Path(dst)
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy a directory recursively.
    
    Args:
        src: Source directory path
        dst: Destination directory path
    """
    src = Path(src)
    dst = Path(dst)
    
    shutil.copytree(src, dst, dirs_exist_ok=True)


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Move a file.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src = Path(src)
    dst = Path(dst)
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def dir_exists(dir_path: Union[str, Path]) -> bool:
    """
    Check if directory exists.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        True if directory exists, False otherwise
    """
    return Path(dir_path).is_dir()


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def list_files(dir_path: Union[str, Path], 
               pattern: str = '*',
               recursive: bool = False) -> List[Path]:
    """
    List files in directory.
    
    Args:
        dir_path: Path to directory
        pattern: File pattern (e.g., '*.txt')
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    dir_path = Path(dir_path)
    
    if recursive:
        return list(dir_path.rglob(pattern))
    else:
        return list(dir_path.glob(pattern))


def list_dirs(dir_path: Union[str, Path]) -> List[Path]:
    """
    List subdirectories in directory.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        List of directory paths
    """
    dir_path = Path(dir_path)
    return [p for p in dir_path.iterdir() if p.is_dir()]


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (including dot)
    """
    return Path(file_path).suffix


def get_file_name(file_path: Union[str, Path], with_extension: bool = True) -> str:
    """
    Get file name.
    
    Args:
        file_path: Path to file
        with_extension: Whether to include extension
        
    Returns:
        File name
    """
    file_path = Path(file_path)
    
    if with_extension:
        return file_path.name
    else:
        return file_path.stem


def join_paths(*paths: Union[str, Path]) -> Path:
    """
    Join multiple paths.
    
    Args:
        paths: Paths to join
        
    Returns:
        Joined path
    """
    result = Path(paths[0])
    for path in paths[1:]:
        result = result / path
    return result


def get_absolute_path(file_path: Union[str, Path]) -> Path:
    """
    Get absolute path.
    
    Args:
        file_path: Path to file
        
    Returns:
        Absolute path
    """
    return Path(file_path).absolute()


def get_relative_path(file_path: Union[str, Path], 
                     base_path: Union[str, Path]) -> Path:
    """
    Get relative path.
    
    Args:
        file_path: Path to file
        base_path: Base path
        
    Returns:
        Relative path
    """
    return Path(file_path).relative_to(base_path)


def create_temp_file(suffix: str = '', prefix: str = 'tmp', 
                    dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory for temp file
        
    Returns:
        Path to temporary file
    """
    import tempfile
    
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(fd)
    return Path(path)


def create_temp_dir(suffix: str = '', prefix: str = 'tmp',
                   dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a temporary directory.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        dir: Parent directory for temp dir
        
    Returns:
        Path to temporary directory
    """
    import tempfile
    
    path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    return Path(path)
