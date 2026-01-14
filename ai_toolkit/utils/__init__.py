"""
Utility module for AI Toolkit.

This module provides various utility functions for logging,
file operations, and common tasks.
"""

from .logger import (
    setup_logger,
    get_logger,
    log_info,
    log_error,
    log_warning,
    log_debug,
    log_critical,
    set_log_level,
    disable_logging,
    enable_logging
)

from .file_utils import (
    read_file,
    write_file,
    read_lines,
    write_lines,
    read_json,
    write_json,
    ensure_dir,
    delete_file,
    delete_dir,
    copy_file,
    copy_dir,
    move_file,
    file_exists,
    dir_exists,
    get_file_size,
    list_files,
    list_dirs,
    get_file_extension,
    get_file_name,
    join_paths,
    get_absolute_path,
    get_relative_path,
    create_temp_file,
    create_temp_dir
)

from .utils import (
    format_messages,
    validate_input,
    sanitize_text,
    format_response,
    truncate_text,
    chunk_text,
    merge_dicts,
    flatten_dict,
    unflatten_dict,
    get_timestamp,
    parse_timestamp,
    calculate_hash,
    retry_on_failure,
    batch_process,
    filter_dict,
    safe_get,
    safe_set,
    is_empty,
    coalesce
)

__all__ = [
    # Logger
    'setup_logger',
    'get_logger',
    'log_info',
    'log_error',
    'log_warning',
    'log_debug',
    'log_critical',
    'set_log_level',
    'disable_logging',
    'enable_logging',
    
    # File utils
    'read_file',
    'write_file',
    'read_lines',
    'write_lines',
    'read_json',
    'write_json',
    'ensure_dir',
    'delete_file',
    'delete_dir',
    'copy_file',
    'copy_dir',
    'move_file',
    'file_exists',
    'dir_exists',
    'get_file_size',
    'list_files',
    'list_dirs',
    'get_file_extension',
    'get_file_name',
    'join_paths',
    'get_absolute_path',
    'get_relative_path',
    'create_temp_file',
    'create_temp_dir',
    
    # General utils
    'format_messages',
    'validate_input',
    'sanitize_text',
    'format_response',
    'truncate_text',
    'chunk_text',
    'merge_dicts',
    'flatten_dict',
    'unflatten_dict',
    'get_timestamp',
    'parse_timestamp',
    'calculate_hash',
    'retry_on_failure',
    'batch_process',
    'filter_dict',
    'safe_get',
    'safe_set',
    'is_empty',
    'coalesce',
]
