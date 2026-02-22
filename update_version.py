#!/usr/bin/env python3
"""
Manual version update script for UoM FSE Deep Learning Workshop.

This script updates the version information in utils/__version__.py
based on the current Git tags and commit.
"""

import os
import subprocess
import re

def get_git_version():
    # Try to get the latest tag
    try:
        tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # Remove 'v' prefix if present
        version = tag.lstrip('v')
    except subprocess.CalledProcessError:
        # No tags found, use default version
        version = '0.9.5'
    
    # Get the current commit hash
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        commit = 'unknown'
    
    return version, commit

def update_version_file(version, commit):
    # Path to version file
    version_file = os.path.join('utils', '__version__.py')
    
    # Create content for the version file
    content = f'''"""
Version information for the UoM Deep Learning Workshop utilities package.

This module contains version information which is used throughout the package
for version checking, logging, and compatibility information.
"""

import os
import subprocess
from typing import Tuple, Optional

def get_version_info() -> Tuple[str, Optional[str]]:
    """
    Get version information from Git.
    
    Returns:
        Tuple containing (version_string, git_commit_hash)
        If not in a git repository, commit hash will be None
    """
    # Default version if not in a Git repository
    version = '{version}'
    commit_hash = '{commit}'
    
    try:
        # Try to get the current tag
        tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        if tag:
            version = tag.lstrip('v')  # Remove 'v' prefix if present
        
        # Get the current commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('utf-8').strip()
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repository or git not installed
        pass
    
    return version, commit_hash

# Get version information
__version__, __commit__ = get_version_info()

# Full version string including commit hash if available
__version_full__ = __version__
if __commit__:
    __version_full__ = f"{{__version__}}+{{__commit__}}"
'''
    
    # Write to the file
    with open(version_file, 'w') as f:
        f.write(content)
    
    print(f"Version updated: {version}+{commit}")

def update_readme_badge(version):
    """Update the version badge in README.md to reflect the current version."""
    readme_path = 'README.md'

    with open(readme_path, 'r') as f:
        content = f.read()

    updated = re.sub(
        r'(\[!\[Version\]\(https://img\.shields\.io/badge/version-)[\d.]+(-blue)',
        rf'\g<1>{version}\2',
        content
    )

    if updated == content:
        print("README.md badge already up to date.")
        return

    with open(readme_path, 'w') as f:
        f.write(updated)

    print(f"README.md version badge updated to {version}")


def main():
    version, commit = get_git_version()
    update_version_file(version, commit)
    update_readme_badge(version)

if __name__ == "__main__":
    main()