from pathlib import Path
import requests
import tarfile
import zipfile
import json
from tqdm import tqdm
from utils.core import find_project_root
from urllib.parse import urlparse, unquote

__all__ = ['download_dataset', 'extract_files']

def download_file_in_chunks(url, f_path, total_size, chunk_size=1024*1024*100):
    """
    Download a file in chunks using HTTP Range requests.
    Args:
        url (str): File URL
        f_path (Path): Destination file path
        total_size (int): Total file size in bytes
        chunk_size (int): Size of each chunk in bytes (default: 100MB)
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    with open(f_path, 'wb') as f:
        for start in tqdm(range(0, total_size, chunk_size), desc=f"Chunked Download {f_path.name}", unit='B', unit_scale=True):
            end = min(start + chunk_size - 1, total_size - 1)
            range_header = {'Range': f'bytes={start}-{end}', **headers}
            r = requests.get(url, headers=range_header, stream=True)
            r.raise_for_status()
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)


def download_dataset(dataset_name: str, dest_path: str = None, extract: bool = False, 
                    remove_compressed: bool = False) -> Path:
    """
    Download a dataset from a predefined repository and optionally extract it.
    
    This function loads dataset information from datasets.json, retrieves
    the dataset file from its URL, and optionally extracts its contents.
    It shows a progress bar during download and handles common error cases.
    
    Args:
        dataset_name (str): Name of the dataset as specified in datasets.json
        dest_path (str, optional): Path where the dataset will be saved.
                                  If None, uses current working directory.
        extract (bool, optional): Whether to extract compressed files after download.
                                 Default is False.
        remove_compressed (bool, optional): Whether to remove the compressed archive
                                          after extraction. Default is False.
    
    Returns:
        Path: Path to the downloaded dataset file or extracted directory
        
    Raises:
        ValueError: If the requested dataset is not found in datasets.json
        Exception: If download or extraction fails
    """
    # Load the dataset metadata from the JSON configuration file
    project_root = find_project_root()
    json_path = project_root / 'utils/data/datasets.json'
    
    # Check if the datasets configuration file exists
    if not json_path.exists():
        print('ERROR: datasets.json file not found')
        return None
   
    # Load dataset definitions from JSON file 
    with open(json_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)
        
    # Validate that the requested dataset exists
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.json")
    
    # Extract dataset metadata
    dataset_info = datasets[dataset_name]
    url = dataset_info['url']
    desc = dataset_info['description']
    authors = ", ".join(dataset_info['authors'])
    year = dataset_info['year']
    website = dataset_info['website']

    # Display dataset information
    print(f'Downloading:\n{desc}')
    print(f'> Authors: {authors}')
    print(f'> Year: {year}')
    print(f'> Website: {website}\n')

    # Set up destination path
    dest_path = Path(dest_path) if dest_path else Path.cwd()
    parsed_url = urlparse(url)
    filename = Path(unquote(parsed_url.path)).name  # get the file name without query params
    f_path = dest_path / filename

    # Check if the file already exists to avoid re-downloading
    if f_path.exists():
        print('File already exists')
        if extract:
            extract_path = dest_path / Path(url).stem
            extract_files(f_path, extract_path, recursive=True,
                          remove_compressed=remove_compressed)
            return extract_path
        return f_path
    else:
        # Create the destination directory if it doesn't exist
        dest_path.mkdir(parents=True, exist_ok=True)

    # Download the file with progress tracking
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Get file size first
        head_resp = requests.head(url, headers=headers, allow_redirects=True)
        total_size = int(head_resp.headers.get('content-length', 0))
        if total_size >= 4 * 1024 * 1024 * 1024:  # 4GB
            print("Large file detected, using chunked download...")
            download_file_in_chunks(url, f_path, total_size)
        else:
            response = requests.get(url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()
            chunk_size = 1024
            pbar = tqdm(total=total_size, unit='iB', unit_scale=True,
                        desc=f'Downloading {filename}', dynamic_ncols=True)
            with open(f_path, 'wb') as file:
                for data in response.iter_content(chunk_size):
                    pbar.update(len(data))
                    file.write(data)
            pbar.close()
            if total_size != 0 and pbar.n != total_size:
                print('ERROR: Download incomplete. File size mismatch.')
                return None
            
    except requests.exceptions.RequestException as e:
        print(f'ERROR: Download failed - {e}')
        if f_path.exists():
            f_path.unlink()
        return None
        
    # Extract the downloaded file if requested
    if extract:
        try:
            extract_path = dest_path / Path(url).stem
            extracted_path = extract_files(f_path, extract_path, recursive=True,
                         remove_compressed=remove_compressed)
            f_path = extracted_path
        except Exception as e:
            print(f'ERROR: Extraction failed - {e}')

    return f_path


def extract_files(f_path: str, dest_path: str, recursive: bool = False, 
                 remove_compressed: bool = False) -> Path:
    """
    Extract files from a compressed archive (zip, tar, tar.gz, tgz).
    
    This function handles multiple compression formats and can recursively
    extract nested archives. It shows a progress bar during extraction.
    It prevents duplicate folder structures by detecting if the archive contains
    a single root folder with the same name as the archive.
    
    Args:
        f_path (str or Path): Path to the compressed file
        dest_path (str or Path): Path where contents will be extracted
        recursive (bool, optional): Whether to extract archives within the extracted
                                   folder. Default is False.
        remove_compressed (bool, optional): Whether to remove the original compressed
                                          file after extraction. Default is False.
    
    Returns:
        Path: Path to the extracted directory
        
    Raises:
        FileNotFoundError: If the compressed file does not exist
        Exception: If extraction fails
    """
    # Convert paths to Path objects for consistent handling
    f_path = Path(f_path)
    dest_path = Path(dest_path)

    # Validate input paths
    if not f_path.exists():
        print(f'ERROR: File {f_path} does not exist')
        return None

    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary extraction directory to prevent duplicate folders
    temp_extract_dir = dest_path / f"_temp_{f_path.stem}"
    temp_extract_dir.mkdir(exist_ok=True)
    
    try:
        # Handle different archive types
        if f_path.suffix == '.tar':
            f = tarfile.open(f_path, 'r')
            members = f.getmembers()
        elif f_path.suffix == '.gz' and f_path.stem.endswith('.tar'):
            f = tarfile.open(f_path, 'r:gz')
            members = f.getmembers()
        elif f_path.suffix == '.tgz':
            f = tarfile.open(f_path, 'r:gz')
            members = f.getmembers()
        elif f_path.suffix == '.zip':
            f = zipfile.ZipFile(f_path, 'r')
            members = f.namelist()
        else:
            print(f'ERROR: Unsupported file format: {f_path.suffix}')
            return None

        # Extract files with a progress bar
        with tqdm(members, desc=f'Extracting {f_path.name}', dynamic_ncols=True) as pbar:
            for member in pbar:
                f.extract(member, temp_extract_dir)
                pbar.update(0)  # Update is done automatically by tqdm iteration

        f.close()

        # Remove the original compressed file if requested
        if remove_compressed and f_path.exists():
            f_path.unlink()
            print(f"Removed compressed file: {f_path}")
        
        # Check if there's a single root directory with the same name as the archive
        items = list(temp_extract_dir.iterdir())
        final_extract_path = dest_path
        
        # If there's a single item and it's a directory, we'll use its contents
        if len(items) == 1 and items[0].is_dir():
            root_dir = items[0]
            # Move contents from temp_dir/single_dir/* to dest_path/
            for item in root_dir.iterdir():
                target_path = dest_path / item.name
                if target_path.exists():
                    if target_path.is_dir():
                        # Merge directories
                        for sub_item in item.iterdir():
                            sub_target = target_path / sub_item.name
                            if not sub_target.exists():
                                sub_item.rename(sub_target)
                    else:
                        # Skip if file exists
                        pass
                else:
                    # Move file or directory directly
                    item.rename(target_path)
        else:
            # Move all contents from temp_dir/* to dest_path/
            for item in items:
                target_path = dest_path / item.name
                if not target_path.exists():
                    item.rename(target_path)
        
        # Clean up the temporary directory
        if temp_extract_dir.exists():
            import shutil
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

        # Recursively extract any archives in the extracted folder
        if recursive:
            for file in dest_path.rglob('*.*'):
                # Skip the original archive and non-archives
                if file == f_path or not file.is_file():
                    continue
                    
                # Check if this is a supported archive type
                is_archive = False
                if file.suffix == '.tar' or file.suffix == '.zip':
                    is_archive = True
                elif file.suffix == '.gz' and file.stem.endswith('.tar'):
                    is_archive = True
                elif file.suffix == '.tgz':
                    is_archive = True
                    
                if is_archive:
                    print(f"Found nested archive: {file.name}")
                    extract_files(file, file.parent, recursive=True,
                                remove_compressed=remove_compressed)
        
        return dest_path
        
    except Exception as e:
        print(f'ERROR: Extraction failed - {e}')
        import traceback
        traceback.print_exc()
        return None
