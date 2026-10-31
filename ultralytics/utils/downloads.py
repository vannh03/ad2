import re
import shutil
import subprocess
import tarfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse
from uuid import uuid4
from ultralytics.utils import ASSETS_URL, LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file
GITHUB_ASSETS_REPO = 'ultralytics/assets'
GITHUB_ASSETS_NAMES = frozenset([f'yolov8{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '-cls', '-seg', '-pose', '-obb', '-oiv7')] + [f'yolo11{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '-cls', '-seg', '-pose', '-obb')] + [f'yolo12{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('',)] + [f'yolo26{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '-cls', '-seg', '-sem', '-pose', '-obb', '-depth')] + [f'yolo26{k}-objv1{suffix}.pt' for k in 'nsmlx' for suffix in ('-150', '-seg')] + [f'yolo26{k}-{suffix}.pt' for k in 'nsmlx' for suffix in ('distill', 'sem-ade20k')] + [f'yolo26{k}-reid.onnx' for k in 'nsmlx'] + [f'yolov5{k}{resolution}u.pt' for k in 'nsmlx' for resolution in ('', '6')] + [f'yolov3{k}u.pt' for k in ('', '-spp', '-tiny')] + [f'yolov8{k}-world.pt' for k in 'smlx'] + [f'yolov8{k}-worldv2.pt' for k in 'smlx'] + [f'yoloe-v8{k}{suffix}.pt' for k in 'sml' for suffix in ('-seg', '-seg-pf')] + [f'yoloe-11{k}{suffix}.pt' for k in 'sml' for suffix in ('-seg', '-seg-pf')] + [f'yoloe-26{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('-seg', '-seg-pf')] + [f'yolov9{k}.pt' for k in 'tsmce'] + [f'yolov9{k}-seg.pt' for k in 'ce'] + [f'yolov10{k}.pt' for k in 'nsmblx'] + [f'yolo_nas_{k}.pt' for k in 'sml'] + [f'sam_{k}.pt' for k in 'bl'] + [f'sam2_{k}.pt' for k in 'blst'] + [f'sam2.1_{k}.pt' for k in 'blst'] + [f'FastSAM-{k}.pt' for k in 'sx'] + [f'rtdetr-{k}.pt' for k in 'lx'] + ['mobile_sam.pt', 'mobileclip_blt.ts', 'mobileclip2_b.ts', 'yolo11n-grayscale.pt', 'yolov8x-pose-p6.pt', 'calibration_image_sample_data_20x128x128x3_float32.npy.zip'])
GITHUB_ASSETS_STEMS = frozenset((k.rpartition('.')[0] for k in GITHUB_ASSETS_NAMES))

def is_url(url, check=False):
    """Validate if the given string is a URL and optionally check if the URL exists online.

    Args:
        url (str | Path): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.

    Returns:
        (bool): True for a valid URL. If 'check' is True, also returns True if the URL exists online.

    Examples:
        >>> valid = is_url("https://www.example.com")
        >>> valid_and_exists = is_url("https://www.example.com", check=True)
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        if not (result.scheme and result.netloc):
            return False
        if check:
            import requests
            return requests.head(url, timeout=3, allow_redirects=True).ok
        return True
    except Exception:
        return False

def delete_dsstore(path, files_to_delete=('.DS_Store', '__MACOSX')):
    """Delete all specified system files and directories in a directory.

    Args:
        path (str | Path): The directory path where the files should be deleted.
        files_to_delete (tuple[str, ...]): Names of files and directories to delete.

    Examples:
        >>> from ultralytics.utils.downloads import delete_dsstore
        >>> delete_dsstore("path/to/dir")

    Notes:
        ".DS_Store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    for file in files_to_delete:
        matches = sorted(Path(path).rglob(file), key=lambda x: len(x.parts), reverse=True)
        LOGGER.info(f'Deleting {file} files: {matches}')
        for f in matches:
            if f.is_dir() and (not f.is_symlink()):
                shutil.rmtree(f)
            else:
                f.unlink()

def zip_directory(directory, compress=True, exclude=('.DS_Store', '__MACOSX'), progress=True):
    """Zip the contents of a directory, excluding specified files.

    The resulting zip file is named after the directory and placed alongside it.

    Args:
        directory (str | Path): The path to the directory to be zipped.
        compress (bool): Whether to compress the files while zipping.
        exclude (tuple[str, ...], optional): A tuple of filename strings to be excluded.
        progress (bool, optional): Whether to display a progress bar.

    Returns:
        (Path): The path to the resulting zip file.

    Examples:
        >>> from ultralytics.utils.downloads import zip_directory
        >>> file = zip_directory("path/to/dir")
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    files = [f for f in directory.rglob('*') if f.is_file() and all((x not in f.name for x in exclude))]
    zip_file = directory.with_suffix('.zip')
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, 'w', compression) as f:
        for file in TQDM(files, desc=f'Zipping {directory} to {zip_file}...', unit='files', disable=not progress):
            f.write(file, file.relative_to(directory))
    return zip_file

def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX'), exist_ok=False, progress=True):
    """Unzip a *.zip file to the specified path, excluding specified files.

    If the zipfile does not contain a single top-level directory, the function will create a new directory with the same
    name as the zipfile (without the extension) to extract its contents. If a path is not provided, the function will
    use the parent directory of the zipfile as the default path.

    Args:
        file (str | Path): The path to the zipfile to be extracted.
        path (str | Path, optional): The path to extract the zipfile to.
        exclude (tuple[str, ...], optional): A tuple of filename strings to be excluded.
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist.
        progress (bool, optional): Whether to display a progress bar.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Examples:
        >>> from ultralytics.utils.downloads import unzip_file
        >>> directory = unzip_file("path/to/file.zip")
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all((x not in f for x in exclude))]
        top_level_dirs = {Path(f).parts[0] for f in files}
        unzip_as_dir = len(top_level_dirs) == 1
        if unzip_as_dir:
            extract_path = path
            path = Path(path) / next(iter(top_level_dirs))
        else:
            path = extract_path = Path(path) / Path(file).stem
        if path.exists() and any(path.iterdir()) and (not exist_ok):
            LOGGER.warning(f'Skipping {file} unzip as destination directory {path} is not empty.')
            return path
        extract_path = Path(extract_path).resolve()
        for f in TQDM(files, desc=f'Unzipping {file} to {Path(path).resolve()}...', unit='files', disable=not progress):
            f_path = Path(f)
            target = (extract_path / f_path).resolve()
            if f_path.is_absolute() or '..' in f_path.parts or target.parts[:len(extract_path.parts)] != extract_path.parts:
                LOGGER.warning(f'Potentially insecure file path: {f}, skipping extraction.')
                continue
            zipObj.extract(f, extract_path)
    return path

def check_disk_space(file_bytes, path=None, sf=1.5, hard=True):
    """Check if there is sufficient disk space to download and store a file.

    Args:
        file_bytes (int): The file size in bytes.
        path (str | Path, optional): The path or drive to check the available free space on.
        sf (float, optional): Safety factor, the multiplier for the required free space.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    (total, _used, free) = shutil.disk_usage(path or Path.cwd())
    if not total or file_bytes * sf < free:
        return True

    def fmt_bytes(b):
        if b < 1 << 20:
            return f'{b / (1 << 10):.1f} KB'
        return f'{b / (1 << 20):.1f} MB' if b < 1 << 30 else f'{b / (1 << 30):.3f} GB'
    text = f'Insufficient free disk space {fmt_bytes(free)} < {fmt_bytes(int(file_bytes * sf))} required, Please free {fmt_bytes(int(file_bytes * sf - free))} additional disk space and try again.'
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False

def get_google_drive_file_info(link):
    """Retrieve the direct download link and filename for a shareable Google Drive file link.

    Args:
        link (str): The shareable link of the Google Drive file.

    Returns:
        url (str): Direct download URL for the Google Drive file.
        filename (str | None): Original filename of the Google Drive file. If filename extraction fails, returns None.

    Examples:
        >>> from ultralytics.utils.downloads import get_google_drive_file_info
        >>> link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        >>> url, filename = get_google_drive_file_info(link)
    """
    import requests
    file_id = link.split('/d/')[1].split('/view', 1)[0]
    drive_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    filename = None
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if 'quota exceeded' in str(response.content.lower()):
            raise ConnectionError(emojis(f'❌  Google Drive file download quota exceeded. Please try again later or download this file manually at {link}.'))
        for (k, v) in response.cookies.items():
            if k.startswith('download_warning'):
                drive_url += f'&confirm={v}'
        cd = response.headers.get('content-disposition')
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return (drive_url, filename)

def safe_download(url, file=None, dir=None, unzip=True, delete=False, curl=False, retry=3, min_bytes=1.0, exist_ok=False, progress=True):
    """Download files from a URL with options for retrying, unzipping, and deleting the downloaded file. Enhanced with
    robust partial download detection using Content-Length validation.

    Args:
        url (str | Path): The URL of the file to be downloaded.
        file (str | Path, optional): The filename of the downloaded file. If not provided, the file will be saved with
            the same name as the URL.
        dir (str | Path, optional): The directory to save the downloaded file. If not provided, the file will be saved
            in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file.
        delete (bool, optional): Whether to delete the downloaded file after unzipping.
        curl (bool, optional): Whether to use curl command line tool for downloading.
        retry (int, optional): The number of times to retry the download in case of failure.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping.
        progress (bool, optional): Whether to display a progress bar during the download.

    Returns:
        (Path | str): The path to the downloaded file or extracted directory.

    Examples:
        >>> from ultralytics.utils.downloads import safe_download
        >>> link = "https://ultralytics.com/assets/bus.jpg"
        >>> path = safe_download(link)
    """
    url = str(url)
    if '://' not in url and Path(url).is_file():
        f = Path(url)
    else:
        import requests
        gdrive = url.startswith('https://drive.google.com/')
        if gdrive:
            (url, file) = get_google_drive_file_info(url)
        url = url.replace(' ', '%20')
        f = Path(dir or '.') / (file or url2file(url))
        if not f.is_file():
            uri = (url if gdrive else clean_url(url)).replace(ASSETS_URL, 'https://ultralytics.com/assets')
            desc = f"Downloading {uri} to '{f}'"
            f.parent.mkdir(parents=True, exist_ok=True)
            target = f
            f = target.with_name(f'.{target.name}.{uuid4().hex}.part')
            curl_installed = shutil.which('curl')
            expected_size = None
            for i in range(retry + 1):
                try:
                    if (curl or i > 0) and curl_installed:
                        s = 'sS' * (not progress)
                        args = ['--connect-timeout', '30', '--speed-limit', '1', '--speed-time', '300']
                        r = subprocess.run(['curl', '-#', f'-{s}fL', url, '-o', f, '--retry', '3', '-C', '-', *args], check=False).returncode
                        assert r == 0, f'Curl return value {r}'
                    else:
                        with requests.get(url, stream=True, headers={'Accept-Encoding': 'identity'}, timeout=(30, 300)) as response:
                            response.raise_for_status()
                            expected_size = int(response.headers.get('Content-Length', 0))
                            if i == 0 and expected_size > 1048576:
                                check_disk_space(expected_size, path=f.parent)
                            buffer_size = max(8192, min(1048576, expected_size // 1000)) if expected_size else 8192
                            with TQDM(total=expected_size, desc=desc, disable=not progress, unit='B', unit_scale=True, unit_divisor=1024) as pbar, open(f, 'wb') as f_opened:
                                for data in response.iter_content(chunk_size=buffer_size):
                                    f_opened.write(data)
                                    pbar.update(len(data))
                    if f.exists():
                        file_size = f.stat().st_size
                        if file_size > min_bytes:
                            if expected_size and file_size != expected_size:
                                LOGGER.warning(f'Partial download: {file_size}/{expected_size} bytes ({file_size / expected_size * 100:.1f}%)')
                            else:
                                f.replace(target)
                                f = target
                                break
                        f.unlink()
                except MemoryError:
                    raise
                except Exception as e:
                    if i == 0 and (not is_online()):
                        f.unlink(missing_ok=True)
                        raise ConnectionError(emojis(f'❌  Download failure for {uri}. Environment may be offline.')) from e
                    elif i >= retry:
                        f.unlink(missing_ok=True)
                        raise ConnectionError(emojis(f'❌  Download failure for {uri}. Retry limit reached. {e}')) from e
                    LOGGER.warning(f'Download failure, retrying {i + 1}/{retry} {uri}... {e}')
            else:
                raise ConnectionError(emojis(f'❌  Download failure for {uri}. Retry limit reached.'))
    if unzip and f.exists() and (f.suffix in {'', '.zip', '.tar', '.gz'}):
        from zipfile import is_zipfile
        unzip_dir = (dir or f.parent).resolve()
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)
        elif f.suffix in {'.tar', '.gz'}:
            LOGGER.info(f'Unzipping {f} to {unzip_dir}...')
            with tarfile.open(f, 'r:*') as tar:
                for m in tar:
                    if not (m.isfile() or m.isdir()) or m.issym() or m.islnk():
                        LOGGER.warning(f'Potentially insecure tar member: {m.name}, skipping extraction.')
                        continue
                    m_path = Path(m.name)
                    target = (unzip_dir / m_path).resolve()
                    if m_path.is_absolute() or '..' in m_path.parts or target.parts[:len(unzip_dir.parts)] != unzip_dir.parts:
                        LOGGER.warning(f'Potentially insecure file path: {m.name}, skipping extraction.')
                        continue
                    if m.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        source = tar.extractfile(m)
                        if source:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with source, open(target, 'wb') as out:
                                shutil.copyfileobj(source, out)
        if delete:
            f.unlink()
        return unzip_dir
    return f

def get_github_assets(repo='ultralytics/assets', version='latest', retry=False):
    """Retrieve the specified version's tag and assets from a GitHub repository.

    If the version is not specified, the function fetches the latest release assets.

    Args:
        repo (str, optional): The GitHub repository in the format 'owner/repo'.
        version (str, optional): The release version to fetch assets from.
        retry (bool, optional): Flag to retry the request in case of a failure.

    Returns:
        tag (str): The release tag.
        assets (list[str]): A list of asset names.

    Examples:
        >>> tag, assets = get_github_assets(repo="ultralytics/assets", version="latest")
    """
    import requests
    if version != 'latest':
        version = f'tags/{version}'
    url = f'https://api.github.com/repos/{repo}/releases/{version}'
    attempts = 2 if retry else 1
    for attempt in range(attempts):
        try:
            r = requests.get(url, timeout=30)
        except requests.exceptions.RequestException as e:
            if attempt < attempts - 1:
                continue
            LOGGER.warning(f'GitHub assets check failure for {url}: {e}')
            return ('', [])
        if r.status_code == 200 or r.reason == 'rate limit exceeded':
            break
    if r.status_code != 200:
        LOGGER.warning(f'GitHub assets check failure for {url}: {r.status_code} {r.reason}')
        return ('', [])
    data = r.json()
    return (data['tag_name'], [x['name'] for x in data['assets']])

def attempt_download_asset(file, repo='ultralytics/assets', release='v8.4.0', **kwargs):
    """Attempt to download a file from GitHub release assets if it is not found locally.

    Args:
        file (str | Path): The filename or file path to be downloaded.
        repo (str, optional): The GitHub repository in the format 'owner/repo'.
        release (str, optional): The specific release version to be downloaded.
        **kwargs (Any): Additional keyword arguments for the download process.

    Returns:
        (str): The path to the downloaded file.

    Examples:
        >>> file_path = attempt_download_asset("yolo26n.pt", repo="ultralytics/assets", release="latest")
    """
    from ultralytics.utils import SETTINGS
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        name = Path(parse.unquote(str(file))).name
        download_url = f'https://github.com/{repo}/releases/download'
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = url2file(name)
            if Path(file).is_file():
                LOGGER.info(f'Found {clean_url(url)} locally at {file}')
            else:
                safe_download(url=url, file=file, min_bytes=100000.0, **kwargs)
        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f'{download_url}/{release}/{name}', file=file, min_bytes=100000.0, **kwargs)
        else:
            (tag, assets) = get_github_assets(repo, release)
            if not assets:
                (tag, assets) = get_github_assets(repo)
            if name in assets:
                safe_download(url=f'{download_url}/{tag}/{name}', file=file, min_bytes=100000.0, **kwargs)
        return str(file)

def download(url, dir=None, unzip=True, delete=False, curl=False, threads=1, retry=3, exist_ok=False):
    """Download files from specified URLs to a given directory.

    Supports concurrent downloads if multiple threads are specified.

    Args:
        url (str | list[str] | Path): The URL or list of URLs of the files to be downloaded.
        dir (Path, optional): The directory where the files will be saved.
        unzip (bool, optional): Flag to unzip the files after downloading.
        delete (bool, optional): Flag to delete the zip files after extraction.
        curl (bool, optional): Flag to use curl for downloading.
        threads (int, optional): Number of threads to use for concurrent downloads.
        retry (int, optional): Number of retries in case of download failure.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping.

    Examples:
        >>> download("https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg", dir="path/to/dir")
    """
    dir = Path(dir or Path.cwd())
    dir.mkdir(parents=True, exist_ok=True)
    urls = [url] if isinstance(url, (str, Path)) else url
    if threads > 1:
        LOGGER.info(f'Downloading {len(urls)} file(s) with {threads} threads to {dir}...')
        with ThreadPool(threads) as pool:
            pool.map(lambda x: safe_download(url=x[0], dir=x[1], unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok, progress=True), zip(urls, repeat(dir)))
            pool.close()
            pool.join()
    else:
        for u in urls:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)