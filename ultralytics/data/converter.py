import asyncio
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
try:
    from filelock import AsyncFileLock, Timeout
except Exception:
    AsyncFileLock = None
    Timeout = Exception
from PIL import Image
from ultralytics.data.utils import get_split_fraction
from ultralytics.utils import ASSETS_URL, DATASETS_DIR, LOGGER, NUM_THREADS, TQDM, YAML, clean_url
from ultralytics.utils.checks import check_file
from ultralytics.utils.downloads import download, zip_directory
from ultralytics.utils.files import increment_path

def coco91_to_coco80_class():
    """Convert 91-index COCO class IDs to 80-index COCO class IDs.

    Returns:
        (list[int | None]): A list of 91 elements where the index represents the 91-index class ID and the value is the
            corresponding 80-index class ID, or None if there is no mapping.
    """
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None, None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, None, 73, 74, 75, 76, 77, 78, 79, None]

def coco80_to_coco91_class():
    """Convert 80-index (val2014) to 91-index (paper).

    Returns:
        (list[int]): A list of 80 class IDs where each value is the corresponding 91-index class ID.

    Examples:
        >>> import numpy as np
        >>> a = np.loadtxt("data/coco.names", dtype="str", delimiter="\\n")
        >>> b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\\n")

        Convert the darknet to COCO format
        >>> x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]

        Convert the COCO to darknet format
        >>> x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]

    References:
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    """
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def convert_coco(labels_dir='../coco/annotations/', save_dir='coco_converted/', use_segments=False, use_keypoints=False, cls91to80=True, lvis=False):
    """Convert COCO dataset annotations to a YOLO annotation format suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
        lvis (bool, optional): Whether to convert data in lvis dataset way.

    Examples:
        >>> from ultralytics.data.converter import convert_coco

        Convert COCO annotations to YOLO format
        >>> convert_coco("coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=False)

        Convert LVIS annotations to YOLO format
        >>> convert_coco("lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
    """
    save_dir = increment_path(save_dir)
    for p in (save_dir / 'labels', save_dir / 'images'):
        p.mkdir(parents=True, exist_ok=True)
    coco80 = coco91_to_coco80_class()
    for json_file in sorted(Path(labels_dir).resolve().glob('*.json')):
        lname = '' if lvis else json_file.stem.replace('instances_', '')
        fn = Path(save_dir) / 'labels' / lname
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            (fn / 'train2017').mkdir(parents=True, exist_ok=True)
            (fn / 'val2017').mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        images = {f"{x['id']:d}": x for x in data['images']}
        annotations = defaultdict(list)
        for ann in data['annotations']:
            annotations[ann['image_id']].append(ann)
        image_txt = []
        dropped = False
        for (img_id, anns) in TQDM(annotations.items(), desc=f'Annotations {json_file}'):
            img = images[f'{img_id:d}']
            (h, w) = (img['height'], img['width'])
            f = str(Path(img['coco_url']).relative_to('http://images.cocodataset.org')) if lvis else img['file_name']
            if lvis:
                image_txt.append(str(Path('./images') / f))
            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get('iscrowd', False):
                    continue
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    if use_keypoints:
                        if ann.get('keypoints') is None:
                            continue
                        keypoints.append(box + (np.array(ann['keypoints']).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist())
                    bboxes.append(box)
                    if use_segments:
                        seg = ann.get('segmentation')
                        polygons = [p for p in seg or [] if isinstance(p, list) and len(p) >= 6 and (not len(p) % 2) and all((isinstance(c, (int, float)) for c in p))] if isinstance(seg, list) else []
                        if not polygons:
                            dropped = True
                            (cx, cy, bw, bh) = box[1:]
                            (x1, y1, x2, y2) = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                            segments.append([cls, x1, y1, x2, y1, x2, y2, x1, y2])
                        elif len(polygons) > 1:
                            s = merge_multi_segment(polygons)
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                            segments.append([cls, *s])
                        else:
                            s = [j for i in polygons for j in i]
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                            segments.append([cls, *s])
            with open((fn / f).with_suffix('.txt'), 'a', encoding='utf-8') as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*keypoints[i],)
                    else:
                        line = (*(segments[i] if use_segments else bboxes[i]),)
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')
        if dropped and (not use_keypoints):
            LOGGER.warning(f'{json_file}: annotations without a usable polygon, because the segmentation is missing, empty, or not a point list such as an RLE mask, use a segment shaped like their bounding box.')
        if lvis:
            filename = Path(save_dir) / json_file.name.replace('lvis_v1_', '').replace('.json', '.txt')
            with open(filename, 'a', encoding='utf-8') as f:
                f.writelines((f'{line}\n' for line in image_txt))
    LOGGER.info(f"{('LVIS' if lvis else 'COCO')} data converted successfully.\nResults saved to {save_dir.resolve()}")

def convert_segment_masks_to_yolo_seg(masks_dir, output_dir, classes):
    """Convert a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation
    format. The converted masks are saved in the specified output directory.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        classes (int): Total number of classes in the dataset, e.g., 80 for COCO.

    Examples:
        >>> from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
        >>> convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)

    Notes:
        The expected directory structure for the masks is:

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in sorted(Path(masks_dir).iterdir()):
        if mask_path.suffix in {'.png', '.jpg'}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            (img_height, img_width) = mask.shape
            LOGGER.info(f'Processing {mask_path} imgsz = {img_height} x {img_width}')
            unique_values = np.unique(mask)
            yolo_format_data = []
            for value in unique_values:
                if value == 0:
                    continue
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(f'Unknown class for pixel value {value} in file {mask_path}, skipping.')
                    continue
                (contours, _) = cv2.findContours((mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        contour = contour.squeeze()
                        yolo_format = [class_index]
                        for point in contour:
                            yolo_format.append(round(point[0] / img_width, 6))
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            output_path = output_dir / f'{mask_path.stem}.txt'
            with open(output_path, 'w', encoding='utf-8') as file:
                for item in yolo_format_data:
                    line = ' '.join(map(str, item))
                    file.write(line + '\n')
            LOGGER.info(f'Processed and stored at {output_path} imgsz = {img_height} x {img_width}')

def convert_dota_to_yolo_obb(dota_root_path):
    """Convert DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Examples:
        >>> from ultralytics.data.converter import convert_dota_to_yolo_obb
        >>> convert_dota_to_yolo_obb("path/to/DOTA")

    Notes:
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)
    class_mapping = {'plane': 0, 'ship': 1, 'storage-tank': 2, 'baseball-diamond': 3, 'tennis-court': 4, 'basketball-court': 5, 'ground-track-field': 6, 'harbor': 7, 'bridge': 8, 'large-vehicle': 9, 'small-vehicle': 10, 'helicopter': 11, 'roundabout': 12, 'soccer-ball-field': 13, 'swimming-pool': 14, 'container-crane': 15, 'airport': 16, 'helipad': 17}

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Convert a single image's DOTA annotation to YOLO OBB format and save it to a specified directory."""
        orig_label_path = orig_label_dir / f'{image_name}.txt'
        save_path = save_dir / f'{image_name}.txt'
        with orig_label_path.open('r') as f, save_path.open('w') as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)]
                formatted_coords = [f'{coord:.6g}' for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")
    for phase in ('train', 'val'):
        image_dir = dota_root_path / 'images' / phase
        orig_label_dir = dota_root_path / 'labels' / f'{phase}_original'
        save_dir = dota_root_path / 'labels' / phase
        save_dir.mkdir(parents=True, exist_ok=True)
        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f'Processing {phase} images'):
            if image_path.suffix != '.png':
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            (h, w) = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)

def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple[int, int]): A tuple (idx1, idx2) where idx1 is the index in arr1 and idx2 is the index in arr2 of the
            pair with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    """Merge multiple segments into one list by connecting the coordinates with the minimum distance between each
    segment.

    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file. Each element is a list of coordinates, like
            [segmentation1, segmentation2,...].

    Returns:
        (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]
    for i in range(1, len(segments)):
        (idx1, idx2) = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)
    for k in range(2):
        if k == 0:
            for (i, idx) in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def yolo_bbox2segment(im_dir, save_dir=None, sam_model='sam_b.pt', device=None):
    """Convert existing object detection dataset (bounding boxes) to segmentation dataset in YOLO format.

    Generates segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path, optional): Path to save the generated labels, labels will be saved into `labels-segment`
            in the same directory level of `im_dir` if save_dir is None.
        sam_model (str): Segmentation model to use for intermediate segmentation data.
        device (int | str, optional): The specific device to run SAM models.

    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ...
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ...
                └─ NNN.txt
    """
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils.ops import xywh2xyxy
    dataset = YOLODataset(im_dir, data={'names': list(range(1000)), 'channels': 3})
    if len(dataset.labels[0]['segments']) > 0:
        LOGGER.info('Segmentation labels detected, no need to generate new ones!')
        return
    LOGGER.info('Detection labels detected, generating segment labels by SAM model!')
    sam_model = SAM(sam_model)
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc='Generating segment labels'):
        (h, w) = label['shape']
        boxes = label['bboxes']
        if len(boxes) == 0:
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label['im_file'])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        label['segments'] = sam_results[0].masks.xyn
    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / 'labels-segment'
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label['im_file']).with_suffix('.txt').name
        txt_file = save_dir / lb_name
        cls = label['cls']
        for (i, s) in enumerate(label['segments']):
            if len(s) < 3:
                continue
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(('%g ' * len(line)).rstrip() % line)
        with open(txt_file, 'a', encoding='utf-8') as f:
            f.writelines((text + '\n' for text in texts))
    LOGGER.info(f'Generated segment labels saved in {save_dir}')

def create_synthetic_coco_dataset():
    """Create a synthetic COCO dataset with random images based on filenames from label lists.

    This function downloads COCO labels, reads image filenames from label list files, creates synthetic images for
    train2017 and val2017 subsets, and organizes them in the COCO dataset structure. It uses multithreading to generate
    images efficiently.

    Examples:
        >>> from ultralytics.data.converter import create_synthetic_coco_dataset
        >>> create_synthetic_coco_dataset()

    Notes:
        - Requires internet connection to download label files.
        - Generates random RGB images of varying sizes (480x480 to 640x640 pixels).
        - Existing test2017 directory is removed as it's not needed.
        - Reads image filenames from train2017.txt and val2017.txt files.
    """

    def create_synthetic_image(image_file):
        """Generate a synthetic image with random size and color for dataset augmentation or testing purposes."""
        if not image_file.exists():
            size = (random.randint(480, 640), random.randint(480, 640))
            Image.new('RGB', size=size, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))).save(image_file)
    dir = DATASETS_DIR / 'coco'
    download([f'{ASSETS_URL}/coco2017labels-segments.zip'], dir=dir.parent)
    shutil.rmtree(dir / 'labels' / 'test2017', ignore_errors=True)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for subset in ('train2017', 'val2017'):
            subset_dir = dir / 'images' / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            label_list_file = dir / f'{subset}.txt'
            if label_list_file.exists():
                with open(label_list_file, encoding='utf-8') as f:
                    image_files = [dir / line.strip() for line in f]
                futures = [executor.submit(create_synthetic_image, image_file) for image_file in image_files]
                for _ in TQDM(as_completed(futures), total=len(futures), desc=f'Generating images for {subset}'):
                    pass
            else:
                LOGGER.warning(f'Labels file {label_list_file} does not exist. Skipping image creation for {subset}.')
    LOGGER.info('Synthetic COCO dataset created successfully.')

def convert_to_multispectral(path, n_channels=10, replace=False, zip=False):
    """Convert RGB images to multispectral images by interpolating across wavelength bands.

    This function takes RGB images and interpolates them to create multispectral images with a specified number of
    channels. It can process either a single image or a directory of images.

    Args:
        path (str | Path): Path to an image file or directory containing images to convert.
        n_channels (int): Number of spectral channels to generate in the output image.
        replace (bool): Whether to replace the original image file with the converted one.
        zip (bool): Whether to zip the converted images into a zip file.

    Examples:
        Convert a single image
        >>> convert_to_multispectral("path/to/image.jpg", n_channels=10)

        Convert a dataset
        >>> convert_to_multispectral("coco8", n_channels=10)
    """
    from ultralytics.data.utils import IMG_FORMATS
    path = Path(path)
    if path.is_dir():
        im_files = [f for ext in IMG_FORMATS - {'tif', 'tiff'} for f in path.rglob(f'*.{ext}')]
        for im_path in im_files:
            try:
                convert_to_multispectral(im_path, n_channels)
                if replace:
                    im_path.unlink()
            except Exception as e:
                LOGGER.info(f'Error converting {im_path}: {e}')
        if zip:
            zip_directory(path)
    else:
        output_path = path.with_suffix('.tiff')
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        rgb_wavelengths = np.array([650, 510, 475])
        target_wavelengths = np.linspace(450, 700, n_channels)
        order = np.argsort(rgb_wavelengths)
        xp = rgb_wavelengths[order]
        seg = np.clip(np.searchsorted(xp, target_wavelengths) - 1, 0, len(xp) - 2)
        w = (target_wavelengths - xp[seg]) / (xp[seg + 1] - xp[seg])
        img = img[..., order]
        multispectral = img[..., seg] * (1 - w) + img[..., seg + 1] * w
        cv2.imwritemulti(str(output_path), np.clip(multispectral, 0, 255).astype(np.uint8).transpose(2, 0, 1))
        LOGGER.info(f'Converted {output_path}')

def _infer_ndjson_kpt_shape(image_records):
    """Infer kpt_shape [num_keypoints, dims] from NDJSON pose annotations.

    Scans up to 50 pose annotations across image records. Annotation format is [classId, cx, cy, w, h, kp1_x, kp1_y,
    kp1_vis, ...] so keypoint values start at index 5.

    Tries dims=3 first (x, y, visibility) with visibility validation ({0, 1, 2}), then falls back to dims=2 (x, y only)
    when values are unambiguously not divisible by 3.
    """
    kpt_lengths = []
    samples = []
    for record in image_records:
        for ann in record.get('annotations', {}).get('pose', []):
            kpt_len = len(ann) - 5
            if kpt_len > 0:
                kpt_lengths.append(kpt_len)
                samples.append(ann[5:])
            if len(kpt_lengths) >= 50:
                break
        if len(kpt_lengths) >= 50:
            break
    if not kpt_lengths or len(set(kpt_lengths)) != 1:
        raise ValueError("Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose")
    n = kpt_lengths[0]
    if n % 3 == 0 and all((v in (0, 1, 2) for s in samples for v in s[2::3])):
        return [n // 3, 3]
    if n % 2 == 0 and n % 3 != 0:
        return [n // 2, 2]
    raise ValueError("Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose")

async def convert_ndjson_to_yolo(ndjson_path, output_path=None, fraction=1.0):
    """Convert NDJSON dataset format to Ultralytics YOLO dataset structure.

    This function converts datasets stored in NDJSON (Newline Delimited JSON) format to the standard YOLO format. For
    detection/segmentation/pose/obb tasks, it creates separate directories for images and labels. Depth datasets use
    parallel images/ and depth/ trees with scaled uint16 PNG targets. Classification tasks use the ImageNet-style
    {split}/{class_name}/ folder structure. Downloads run concurrently.

    The NDJSON format consists of:
    - First line: Dataset metadata with class names, task type, and configuration
    - Subsequent lines: Individual image records with annotations and optional URLs

    Args:
        ndjson_path (str | Path): Path to the input NDJSON file containing dataset information.
        output_path (str | Path | None, optional): Directory where the converted YOLO dataset will be saved. If None,
            uses the DATASETS_DIR directory. Defaults to None.
        fraction (float | int | list): Train ratio/count or [train, val, test] ratios/counts to download.

    Returns:
        (Path): Path to the generated data.yaml file (detection) or dataset directory (classification).

    Examples:
        Convert a local NDJSON file:
        >>> yaml_path = await convert_ndjson_to_yolo("dataset.ndjson")
        >>> print(f"Dataset converted to: {yaml_path}")

        Convert with custom output directory:
        >>> yaml_path = await convert_ndjson_to_yolo("dataset.ndjson", output_path="./converted_datasets")

        Use with YOLO training
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-ndjson.ndjson")
    """
    source = str(ndjson_path)
    output_path = Path(output_path or DATASETS_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    if isinstance(fraction, list):
        fraction = [get_split_fraction(fraction, split) for split in ('train', 'val', 'test')[:len(fraction)]]
    else:
        fraction = get_split_fraction(fraction, 'train')
    local = Path(source).is_file()
    source_id = str(Path(source).resolve()) if local else clean_url(source)
    source_hash = hashlib.sha256(repr((source_id, fraction)).encode()).hexdigest()[:8]
    cache_path = output_path / f'.{Path(source_id).stem}-{source_hash}.cache'

    async def convert():
        cache_path.unlink(missing_ok=True)
        result = await _convert_ndjson_to_yolo(Path(check_file(source)), output_path, local, fraction)
        cache_path.write_text(str(result.relative_to(output_path)))
        return result
    try:
        async with AsyncFileLock(cache_path.with_suffix('.lock'), timeout=0):
            return await convert()
    except Timeout:
        pass
    async with AsyncFileLock(cache_path.with_suffix('.lock')):
        if cache_path.is_file():
            result = output_path / cache_path.read_text()
            marker = result / '.ndjson.yaml' if result.is_dir() else result
            if marker.is_file():
                return result
        return await convert()

async def _convert_ndjson_to_yolo(ndjson_path, output_path, local, fraction):
    """Convert a resolved NDJSON source while its conversion lock is held."""
    from ultralytics.utils.checks import check_requirements
    check_requirements('aiohttp')
    import aiohttp

    def read_records():
        with ndjson_path.open() as file:
            return [json.loads(line) for line in file if line.strip()]
    lines = await asyncio.get_running_loop().run_in_executor(None, read_records)
    (dataset_record, image_records) = (lines[0], lines[1:])
    task = dataset_record.get('task', 'detect')
    is_classification = task == 'classify'
    is_depth = task == 'depth'
    depth_scale = dataset_record.get('depth_scale', 1000)
    if is_depth and (not isinstance(depth_scale, (int, float)) or isinstance(depth_scale, bool) or (not math.isfinite(depth_scale)) or (depth_scale <= 0)):
        raise ValueError('Depth datasets require a positive finite depth_scale')
    class_names = {int(k): v for (k, v) in dataset_record.get('class_names', {}).items()}
    classification_ids = set()
    local_path = dataset_record.pop('path', None) if local and (not (is_classification or is_depth)) else None
    _h = hashlib.sha256(repr(fraction).encode())
    for (i, r) in enumerate(lines):
        if i:
            (split, source_name) = (r.get('split'), r.get('file'))
            if split not in {'train', 'val', 'test'}:
                raise ValueError(f'Invalid NDJSON split: {split!r}')
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(f'Invalid NDJSON image name: {source_name!r}')
            if local_path:
                if source_name != Path(source_name).name:
                    raise ValueError(f'Invalid NDJSON image name: {source_name!r}')
                r['url'] = (ndjson_path.parent / local_path / 'images' / split / source_name).resolve()
            suffix = source_name.rsplit('.', 1)[-1]
            stems = (Path(clean_url(r.get('url') or '')).stem, Path(source_name).stem)
            content_hash = next((s.lower() for s in stems if len(s) == 32 and all((c in '0123456789abcdef' for c in s))), None)
            stem = f'{content_hash}_{i}' if content_hash else i
            r['file'] = f'{stem}.{suffix}' if suffix.isalnum() and len(suffix) <= 10 else f'{stem}.jpg'
            if is_classification:
                ids = r.get('annotations', {}).get('classification', [])
                class_id = ids[0] if ids else 0
                if not isinstance(class_id, int):
                    raise ValueError(f'Invalid NDJSON classification ID: {class_id!r}')
                classification_ids.add(class_id)
        hash_record = {k: v for (k, v) in r.items() if k != 'url'}
        if isinstance(r.get('depth'), dict):
            hash_record['depth'] = {k: v for (k, v) in r['depth'].items() if k != 'url'}
            if r['depth'].get('url'):
                hash_record['depth']['_source'] = clean_url(r['depth']['url'])
        if r.get('file'):
            hash_record['_source'] = clean_url(r['url']) if r.get('url') else str(ndjson_path.parent.resolve())
        _h.update(json.dumps(hash_record, sort_keys=True).encode())
    _hash = _h.hexdigest()[:8]
    class_dirs = {class_id: f'{i:06d}' for (i, class_id) in enumerate(sorted(classification_ids))}
    classification_names = {i: class_names.get(class_id, str(class_id)) for (i, class_id) in enumerate(class_dirs)}
    if is_depth:
        for record in image_records:
            depth = record.get('depth')
            if not isinstance(depth, dict) or not isinstance(depth.get('url'), str) or (not depth['url']):
                raise ValueError(f"Depth record '{record.get('file', '<unknown>')}' is missing depth.url")
    dataset_dir = output_path / f'{ndjson_path.stem}-{_hash}'
    metadata_path = dataset_dir / ('.ndjson.yaml' if is_classification else 'data.yaml')
    if metadata_path.is_file():
        try:
            cached = YAML.load(metadata_path)
            if cached.get('hash') == _hash and cached.get('complete') is True:
                return dataset_dir if is_classification else metadata_path
        except Exception:
            pass
    splits = {record['split'] for record in image_records}
    if not is_classification:
        if 'train' not in splits:
            raise ValueError(f"Dataset missing required 'train' split. Found splits: {sorted(splits)}")
        if 'val' not in splits:
            train_records = [r for r in image_records if r.get('split') == 'train']
            if len(train_records) < 2:
                raise ValueError(f"Dataset has only {len(train_records)} image(s) and no 'val' split. Need at least 2 images to auto-split into train/val.")
            random.Random(0).shuffle(train_records)
            val_count = max(1, len(train_records) // 10)
            for r in train_records[:val_count]:
                r['split'] = 'val'
            splits.add('val')
            LOGGER.warning(f"No 'val' split found in dataset. Auto-splitting {len(train_records)} images into {len(train_records) - val_count} train, {val_count} val. For best results, manually assign validation images in Platform dataset page.")
    inferred_nc = None
    if not is_classification:
        class_ids = {int(label[0]) for record in image_records for labels in record.get('annotations', {}).values() for label in labels if label}
        if class_ids or class_names:
            max_class_id = max(class_ids | set(class_names))
            if class_names:
                for i in range(max_class_id + 1):
                    class_names.setdefault(i, f'class{i}')
            else:
                inferred_nc = max_class_id + 1
    if task == 'pose' and 'kpt_shape' not in dataset_record:
        dataset_record['kpt_shape'] = _infer_ndjson_kpt_shape(image_records)
    selected = []
    for split in ('train', 'val', 'test'):
        limit = get_split_fraction(fraction, split)
        if limit:
            records = sorted((r for r in image_records if r['split'] == split), key=lambda r: r['file'])
            count = min(limit if type(limit) is int else round(len(records) * limit), len(records))
            selected.extend((records[i] for i in np.linspace(0, len(records) - 1, count, dtype=int)))
    image_records = selected
    split_counts = {split: sum((r['split'] == split for r in image_records)) for split in ('train', 'val', 'test')}
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = None
    if not is_classification:
        if is_depth:
            data_yaml = {'task': 'depth', 'nc': 1, 'names': {0: 'depth'}, 'depth_scale': depth_scale}
        else:
            data_yaml = dict(dataset_record)
            if class_names:
                data_yaml['names'] = class_names
            elif inferred_nc is not None:
                data_yaml['nc'] = inferred_nc
        data_yaml.pop('class_names', None)
        data_yaml.pop('type', None)
        for split in sorted(splits):
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / ('depth' if is_depth else 'labels') / split).mkdir(parents=True, exist_ok=True)
            data_yaml[split] = f'images/{split}'

    async def ensure_file(session, path, url):
        """Return True when the file exists locally, otherwise download one URL with the retry policy."""
        if path.exists():
            return True
        if not url:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(url, Path):
            if not url.is_file():
                return False
            await asyncio.get_running_loop().run_in_executor(None, shutil.copy2, url, path)
            return True
        for attempt in range(3):
            error = None
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(sock_connect=30, sock_read=30)) as response:
                    response.raise_for_status()
                    path.write_bytes(await response.read())
                return True
            except aiohttp.ClientResponseError as e:
                error = e
                if e.status not in {408, 429} and e.status < 500:
                    LOGGER.warning(f'Failed to download {clean_url(url)}: HTTP {e.status}')
                    return False
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error = e
            except Exception as e:
                LOGGER.warning(f'Failed to save {clean_url(url)}: {e}')
                return False
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                LOGGER.warning(f"Failed to download {clean_url(url)} after 3 attempts: {(type(error).__name__ if error else 'unknown')}")
        return False

    async def process_record(session, semaphore, record):
        """Process single image record with async session."""
        async with semaphore:
            (split, original_name) = (record['split'], record['file'])
            annotations = record.get('annotations', {})
            if is_classification:
                class_ids = annotations.get('classification', [])
                class_id = class_ids[0] if class_ids else 0
                class_name = class_dirs[class_id]
                image_path = dataset_dir / split / class_name / original_name
            else:
                image_path = dataset_dir / 'images' / split / original_name
                if not is_depth:
                    stem = original_name.rsplit('.', 1)[0] or original_name
                    label_path = dataset_dir / 'labels' / split / f'{stem}.txt'
                    lines_to_write = []
                    for key in annotations:
                        lines_to_write = [' '.join(map(str, item)) for item in annotations[key]]
                        break
                    label_path.write_text('\n'.join(lines_to_write) + '\n' if lines_to_write else '')
            image_ok = await ensure_file(session, image_path, record.get('url'))
            if not is_depth:
                return image_ok
            stem = original_name.rsplit('.', 1)[0] or original_name
            depth_path = dataset_dir / 'depth' / split / f'{stem}.png'
            depth_ok = await ensure_file(session, depth_path, record['depth']['url'])
            if not image_ok or not depth_ok:
                image_path.unlink(missing_ok=True)
                depth_path.unlink(missing_ok=True)
                return False
            return True
    semaphore = asyncio.Semaphore(min(128, len(image_records)))
    async with aiohttp.ClientSession(trust_env=True) as session:
        pbar = TQDM(total=len(image_records), desc=f"Converting {ndjson_path.name} fraction={fraction} → {dataset_dir} using {split_counts['train']} train, {split_counts['val']} val, {split_counts['test']} test images")

        async def tracked_process(record):
            result = await process_record(session, semaphore, record)
            pbar.update(1)
            return result
        success_count = 0
        for start in range(0, len(image_records), 1024):
            results = await asyncio.gather(*[tracked_process(record) for record in image_records[start:start + 1024]])
            success_count += sum(results)
        pbar.close()
    if not image_records or success_count < len(image_records):
        raise RuntimeError(f'Downloaded {success_count}/{len(image_records)} images from {ndjson_path}')
    if is_classification:
        YAML.save(metadata_path, {'names': classification_names, 'hash': _hash, 'complete': True})
        return dataset_dir
    else:
        data_yaml.update(hash=_hash, complete=True)
        YAML.save(metadata_path, data_yaml)
        return metadata_path