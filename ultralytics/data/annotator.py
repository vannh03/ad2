from pathlib import Path
from ultralytics import SAM, YOLO

def auto_annotate(data, det_model='yolo26x.pt', sam_model='sam_b.pt', device='', conf=0.25, iou=0.45, imgsz=640, max_det=300, classes=None, output_dir=None):
    """Automatically annotate images using a YOLO object detection model and a SAM segmentation model.

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates
    segmentation masks using a SAM model. The resulting annotations are saved as text files in YOLO format.

    Args:
        data (str | Path): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        sam_model (str): Path or name of the pre-trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0'). Empty string for auto-selection.
        conf (float): Confidence threshold for detection model.
        iou (float): IoU threshold for filtering overlapping boxes in detection results.
        imgsz (int): Input image resize dimension.
        max_det (int): Maximum number of detections per image.
        classes (list[int], optional): Filter predictions to specified class IDs, returning only relevant detections.
        output_dir (str | Path, optional): Directory to save the annotated results. If None, creates a default directory
            based on the input data path.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo26n.pt", sam_model="mobile_sam.pt")
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)
    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f'{data.stem}_auto_annotate_labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    det_results = det_model(data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes)
    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()
        if class_ids:
            boxes = result.boxes.xyxy
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn
            with open(f'{Path(output_dir) / Path(result.path).stem}.txt', 'w', encoding='utf-8') as f:
                for (i, s) in enumerate(segments):
                    if len(s) >= 3:
                        segment = map(str, s.reshape(-1).tolist())
                        f.write(f'{class_ids[i]} ' + ' '.join(segment) + '\n')