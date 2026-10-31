from collections import deque
from functools import wraps
from typing import Any
import numpy as np
import torch
from ultralytics.utils.metrics import bbox_ioa
from ..utils import LOGGER
from .basetrack import TrackState
from .bot_sort import BOTrack
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH
from .utils.reid import smooth_feature
from .utils.stracks import joint_stracks, merge_track_pools, multi_gmc, parse_bboxes
_CORNER_DX_IDX = np.array([0, 0, 2, 2])
_CORNER_DY_IDX = np.array([1, 3, 1, 3])
_LOOSE_NMS_IOU = 0.95
_LOOSE_NMS_DEDUP_IOU = 0.97

def _hmiou_distance(tracks_a, tracks_b):
    """Return (iou_sim, 1 - HMIoU) where HMIoU = HIoU * IoU and HIoU is vertical-overlap / vertical-union."""
    (n, m) = (len(tracks_a), len(tracks_b))
    if n == 0 or m == 0:
        return (np.zeros((n, m), dtype=np.float32), np.ones((n, m), dtype=np.float32))
    boxes_a = np.ascontiguousarray([track.xyxy for track in tracks_a], dtype=np.float32)
    boxes_b = np.ascontiguousarray([track.xyxy for track in tracks_b], dtype=np.float32)
    iou_sim = bbox_ioa(boxes_a, boxes_b, iou=True)
    h_over = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    h_union = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.minimum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    h_iou = np.clip(h_over / (h_union + 1e-09), 0, 1)
    return (iou_sim, 1.0 - h_iou * iou_sim)

def _angle_distance(tracks, dets, frame_id, pairs, delta_t=3):
    """Return angle distance for the IoU-supported `(track, det)` index `pairs` only.

    `_cost_matrix` overwrites unsupported pairs, so a full `(N, M)` grid would compute discarded work.
    """
    (track_idx, det_idx) = pairs
    track_boxes = np.stack([tracks[i].get_history_box(frame_id, delta_t) for i in track_idx])
    det_boxes = np.stack([dets[i].xyxy for i in det_idx])
    deltas = det_boxes - track_boxes
    dx = deltas[:, _CORNER_DX_IDX]
    dy = deltas[:, _CORNER_DY_IDX]
    norms = np.sqrt(dx * dx + dy * dy) + 1e-05
    dx /= norms
    dy /= norms
    track_velocities = np.stack([tracks[i].velocity for i in track_idx])
    dot = track_velocities[:, :, 0] * dx + track_velocities[:, :, 1] * dy
    dist = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi
    return dist * np.array([dets[i].score for i in det_idx])

def _confidence_distance(tracks, dets):
    """Absolute difference between each track's projected score and each detection's confidence."""
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    track_prev_scores = np.array([track.prev_score for track in tracks])
    track_curr_scores = np.array([track.score for track in tracks])
    track_proj_scores = track_curr_scores + (track_curr_scores - track_prev_scores)
    det_scores = np.array([det.score for det in dets])
    return np.abs(track_proj_scores[:, None] - det_scores[None])

def _iterative_associate(cost, match_thr, reduce_step=0.05):
    """Greedy mutually-nearest matching with a threshold that shrinks each iteration.

    Returns (matches, unmatched_tracks, unmatched_dets).
    """
    matches = []
    cost = cost.copy()
    while cost.shape[0] > 0 and cost.shape[1] > 0:
        nearest_det = np.argmin(cost, axis=1)
        nearest_track = np.argmin(cost, axis=0)
        new_matches = [[track_idx, nearest_det[track_idx]] for track_idx in range(cost.shape[0]) if nearest_track[nearest_det[track_idx]] == track_idx and cost[track_idx, nearest_det[track_idx]] < match_thr]
        if not new_matches:
            break
        matches.extend(new_matches)
        for (track_idx, det_idx) in new_matches:
            cost[track_idx, :] = np.inf
            cost[:, det_idx] = np.inf
        match_thr -= reduce_step
    matched_tracks = {track_idx for (track_idx, _) in matches}
    matched_dets = {det_idx for (_, det_idx) in matches}
    unmatched_tracks = [i for i in range(cost.shape[0]) if i not in matched_tracks]
    unmatched_dets = [i for i in range(cost.shape[1]) if i not in matched_dets]
    return (matches, unmatched_tracks, unmatched_dets)

def _track_aware_nms(tracks, dets, tai_thr, new_track_thresh):
    """TAI NMS: suppress detections that heavily overlap an existing track or a stronger detection."""
    if not dets:
        return []
    scores = np.array([det.score for det in dets])
    allow = scores > new_track_thresh
    (n_tracks, n_dets) = (len(tracks), len(dets))
    if n_tracks + n_dets < 2:
        return allow.tolist()
    boxes = np.ascontiguousarray([obj.xyxy for obj in tracks + dets], dtype=np.float32)
    iou = bbox_ioa(boxes, boxes, iou=True)
    if n_tracks:
        allow &= iou[n_tracks:, :n_tracks].max(axis=1) <= tai_thr
    det_iou = iou[n_tracks:, n_tracks:]
    order = scores.argsort()[::-1]
    for i in order:
        if not allow[i]:
            continue
        suppress = det_iou[i] > tai_thr
        suppress[i] = False
        allow[suppress] = False
    return allow.tolist()

def attach_raw_preds_hook(predictor):
    """Wrap `predictor.postprocess` to capture raw pre-NMS predictions and inputs (idempotent)."""
    if hasattr(predictor, '_orig_postprocess'):
        return
    orig = predictor.postprocess

    @wraps(orig)
    def _wrapped(preds, img, orig_imgs, *args, **kwargs):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        predictor._raw_preds = raw.detach().clone() if isinstance(raw, torch.Tensor) else raw
        predictor._postprocess_im = img
        predictor._postprocess_im0s = orig_imgs
        return orig(preds, img, orig_imgs, *args, **kwargs)
    predictor._orig_postprocess = orig
    predictor.postprocess = _wrapped

def compute_dets_del(predictor):
    """Return per-batch `(xywh, conf, cls)` tuples for detections the tight NMS dropped, or None if unavailable."""
    raw = getattr(predictor, '_raw_preds', None)
    if raw is None or not isinstance(raw, torch.Tensor):
        return None
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou
    loose_results = predictor._orig_postprocess(raw, predictor._postprocess_im, predictor._postprocess_im0s, iou=_LOOSE_NMS_IOU)
    is_obb = predictor.args.task == 'obb'
    out = []
    for (loose, tight) in zip(loose_results, predictor.results):
        tight_boxes = tight.obb if is_obb else tight.boxes
        loose_boxes = loose.obb if is_obb else loose.boxes
        if len(loose_boxes) == 0 or len(tight_boxes) == 0:
            out.append(None)
            continue
        max_iou = box_iou(loose_boxes.xyxy, tight_boxes.xyxy).max(dim=1).values
        mask = max_iou < _LOOSE_NMS_DEDUP_IOU
        if not mask.any():
            out.append(None)
            continue
        dels = loose_boxes.data[mask].cpu()
        if is_obb:
            xywh = dels[:, :5].numpy()
            out.append((xywh, dels[:, 5].numpy(), dels[:, 6].numpy()))
        else:
            xywh = ops.xyxy2xywh(dels[:, :4]).numpy()
            out.append((xywh, dels[:, 4].numpy(), dels[:, 5].numpy()))
    predictor._raw_preds = None
    return out

def _cosine_distance(tracks, dets):
    """Cosine distance in `[0, 1]` between track and detection embeddings; NaN where either side has no feature.

    A NaN entry signals "no appearance evidence for this pair" so the caller falls back to motion rather than treating a
    missing/occlusion-suppressed embedding as maximally dissimilar (which would penalize true matches).
    """
    if not tracks or not dets:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    tfeat = [t.smooth_feat if t.smooth_feat is not None else t.curr_feat for t in tracks]
    dfeat = [d.curr_feat for d in dets]
    dim = next((f.shape[0] for f in (*tfeat, *dfeat) if f is not None), 128)
    zeros = np.zeros(dim, dtype=np.float32)
    T = np.asarray([f if f is not None else zeros for f in tfeat], dtype=np.float32)
    D = np.asarray([f if f is not None else zeros for f in dfeat], dtype=np.float32)
    valid = np.array([f is not None for f in tfeat])[:, None] & np.array([f is not None for f in dfeat])[None, :]
    return np.where(valid, np.clip(1 - T @ D.T, 0, 1), np.nan).astype(np.float32)

class TTSTrack(BOTrack):
    """Single-object track for TrackTrack with corner velocity, score history, and ReID features.

    Extends `BOTrack` (XYWH Kalman state and EMA ReID smoothing), adding corner-velocity motion, score history, and
    score-adaptive feature smoothing.

    Attributes:
        min_track_len (int): Class-level default; overridden by TRACKTRACK from config.
        kalman_filter (KalmanFilterXYWH): Per-track Kalman filter used after activation.
        mean (np.ndarray): Mean state vector.
        covariance (np.ndarray): Covariance matrix.
        score (float): Current detection confidence.
        prev_score (float): Confidence from the previous update (for score projection).
        tracklet_len (int): Successful updates since activation.
        velocity (np.ndarray): Per-corner (4,2) unit velocity vectors.
        smooth_feat (np.ndarray | None): EMA-smoothed ReID embedding.
        curr_feat (np.ndarray | None): Raw ReID embedding from the current frame.

    Examples:
        Create and activate a new track
        >>> track = TTSTrack(np.array([100, 200, 50, 80, 0]), score=0.9, cls="person")
        >>> track.activate(KalmanFilterXYWH(), frame_id=1)
    """
    min_track_len = 3
    _alpha = 0.95
    _delta_t = 3

    def __init__(self, xywh, score, cls, feat=None):
        """Initialize a TTSTrack from a detection bounding box.

        Args:
            xywh (np.ndarray): `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)`, center-based with detection index.
            score (float): Detection confidence.
            cls (Any): Class label.
            feat (np.ndarray | None): Optional ReID feature vector.
        """
        super().__init__(xywh, score, cls)
        self.prev_score = score
        self.velocity = np.zeros((4, 2), dtype=np.float32)
        self._history = deque(maxlen=self._delta_t + 1)
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        """Normalize `feat` and blend it into `smooth_feat` via score-adaptive EMA."""
        beta = self._alpha + (1 - self._alpha) * (1 - self.score)
        (curr, smooth) = smooth_feature(feat, self.smooth_feat, beta)
        if curr is not None:
            (self.curr_feat, self.smooth_feat) = (curr, smooth)

    def get_history_box(self, frame_id, dt):
        """Return the box from `dt` frames back, or the most recent box, or the current box."""
        target = frame_id - dt
        for (fid, box) in self._history:
            if fid == target:
                return box.copy()
        if self._history:
            return self._history[-1][1].copy()
        return self.xyxy

    def activate(self, kalman_filter, frame_id):
        """Initialize Kalman state and promote to New state."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        (self.mean, self.covariance) = kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._history.append((frame_id, self.xyxy))
        self.tracklet_len = 0
        self.state = TrackState.Tracked if self.min_track_len <= 1 else TrackState.New
        self.is_activated = frame_id == 1 or self.state == TrackState.Tracked
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Rebind a lost track to a fresh detection via NSA-Kalman."""
        self.prev_score = self.score
        (self.mean, self.covariance) = self.kalman_filter.update(self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score)
        self._history.append((frame_id, self.xyxy))
        self.score = new_track.score
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        (self.cls, self.angle, self.idx) = (new_track.cls, new_track.angle, new_track.idx)

    def update(self, new_track, frame_id):
        """Update a matched track with a new detection; promote to Tracked after min_track_len."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.prev_score = self.score
        (self.mean, self.covariance) = self.kalman_filter.update(self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score)
        self._history.append((frame_id, new_track.xyxy))
        velocity = np.zeros((4, 2), dtype=np.float32)
        curr_box = new_track.xyxy
        for dt in range(1, self._delta_t + 1):
            delta = curr_box - self.get_history_box(frame_id, dt)
            (dx, dy) = (delta[_CORNER_DX_IDX], delta[_CORNER_DY_IDX])
            norm = np.sqrt(dx * dx + dy * dy) + 1e-05
            velocity += np.stack([dx / norm, dy / norm], axis=-1) / dt
        self.velocity = velocity / self._delta_t
        self.score = new_track.score
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        if self.state == TrackState.Tracked or self.tracklet_len + 1 >= self.min_track_len:
            self.state = TrackState.Tracked
            self.is_activated = True
        (self.cls, self.angle, self.idx) = (new_track.cls, new_track.angle, new_track.idx)

    def __repr__(self):
        """Short string representation of the track."""
        return f'TT_{self.track_id}_({self.start_frame}-{self.end_frame})'

class TRACKTRACK:
    """Multi-object tracker implementing Track-Perspective Association and Track-Aware Initialization.

    Detections are partitioned into high, low, and deleted (loose-NMS recovered) sets, then matched against the union of
    tracked and lost tracks using a multi-cue cost (HMIoU + cosine ReID + confidence + angle distance) solved with
    iterative assignment. Unmatched still-Lost tracks may optionally be re-associated against leftover detections in a
    relaxed second pass, and surviving leftover detections spawn new tracks via track-aware NMS.

    Attributes:
        tracked_stracks (list[TTSTrack]): Currently tracked tracks.
        lost_stracks (list[TTSTrack]): Tracks that lost their detection but remain within the buffer window.
        frame_id (int): Current frame index.
        args (Any): Parsed tracker configuration.
        max_time_lost (int): Frame budget before a lost track is removed (scaled to source frame rate).
        kalman_filter (KalmanFilterXYWH): Kalman filter for new-track initialization.
        match_thr (float): Cost gate for the main iterative assignment.
        lost_match_thr (float): Cost gate for the optional relaxed lost-rebind pass; 0 disables it.
        gmc (GMC): Global motion compensation for camera-motion warping.
        encoder (Any): ReID encoder, or None when ReID is disabled.

    Methods:
        update: Advance the tracker by one frame and return per-object tracking results.
        reset: Clear all tracker state.

    Examples:
        Initialize and run on a single frame
        >>> tracker = TRACKTRACK(args)
        >>> tracked_objects = tracker.update(yolo_results, img=image)
    """

    def __init__(self, args):
        """Initialize TRACKTRACK from a tracker config (see `ultralytics/cfg/trackers/tracktrack.yaml`).

        Args:
            args (Any): Parsed tracker configuration. All knobs are read with `getattr(..., default)` so legacy YAMLs
                missing recently added keys still load.
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.max_time_lost = args.track_buffer
        self.kalman_filter = KalmanFilterXYWH()
        self.match_thr = getattr(args, 'match_thresh', 0.7)
        self.lost_match_thr = getattr(args, 'lost_match_thr', 0.0)
        self.penalty_p = getattr(args, 'penalty_p', 0.2)
        self.penalty_q = getattr(args, 'penalty_q', 0.4)
        self.reduce_step = getattr(args, 'reduce_step', 0.05)
        self.iou_weight = getattr(args, 'iou_weight', 0.5)
        self.reid_weight = getattr(args, 'reid_weight', 0.5)
        self.conf_weight = getattr(args, 'conf_weight', 0.1)
        self.angle_weight = getattr(args, 'angle_weight', 0.05)
        self.tai_thr = getattr(args, 'tai_thr', 0.55)
        self.new_track_thresh = getattr(args, 'new_track_thresh', 0.7)
        self.min_track_len = getattr(args, 'min_track_len', 3)
        self.gmc = GMC(method=getattr(args, 'gmc_method', 'sparseOptFlow'))
        from .utils.reid import build_encoder
        self.encoder = build_encoder(getattr(args, 'with_reid', False), getattr(args, 'model', 'auto'), getattr(args, 'device', None))

    @classmethod
    def setup_predictor(cls, predictor):
        """Attach the raw-predictions hook for Track-Aware Initialization (detect/obb only).

        Recovered (loose-NMS) detections are box-only and have no row in the post-NMS Results, so on segment/pose
        tasks they cannot carry mask/keypoint data and would mis-index downstream; skip recovery (and its
        per-frame overhead) for those tasks and RT-DETR predictors, which do not suppress boxes with NMS.
        """
        from ultralytics.models.yolo.detect import DetectionPredictor
        if predictor.args.task in {'detect', 'obb'} and isinstance(predictor, DetectionPredictor):
            attach_raw_preds_hook(predictor)

    @classmethod
    def compute_frame_extras(cls, predictor):
        """Return per-batch ``(xywh, conf, cls)`` tuples for detections dropped by tight NMS."""
        return compute_dets_del(predictor)

    def _cost_matrix(self, tracks, dets):
        """Return the multi-cue cost matrix (HMIoU + ReID + confidence + angle), gated by IoU support."""
        (iou_sim, hmiou_dist) = _hmiou_distance(tracks, dets)
        if self.encoder is not None:
            cos = _cosine_distance(tracks, dets)
            cost = np.where(np.isnan(cos), hmiou_dist, self.iou_weight * hmiou_dist + self.reid_weight * cos)
        else:
            cost = hmiou_dist
        cost += self.conf_weight * _confidence_distance(tracks, dets)
        if iou_sim.size > 0:
            supported = iou_sim > 0.1
            pairs = np.nonzero(supported)
            if len(pairs[0]):
                cost[pairs] += self.angle_weight * _angle_distance(tracks, dets, self.frame_id, pairs)
            cost[~supported] = 1.0
        return np.clip(cost, 0, 1)

    def _apply_gmc(self, img, detections, pools):
        """Warp `pools` in place by the current GMC affine."""
        try:
            warp = self.gmc.apply(img, [det.xyxy for det in detections])
        except Exception as e:
            LOGGER.warning(f'GMC failed, falling back to identity: {e}')
            warp = np.eye(2, 3)
        for pool in pools:
            multi_gmc(pool, warp)

    def update(self, results, img=None, dets_del=None, **kwargs):
        """Advance the tracker by one frame and return an `(N, 8)` array of `[x1, y1, x2, y2, id, score, cls, idx]`."""
        self.frame_id += 1
        (activated, refind, lost, removed) = ([], [], [], [])
        scores = np.asarray(results.conf)
        boxes = parse_bboxes(results)
        high_mask = scores >= self.args.track_high_thresh
        low_mask = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        def _new_track(box, score, cls, feat=None):
            track = TTSTrack(box, score, cls, feat) if feat is not None else TTSTrack(box, score, cls)
            track.min_track_len = self.min_track_len
            return track
        (high_boxes, high_scores, high_cls) = (boxes[high_mask], scores[high_mask], results.cls[high_mask])
        feats = kwargs.get('feats')
        use_native = getattr(self.args, 'model', 'auto') == 'auto'
        encoder_input = None
        if self.encoder is not None and len(high_boxes) > 0:
            if use_native:
                encoder_input = feats[high_mask] if feats is not None and len(feats) else None
            elif img is not None:
                encoder_input = img
        if encoder_input is not None:
            features = self.encoder(encoder_input, high_boxes)
            dets_high = [_new_track(b, s, c, f) for (b, s, c, f) in zip(high_boxes, high_scores, high_cls, features)]
        else:
            dets_high = [_new_track(b, s, c) for (b, s, c) in zip(high_boxes, high_scores, high_cls)]
        dets_low = [_new_track(b, s, c) for (b, s, c) in zip(boxes[low_mask], scores[low_mask], results.cls[low_mask])]
        dets_recovered = []
        if dets_del is not None:
            (del_xywh, del_conf, del_cls) = dets_del
            mask = del_conf > self.args.track_high_thresh
            if mask.any():
                del_boxes = np.concatenate([del_xywh[mask], -np.ones((mask.sum(), 1))], axis=-1)
                dets_recovered = [_new_track(b, s, c) for (b, s, c) in zip(del_boxes, del_conf[mask], del_cls[mask])]
        (unconfirmed, tracked) = ([], [])
        for track in self.tracked_stracks:
            (tracked if track.state == TrackState.Tracked else unconfirmed).append(track)
        pool = joint_stracks(tracked, self.lost_stracks)
        if img is not None and self.gmc.method is not None:
            self._apply_gmc(img, dets_high, [pool, unconfirmed])
        TTSTrack.multi_predict(pool)
        TTSTrack.multi_predict(unconfirmed)
        all_dets = dets_high + dets_low + dets_recovered
        (n_high, n_low) = (len(dets_high), len(dets_low))
        cost = self._cost_matrix(pool, all_dets)
        if cost.shape[1] > n_high:
            cost[:, n_high:n_high + n_low] += self.penalty_p
        if dets_recovered:
            cost[:, n_high + n_low:] += self.penalty_q
        cost = np.clip(cost, 0, 1)
        (matches, unmatched_tracks, unmatched_dets) = _iterative_associate(cost, self.match_thr, self.reduce_step)
        for (track_idx, det_idx) in matches:
            (track, det) = (pool[track_idx], all_dets[det_idx])
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        for track_idx in unmatched_tracks:
            track = pool[track_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)
        leftover = [all_dets[i] for i in unmatched_dets if i < n_high]
        if unconfirmed and leftover:
            uc_cost = self._cost_matrix(unconfirmed, leftover)
            (uc_matches, uc_unmatched_tracks, uc_unmatched_dets) = _iterative_associate(uc_cost, self.match_thr, self.reduce_step)
            for (track_idx, det_idx) in uc_matches:
                unconfirmed[track_idx].update(leftover[det_idx], self.frame_id)
                activated.append(unconfirmed[track_idx])
            for track_idx in uc_unmatched_tracks:
                unconfirmed[track_idx].mark_removed()
                removed.append(unconfirmed[track_idx])
            leftover = [leftover[i] for i in uc_unmatched_dets]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed.append(track)
        if self.lost_match_thr > 0 and leftover:
            unmatched_lost = [t for t in pool if t.state == TrackState.Lost and t not in lost]
            if unmatched_lost:
                lost_cost = self._cost_matrix(unmatched_lost, leftover)
                (lost_matches, _, lost_unmatched) = _iterative_associate(lost_cost, self.lost_match_thr, self.reduce_step)
                for (track_idx, det_idx) in lost_matches:
                    unmatched_lost[track_idx].re_activate(leftover[det_idx], self.frame_id, new_id=False)
                    refind.append(unmatched_lost[track_idx])
                leftover = [leftover[i] for i in lost_unmatched]
        active = [track for track in self.tracked_stracks if track.state == TrackState.Tracked] + activated
        for (det, ok) in zip(leftover, _track_aware_nms(active, leftover, self.tai_thr, self.new_track_thresh)):
            if ok:
                det.activate(self.kalman_filter, self.frame_id)
                activated.append(det)
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)
        merge_track_pools(self, activated, refind, lost, removed)
        return np.asarray([track.result for track in self.tracked_stracks if track.is_activated and track.frame_id == self.frame_id], dtype=np.float32)

    def reset(self):
        """Clear all tracker state including GMC warp history and the global ID counter."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        TTSTrack.reset_id()
        self.gmc.reset_params()