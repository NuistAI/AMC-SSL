from functools import lru_cache
import hashlib,json
from pathlib import Path
import numpy as np
import cv2
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_dataset(value):
    name = str(value).lower().replace("-", "").replace("_", "")
    if name in ("hcms", "dat aoct".replace(" ", "")):
        return "HCMS"
    if name in ("octa", "octa500"):
        return "OCTA500"
    raise ValueError("Unknown dataset: %s" % value)


@lru_cache(maxsize=128)
def _mat_layers(path):
    layer = loadmat(str(path))["Layer"]
    if layer.ndim != 3 or layer.shape != (6, 400, 400):
        raise ValueError("Unexpected OCTA 6mm Layer shape at %s: %s" % (path, layer.shape))
    if not np.isfinite(layer).all():
        raise ValueError("Nonfinite original OCTA annotation: %s" % path)
    return layer.astype(np.int32)


def rasterize_oct_layers(boundaries, height=640):







    boundaries = np.asarray(boundaries)
    if boundaries.shape != (6, 400):
        raise ValueError("Expected six boundary curves with 400 columns")
    if np.any(boundaries < 0) or np.any(boundaries > height):
        raise ValueError("Original boundary outside image height")
    rows = np.arange(height)[:, None]
    result = np.zeros((height, 400), dtype=np.uint8)
    for i in range(5):
        chosen = (rows >= boundaries[i, None, :]) & (rows <= boundaries[i + 1, None, :])
        result[chosen] = i + 1
    return result


def _read_grayscale(path):
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if encoded.size else None


def _load_record(record, include_label=True):
    image = _read_grayscale(record["image_path"])
    if image is None:
        raise FileNotFoundError(record["image_path"])
    if record["kind"] == "hcms_png":
        if image.shape != (128, 1024):
            raise ValueError("Unexpected HCMS image dimensions: %s" % (image.shape,))
        if include_label:
            raw_label = _read_grayscale(record["label_path"])
            if raw_label is None or raw_label.shape != image.shape:
                raise ValueError("Missing or mismatched HCMS label: %s" % record["label_path"])
            label = np.unique(raw_label, return_inverse=True)[1].reshape(image.shape).astype(np.int64)
        else:
            label = np.zeros(image.shape, dtype=np.int64)
    elif record["kind"] == "octa_raw_mat":
        if image.shape != (640, 400):
            raise ValueError("Unexpected raw OCTA image dimensions: %s" % (image.shape,))
        if include_label:
            boundaries = _mat_layers(record["label_path"])[:, record["slice_index"], :]
            label = rasterize_oct_layers(boundaries)
            label = cv2.resize(label[120:520, :400], (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        else:
            label = np.zeros((256, 256), dtype=np.int64)
        image = cv2.resize(image[120:520, :400], (256, 256), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("Unsupported record kind: %s" % record["kind"])
    if image.dtype != np.uint8:
        raise ValueError("Expected original uint8 intensities")
    return image, label


class RevisionDataset(Dataset):
    def __init__(self, config, split="train", labeled=None):
        self.config = dict(config)
        self.dataset_name = canonical_dataset(config["dataset"])
        self.split = split
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        split_record = None
        if config.get("split_file"):
            split_path = Path(config["split_file"])
            split_record = json.loads(split_path.read_text())
            if split_record["dataset"] != self.dataset_name:
                raise ValueError("Dataset and split_file disagree")
            self.split_sha256 = file_hash(split_path)
            protocol_path = Path(config.get("protocol_path") or split_record["protocol_path"])
            ratio = str(split_record["nominal_percent"])
        else:
            protocol_path = Path(config.get("protocol_path", "protocol.json"))
            ratio = str(int(float(config.get("label_ratio",20))))
            self.split_sha256 = None
        self.protocol_sha256 = file_hash(protocol_path)
        if split_record and split_record["protocol_sha256"] != self.protocol_sha256:
            raise RuntimeError("Split manifest references a different protocol hash")
        full_protocol = json.loads(protocol_path.read_text())
        self.protocol = full_protocol["datasets"][self.dataset_name]
        self.ratio = self.protocol["ratios"][ratio]
        self.num_classes = int(self.protocol["num_classes"])
        self.image_size = tuple(self.protocol["image_size"])
        self.labeled_subjects = set(self.ratio["labeled_subjects"])
        records = self.protocol["splits"][split]
        if split == "train" and labeled is not None:
            records = [r for r in records if (r["subject"] in self.labeled_subjects) == bool(labeled)]
        self.records = list(records)
        self.labeled_indices = [i for i, r in enumerate(self.records) if r["subject"] in self.labeled_subjects]
        self.unlabeled_indices = [i for i, r in enumerate(self.records) if r["subject"] not in self.labeled_subjects]
        self.ids = [r["id"] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        is_labeled = self.split != "train" or record["subject"] in self.labeled_subjects
        image, label = _load_record(record, include_label=is_labeled)
        if label.min() < 0 or label.max() >= self.num_classes:
            raise ValueError("Label outside configured class range")
        return {"image": torch.from_numpy(np.ascontiguousarray(image[None])).float().div(255.),
                "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
                "id": record["id"], "subject": record["subject"], "is_labeled": is_labeled}


def build_dataset(config, split="train", labeled=None):
    return RevisionDataset(config, split=split, labeled=labeled)
