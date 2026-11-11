import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch


ROOT = Path(os.environ.get("OCTSSL_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs"))).resolve()
REVISION_ROOT = ROOT
CHECKPOINT_ROOT = REVISION_ROOT / "checkpoints"
DEFAULT_RESERVE_BYTES = 512 * 1024 ** 2
_METADATA_ALLOWANCE = 16 * 1024 ** 2
_VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _inside_root(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Checkpoint paths must not be symlinks: %s" % path)
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        raise ValueError("Checkpoint write escapes isolated rerun: %s" % path)
    if resolved == ROOT:
        raise ValueError("A file path is required")
    return resolved


def _stage_directory(path, allowed_root=ROOT):
    if Path(allowed_root).resolve() != ROOT:
        raise ValueError("Retirement is restricted to this isolated rerun")
    stage = _inside_root(path)
    if stage.parent != CHECKPOINT_ROOT or not _VALID_ID.fullmatch(stage.name):
        raise ValueError("Expected checkpoints/<experiment_id>: %s" % stage)
    if not stage.is_dir():
        raise FileNotFoundError("Stage directory does not exist: %s" % stage)
    return stage


def _fsync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha(path):

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimated_bytes(payload):

    visited = set()
    storages = set()
    size = 0

    def visit(value):
        nonlocal size
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if torch.is_tensor(value):
            if value.layout != torch.strided:
                if value.layout == torch.sparse_coo:
                    visit(value._indices())
                    visit(value._values())
                    return
                raise ValueError("Unsupported checkpoint tensor layout: %s" % value.layout)
            storage = value.untyped_storage() if hasattr(value, "untyped_storage") else value.storage()
            nbytes = storage.nbytes()
            key = (str(value.device), storage.data_ptr(), nbytes)
            if key not in storages:
                storages.add(key)
                size += nbytes
            size += 512
        elif isinstance(value, dict):
            size += 128 * len(value)
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (tuple, list, set, frozenset)):
            size += 64 * len(value)
            for item in value:
                visit(item)
        elif isinstance(value, str):
            size += 2 * len(value.encode("utf8")) + 128
        elif isinstance(value, (bytes, bytearray, memoryview)):
            size += len(value) + 128
        elif hasattr(value, "nbytes"):
            size += int(value.nbytes) + 512
        else:
            size += 256

    visit(payload)
    return size + _METADATA_ALLOWANCE


def _check_space(parent, required, reserve_bytes):
    reserve_bytes = int(reserve_bytes)
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be nonnegative")
    available = shutil.disk_usage(parent).free
    if available < required + reserve_bytes:
        raise OSError(
            "Insufficient checkpoint space: need estimated %d bytes plus %d "
            "reserve, available %d" % (required, reserve_bytes, available)
        )


def atomic_save(path, payload, reserve_bytes=DEFAULT_RESERVE_BYTES):

    path = _inside_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_space(path.parent, _estimated_bytes(payload), reserve_bytes)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        if hasattr(os, "fchmod"): os.fchmod(descriptor, 0o660)
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):

    path = _inside_root(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf8")
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_space(path.parent, len(raw), 0)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        if hasattr(os, "fchmod"): os.fchmod(descriptor, 0o660)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def light_bundle(full):





    required = {"schema", "config", "fingerprint", "model", "epoch", "best"}
    if not required.issubset(full):
        raise ValueError("Incomplete full checkpoint: %s" % sorted(required - set(full)))
    light = {key: value for key, value in full.items()
             if key not in {"optimizer", "teacher", "scheduler", "heads"}}
    light["checkpoint_kind"] = "inference"
    return light


def _read_json(path):
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("Metadata must be a regular file: %s" % path)
    value = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise ValueError("Metadata must contain a JSON object: %s" % path)
    return value


def _stat_identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Refusing to retire a nonregular checkpoint: %s" % path)
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _retire(stage_dir, names, consumer, allowed_root):
    stage = _stage_directory(stage_dir, allowed_root)
    if not isinstance(consumer, str) or not consumer.strip():
        raise ValueError("A nonempty consumer receipt is required")
    receipt_path = stage / "retired_checkpoint.json"
    receipt = _read_json(receipt_path)
    if receipt is None:
        receipt = {"schema": 1, "experiment_id": stage.name,
                   "root": str(ROOT), "files": {}, "events": []}
    if (receipt.get("schema") != 1 or receipt.get("experiment_id") != stage.name
            or receipt.get("root") != str(ROOT)
            or not isinstance(receipt.get("files"), dict)):
        raise ValueError("Retirement receipt belongs to a different stage")
    identities = {}
    for name in names:
        path = stage / name
        if path.exists() or path.is_symlink():
            before = _stat_identity(path)
            digest = sha(path)
            if _stat_identity(path) != before:
                raise RuntimeError("Checkpoint changed while hashing: %s" % path)
            previous = receipt["files"].get(name)
            if previous and previous.get("sha256") != digest:
                raise RuntimeError("Checkpoint differs from retirement receipt: %s" % path)
            receipt["files"][name] = {
                "path": str(path), "sha256": digest, "size_bytes": before[2],
                "consumer": consumer, "retention": "retire_requested"
            }
            identities[name] = before
        elif name not in receipt["files"]:


            if name == "best.pt":
                raise FileNotFoundError("Cannot retire an unrecorded missing best checkpoint")
    receipt.setdefault("events", []).append({"time": _now(), "consumer": consumer,
                                             "requested_files": list(names)})


    write_json(receipt_path, receipt)
    for name, identity in identities.items():
        path = stage / name
        if _stat_identity(path) != identity:
            raise RuntimeError("Checkpoint changed before retirement: %s" % path)
        os.unlink(str(path))
    _fsync_directory(stage)
    for name in names:
        if name in receipt["files"] and not (stage / name).exists():
            receipt["files"][name]["retention"] = "retired"
    receipt["updated_utc"] = _now()
    write_json(receipt_path, receipt)
    return receipt


def retire_stage(stage_dir, consumer, allowed_root=ROOT):

    return _retire(stage_dir, ("best.pt", "latest.pt"), consumer, allowed_root)


def retire_latest(stage_dir, consumer="stage_completed", allowed_root=ROOT):

    return _retire(stage_dir, ("latest.pt",), consumer, allowed_root)


def _recorded_best_hashes(stage):
    hashes = []
    for name in ("result.json", "provenance.json"):
        value = _read_json(stage / name)
        if value is None:
            continue
        for key in ("checkpoint_sha256", "best_checkpoint_sha256"):
            if value.get(key) is not None:
                hashes.append(value[key])
        artifacts = value.get("artifacts_sha256", {})
        if isinstance(artifacts, dict):
            for key, digest in artifacts.items():
                if key in (str(stage / "best.pt"), "best.pt"):
                    hashes.append(digest)
    receipt = _read_json(stage / "retired_checkpoint.json")
    if receipt is not None:
        if (receipt.get("experiment_id") != stage.name or receipt.get("root") != str(ROOT)):
            raise ValueError("Upstream retirement receipt belongs to a different stage")
        entry = receipt.get("files", {}).get("best.pt")
        if entry:
            hashes.append(entry.get("sha256"))
    else:
        entry = None
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in hashes):
        raise ValueError("Invalid recorded upstream SHA256")
    if len(set(hashes)) > 1:
        raise RuntimeError("Upstream checkpoint hashes disagree: %s" % stage)
    return hashes, entry


def resolve_upstream(root, eid):






    root = Path(root).resolve()
    if root == ROOT:
        root = REVISION_ROOT
    if root != REVISION_ROOT:
        raise ValueError("Upstream lookup escapes isolated rerun")
    if not isinstance(eid, str) or not _VALID_ID.fullmatch(eid):
        raise ValueError("Invalid upstream experiment ID")
    stage = _stage_directory(root / "checkpoints" / eid)
    hashes, retirement = _recorded_best_hashes(stage)
    best = stage / "best.pt"
    if best.exists() or best.is_symlink():
        identity = _stat_identity(best)
        digest = sha(best)
        if _stat_identity(best) != identity:
            raise RuntimeError("Upstream changed while hashing: %s" % best)
        if hashes and digest != hashes[0]:
            raise RuntimeError("Upstream checkpoint does not match its receipt")
        return best, digest
    if retirement is None or not hashes:
        raise FileNotFoundError("Upstream best is missing without a retirement receipt: %s" % best)
    return None, hashes[0]


def cleanup_uncommitted_temps():





    removed = []
    if not CHECKPOINT_ROOT.exists():
        return removed
    pattern = re.compile(r'^\.(best\.pt|latest\.pt|[A-Za-z0-9_]+\.json)\.[A-Za-z0-9_\-]+\.tmp$')
    for candidate in CHECKPOINT_ROOT.glob('*/*'):
        if not pattern.fullmatch(candidate.name):
            continue
        _stage_directory(candidate.parent)
        path = _inside_root(candidate)
        if not path.is_file():
            raise ValueError('Unexpected atomic temporary type')
        removed.append({'path':str(path),'size_bytes':path.stat().st_size})
        os.unlink(str(path))
        _fsync_directory(path.parent)
    if removed:
        write_json(REVISION_ROOT/'results'/'uncommitted_temp_cleanup.json',
                   {'time':_now(),'removed':removed,'exclusive_lock_held':True})
    return removed
