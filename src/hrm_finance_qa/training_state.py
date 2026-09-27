from __future__ import annotations

import json
import os
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch


TRAINING_STATE_SCHEMA = "document-hrm-training-state-v1"


class TrainingStateError(ValueError):
    """Raised when a resumable training state is missing or inconsistent."""


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def encode_python_random_state(state: object) -> list[object]:
    version, internal, gaussian = state  # type: ignore[misc]
    return [int(version), [int(value) for value in internal], gaussian]


def decode_python_random_state(payload: object) -> tuple[object, ...]:
    if not isinstance(payload, list) or len(payload) != 3:
        raise TrainingStateError("Invalid Python RNG state")
    version, internal, gaussian = payload
    if not isinstance(version, int) or not isinstance(internal, list):
        raise TrainingStateError("Invalid Python RNG state")
    if not all(isinstance(value, int) for value in internal):
        raise TrainingStateError("Invalid Python RNG state")
    return (version, tuple(internal), gaussian)


def restore_optimizer_state_device(
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> None:
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(target)


def load_training_state(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise TrainingStateError(f"Training state not found: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TrainingStateError(
            f"Cannot load training state: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise TrainingStateError("Training state payload must be a mapping")
    if payload.get("schema") != TRAINING_STATE_SCHEMA:
        raise TrainingStateError("Unsupported training state schema")
    contract = payload.get("training_contract")
    observed = payload.get("training_contract_sha256")
    if not isinstance(contract, dict) or not isinstance(observed, str):
        raise TrainingStateError("Training state contract is missing")
    if canonical_payload_sha256(contract) != observed:
        raise TrainingStateError("Training state contract hash mismatch")
    completed_epoch = payload.get("completed_epoch")
    best_epoch = payload.get("best_epoch")
    if not isinstance(completed_epoch, int) or completed_epoch < 0:
        raise TrainingStateError("Invalid completed epoch")
    if not isinstance(best_epoch, int) or not 0 <= best_epoch <= completed_epoch:
        raise TrainingStateError("Invalid best epoch")
    for key in ("model_state_dict", "optimizer_state_dict"):
        if not isinstance(payload.get(key), dict):
            raise TrainingStateError(f"Training state is missing {key}")
    decode_python_random_state(payload.get("python_random_state"))
    decode_python_random_state(payload.get("training_rng_state"))
    if not torch.is_tensor(payload.get("torch_cpu_rng_state")):
        raise TrainingStateError("Training state is missing Torch CPU RNG")
    return payload


def atomic_save_training_state(
    payload: Mapping[str, Any],
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        load_training_state(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def capture_python_random_state() -> list[object]:
    return encode_python_random_state(random.getstate())

