"""Runtime profile and path resolution for local and server execution."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path


MODES = ("auto", "local", "server")
SERVER_PATH_PREFIXES = ("/data/", "/scratch/", "/lustre/", "/work/")
LOCAL_ELECTRA_PATH = Path(
    "/mnt/f/XJJ/KLJP-master1/chinese-electra-180g-small-discriminator"
)
SERVER_ELECTRA_PATH = Path(
    "/mntF/XJJ/KLJP-master1/chinese-electra-180g-small-discriminator"
)
SERVER_DATA_ROOT = Path("/home/user/KLJP-DATA")
SERVER_LOG_ROOT = Path("/home/user/KLJP-logs")
SERVER_SCHEDULER_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_CLUSTER_NAME",
    "PBS_JOBID",
    "LSB_JOBID",
    "KUBERNETES_SERVICE_HOST",
)


def _normalise_mode(value: str | None) -> str | None:
    if value is None:
        return None
    mode = value.strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"Invalid KLJP runtime mode {value!r}; expected one of {MODES}."
        )
    return mode


def cli_mode(argv: list[str] | None = None) -> str | None:
    """Read only ``--mode`` without rejecting other application arguments."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=MODES)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args.mode


def detect_mode(
    project_root: Path, requested_mode: str | None = None
) -> tuple[str, str]:
    """Return ``(mode, reason)`` using explicit settings before heuristics."""

    requested = _normalise_mode(requested_mode)
    if requested in ("local", "server"):
        return requested, "explicit configuration"

    marker = project_root / ".kljp-server"
    if marker.is_file():
        return "server", f"server marker found at {marker}"

    if os.environ.get("KLJP_SERVER", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return "server", "KLJP_SERVER is enabled"

    for variable in SERVER_SCHEDULER_VARIABLES:
        if os.environ.get(variable):
            return "server", f"scheduler environment variable {variable} is set"

    if SERVER_ELECTRA_PATH.is_dir():
        return "server", f"server model mount found at {SERVER_ELECTRA_PATH}"

    project_text = project_root.as_posix().lower()
    if project_text.startswith(SERVER_PATH_PREFIXES):
        return "server", "project path uses a conventional server mount"

    server_host_pattern = os.environ.get("KLJP_SERVER_HOST_PATTERN")
    if server_host_pattern and server_host_pattern.lower() in socket.gethostname().lower():
        return "server", "hostname matches KLJP_SERVER_HOST_PATTERN"

    return "local", "no server marker or server environment detected"


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved paths and the explanation for the selected profile."""

    mode: str
    reason: str
    project_root: Path
    data_root: Path
    log_root: Path
    electra_path: Path
    embedding_path: Path
    allow_model_download: bool


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def build_runtime_config(
    project_root: Path, requested_mode: str | None = None
) -> RuntimeConfig:
    """Build the active profile; every path can be overridden by an env var."""

    project_root = Path(project_root).resolve()
    requested = (
        requested_mode
        or cli_mode()
        or os.environ.get("KLJP_MODE")
        or "auto"
    )
    mode, reason = detect_mode(project_root, requested)

    default_data_root = (
        SERVER_DATA_ROOT if mode == "server" else project_root / "KLJP-DATA"
    )
    data_root = Path(
        os.environ.get("KLJP_DATA_ROOT", str(default_data_root))
    ).expanduser()
    default_log_root = SERVER_LOG_ROOT if mode == "server" else project_root / "logs"
    log_root = Path(os.environ.get("KLJP_LOG_ROOT", str(default_log_root))).expanduser()
    default_electra_path = (
        SERVER_ELECTRA_PATH if mode == "server" else LOCAL_ELECTRA_PATH
    )
    electra_path = Path(
        os.environ.get("KLJP_ELECTRA_PATH", str(default_electra_path))
    ).expanduser()
    embedding_path = Path(
        os.environ.get(
            "KLJP_EMBEDDING_PATH",
            str(
                (data_root if mode == "server" else project_root)
                / "gensim_train"
                / "word2vec.model"
            ),
        )
    ).expanduser()

    return RuntimeConfig(
        mode=mode,
        reason=reason,
        project_root=project_root,
        data_root=data_root,
        log_root=log_root,
        electra_path=electra_path,
        embedding_path=embedding_path,
        allow_model_download=(
            False if mode == "server" else _env_flag("KLJP_ALLOW_MODEL_DOWNLOAD", True)
        ),
    )
