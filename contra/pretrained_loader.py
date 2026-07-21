"""Pretrained model/tokenizer loading with a local Electra fallback."""

from pathlib import Path

from setting import (ALLOW_MODEL_DOWNLOAD, ELECTRA_MODEL_NAME,
                     LOCAL_ELECTRA_MODEL_PATH, RUNTIME_MODE)


def electra_sources():
    """Return permitted Electra sources in priority order."""
    local_path = Path(LOCAL_ELECTRA_MODEL_PATH)
    yield str(local_path), True
    if ALLOW_MODEL_DOWNLOAD:
        yield ELECTRA_MODEL_NAME, False


def get_electra_source():
    """Return the source that will normally be selected for logging/config."""
    local_path = Path(LOCAL_ELECTRA_MODEL_PATH)
    if local_path.is_dir() or not ALLOW_MODEL_DOWNLOAD:
        return str(local_path)
    return ELECTRA_MODEL_NAME


def _load_with_fallback(loader, component_name):
    errors = []
    for source, source_is_local in electra_sources():
        if source_is_local and not Path(source).is_dir():
            errors.append(f"{source}: local directory does not exist")
            continue
        try:
            kwargs = {"local_files_only": True} if source_is_local else {}
            component = loader(source, **kwargs)
            print(f"{component_name} loaded from: {source}")
            return component
        except Exception as exc:  # transformers raises different errors per backend/version
            errors.append(f"{source}: {type(exc).__name__}: {exc}")

    detail = "\n".join(errors)
    raise OSError(
        "Unable to load the Electra "
        f"{component_name.lower()} in {RUNTIME_MODE} mode. "
        f"Model downloads are {'enabled' if ALLOW_MODEL_DOWNLOAD else 'disabled'}. "
        f"Tried these sources:\n{detail}"
    )


def load_electra_model(loader):
    return _load_with_fallback(loader, "model")


def load_electra_tokenizer(loader):
    return _load_with_fallback(loader, "tokenizer")
