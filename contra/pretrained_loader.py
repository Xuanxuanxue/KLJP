"""Pretrained model/tokenizer loading with a local Electra fallback."""

from pathlib import Path

from setting import ELECTRA_MODEL_NAME, LOCAL_ELECTRA_MODEL_PATH


def electra_sources():
    """Return local-first then Hugging Face sources for the Electra model."""
    local_path = Path(LOCAL_ELECTRA_MODEL_PATH)
    if local_path.is_dir():
        yield str(local_path)
    yield ELECTRA_MODEL_NAME


def get_electra_source():
    """Return the source that will normally be selected for logging/config."""
    return next(electra_sources())


def _load_with_fallback(loader, component_name):
    errors = []
    for source in electra_sources():
        source_is_local = Path(source).is_dir()
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
        f"{component_name.lower()}. Tried these sources:\n{detail}"
    )


def load_electra_model(loader):
    return _load_with_fallback(loader, "model")


def load_electra_tokenizer(loader):
    return _load_with_fallback(loader, "tokenizer")
