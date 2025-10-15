from typing import Any, Dict, Tuple, Optional

from pykeen.models import TransE, TransR, HolE, RotatE

# Analogy is not available in all PyKEEN versions. Try to import if present.
try:
    from pykeen.models import Analogy  # type: ignore
    ANALOGY_CLS: Optional[Any] = Analogy
except Exception:  # noqa: BLE001
    ANALOGY_CLS = None


SUPPORTED_MODELS: Dict[str, Any] = {
    "TransE": TransE,
    "TransR": TransR,
    "HolE": HolE,
    "RotatE": RotatE,
}

if ANALOGY_CLS is not None:
    SUPPORTED_MODELS["Analogy"] = ANALOGY_CLS


def build_model(model_name: str, embedding_dim: int) -> Tuple[Any, Dict[str, Any]]:
    """Return model class and default kwargs for a uniform embedding_dim.

    Returns (model_cls, model_kwargs)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. Installed PyKEEN does not provide this model."
        )

    model_cls = SUPPORTED_MODELS[model_name]

    # Common config across models where applicable
    kwargs: Dict[str, Any] = {}

    # Some models use different kwargs names; PyKEEN standardizes via embedding_dim or entity_embedding_dim
    if model_name in {"TransE", "Analogy", "HolE", "RotatE"}:
        kwargs["embedding_dim"] = embedding_dim
    elif model_name == "TransR":
        kwargs["embedding_dim"] = embedding_dim
        kwargs["relation_dim"] = embedding_dim

    return model_cls, kwargs


