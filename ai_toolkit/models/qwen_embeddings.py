"""
LangChain Embeddings wrapper for local Qwen3-VL-Embedding-2B (text-only).

Callers must ensure the Qwen3-VL-Embedding model scripts are importable before
instantiating this class (e.g. add models/Qwen3-VL-Embedding-2B/scripts to
sys.path). On PyTorch < 2.3 with transformers >= 4.57, apply the
torch.is_autocast_enabled compatibility patch before first use.

Usage:
    >>> import sys
    >>> from pathlib import Path
    >>> project_root = Path(__file__).resolve().parents[2]
    >>> sys.path.insert(0, str(project_root / "models/Qwen3-VL-Embedding-2B/scripts"))
    >>> from ai_toolkit.models import LocalQwenEmbeddings
    >>> embeddings = LocalQwenEmbeddings("/path/to/Qwen3-VL-Embedding-2B")
    >>> vectors = embeddings.embed_documents(["text1", "text2"])
"""

from typing import List

import torch
from langchain_core.embeddings import Embeddings


class LocalQwenEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for local Qwen3-VL-Embedding-2B (text-only)."""

    def __init__(self, model_path: str, max_length: int = 8192):
        """
        Initialize the embedder with a local Qwen3-VL-Embedding-2B model.

        Args:
            model_path: Path to the model directory (e.g. .../Qwen3-VL-Embedding-2B).
            max_length: Maximum sequence length for the embedder (default 8192).
        """
        # Lazy import: Qwen3VLEmbedder lives in the model repo scripts, not in ai-toolkit.
        # Caller must add that scripts directory to sys.path before instantiating.
        from qwen3_vl_embedding import Qwen3VLEmbedder

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            max_length=max_length,
        )
        print(f"Loaded local Qwen3-VL-Embedding-2B from {model_path}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        inputs = [{"text": t} for t in texts]
        out = self.embedder.process(inputs, normalize=True)
        # Qwen3-VL may return bfloat16 tensors; convert to float32 for NumPy/langchain.
        out = out.to(dtype=torch.float32)
        arr = out.cpu().numpy()
        if arr.ndim == 1:
            return [arr.tolist()]
        return [arr[i].tolist() for i in range(len(arr))]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        inputs = [{"text": text}]
        out = self.embedder.process(inputs, normalize=True)
        # Qwen3-VL may return bfloat16 tensors; convert to float32 for NumPy/langchain.
        out = out.to(dtype=torch.float32)
        arr = out.cpu().numpy()
        if arr.ndim == 2:
            return arr[0].tolist()
        return arr.tolist()
