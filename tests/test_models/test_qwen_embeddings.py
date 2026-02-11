"""
Tests for LocalQwenEmbeddings (Qwen3-VL-Embedding-2B LangChain wrapper).

Unit tests mock Qwen3VLEmbedder so the real model is not required.
"""

import pytest
from unittest.mock import MagicMock, patch

import torch

from ai_toolkit.models import LocalQwenEmbeddings


def _make_mock_embedder(process_return_value):
    """Build a mock qwen3_vl_embedding module so LocalQwenEmbeddings can be instantiated."""
    mock_process = MagicMock(return_value=process_return_value)
    mock_instance = MagicMock()
    mock_instance.process = mock_process
    mock_class = MagicMock(return_value=mock_instance)
    mock_module = MagicMock()
    mock_module.Qwen3VLEmbedder = mock_class
    return mock_module, mock_instance


class TestLocalQwenEmbeddingsImport:
    """Test that LocalQwenEmbeddings is importable from ai_toolkit.models."""

    def test_import_from_models(self):
        """LocalQwenEmbeddings can be imported without instantiating."""
        from ai_toolkit.models import LocalQwenEmbeddings as Emb
        assert Emb is LocalQwenEmbeddings


class TestLocalQwenEmbeddingsWithMock:
    """Test LocalQwenEmbeddings behavior with mocked Qwen3VLEmbedder."""

    def test_embed_documents_empty_returns_empty_list(self):
        """embed_documents([]) returns [] without calling the embedder."""
        mock_module, mock_instance = _make_mock_embedder(torch.randn(0, 8))
        with patch.dict("sys.modules", {"qwen3_vl_embedding": mock_module}):
            emb = LocalQwenEmbeddings("/fake/path")
        result = emb.embed_documents([])
        assert result == []
        mock_instance.process.assert_not_called()

    def test_embed_documents_returns_list_of_float_lists(self):
        """embed_documents with mock returns list of list of floats, correct length."""
        # Two documents, embedding dim 8
        mock_out = torch.randn(2, 8).to(torch.float32)
        mock_module, mock_instance = _make_mock_embedder(mock_out)
        with patch.dict("sys.modules", {"qwen3_vl_embedding": mock_module}):
            emb = LocalQwenEmbeddings("/fake/path")
        result = emb.embed_documents(["doc1", "doc2"])
        assert len(result) == 2
        assert all(len(vec) == 8 for vec in result)
        assert all(isinstance(x, float) for vec in result for x in vec)
        mock_instance.process.assert_called_once_with(
            [{"text": "doc1"}, {"text": "doc2"}], normalize=True
        )

    def test_embed_documents_converts_bfloat16_to_float32(self):
        """Output is converted from bfloat16 to Python floats (NumPy-safe)."""
        try:
            mock_out = torch.randn(2, 8).to(torch.bfloat16)
        except (RuntimeError, TypeError):
            pytest.skip("bfloat16 not available on this build")
        mock_module, mock_instance = _make_mock_embedder(mock_out)
        with patch.dict("sys.modules", {"qwen3_vl_embedding": mock_module}):
            emb = LocalQwenEmbeddings("/fake/path")
        result = emb.embed_documents(["a", "b"])
        assert len(result) == 2
        assert all(isinstance(x, float) for vec in result for x in vec)

    def test_embed_documents_single_doc_ndim1(self):
        """Single document can return 1d tensor from process; still returns list of one vector."""
        mock_out = torch.randn(4).to(torch.float32)  # 1d
        mock_module, mock_instance = _make_mock_embedder(mock_out)
        with patch.dict("sys.modules", {"qwen3_vl_embedding": mock_module}):
            emb = LocalQwenEmbeddings("/fake/path")
        result = emb.embed_documents(["only one"])
        assert len(result) == 1
        assert len(result[0]) == 4
        assert all(isinstance(x, float) for x in result[0])

    def test_embed_query_returns_list_of_floats(self):
        """embed_query returns a single list of floats."""
        mock_out = torch.randn(1, 8).to(torch.float32)
        mock_module, mock_instance = _make_mock_embedder(mock_out)
        with patch.dict("sys.modules", {"qwen3_vl_embedding": mock_module}):
            emb = LocalQwenEmbeddings("/fake/path")
        result = emb.embed_query("query text")
        assert isinstance(result, list)
        assert len(result) == 8
        assert all(isinstance(x, float) for x in result)
        mock_instance.process.assert_called_once_with(
            [{"text": "query text"}], normalize=True
        )

    def test_embed_query_handles_2d_output(self):
        """embed_query when process returns 2d tensor uses first row."""
        mock_out = torch.randn(1, 5).to(torch.float32)
        mock_module, mock_instance = _make_mock_embedder(mock_out)
        with patch.dict("sys.modules", {"qwen3_vl_embedding": mock_module}):
            emb = LocalQwenEmbeddings("/fake/path")
        result = emb.embed_query("q")
        assert len(result) == 5
        assert result == mock_out[0].tolist()

    def test_embed_query_handles_1d_output(self):
        """embed_query when process returns 1d tensor returns that as list."""
        mock_out = torch.randn(5).to(torch.float32)
        mock_module, mock_instance = _make_mock_embedder(mock_out)
        with patch.dict("sys.modules", {"qwen3_vl_embedding": mock_module}):
            emb = LocalQwenEmbeddings("/fake/path")
        result = emb.embed_query("q")
        assert len(result) == 5
        assert result == mock_out.tolist()
