"""Hierarchical reasoning + neuro-symbolic financial QA and retrieval.

Public symbols are loaded lazily so page indexing/OCR utilities do not import
Torch and the HRM checkpoint stack merely by importing this package.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "DenseEmbeddingIndex",
    "DocumentEvidenceRetriever",
    "DocumentHRMBundle",
    "DocumentHierarchicalReasoningModel",
    "DocumentPageIndex",
    "PageIndexStats",
    "DocumentHRMReasoningEngine",
    "FinancialQAEngine",
    "FinancialRepository",
    "HierarchicalReasoningBundle",
    "HierarchicalReasoningModel",
    "HierarchicalReasoningQAEngine",
    "HybridFinancialDocumentIndex",
    "NeuralModelBundle",
    "NeuroSymbolicHRMEngine",
    "OCRConfig",
    "ProgramRerankerBundle",
    "VietnameseFinancialDocumentIndex",
]


_EXPORTS = {
    "DenseEmbeddingIndex": (".dense_retrieval", "DenseEmbeddingIndex"),
    "HybridFinancialDocumentIndex": (
        ".dense_retrieval",
        "HybridFinancialDocumentIndex",
    ),
    "DocumentEvidenceRetriever": (".document_agent", "DocumentEvidenceRetriever"),
    "DocumentHRMBundle": (".document_hrm", "DocumentHRMBundle"),
    "DocumentHierarchicalReasoningModel": (
        ".document_hrm",
        "DocumentHierarchicalReasoningModel",
    ),
    "DocumentPageIndex": (".document_page_index", "DocumentPageIndex"),
    "PageIndexStats": (".document_page_index", "PageIndexStats"),
    "DocumentHRMReasoningEngine": (
        ".document_reasoning_engine",
        "DocumentHRMReasoningEngine",
    ),
    "VietnameseFinancialDocumentIndex": (
        ".document_search",
        "VietnameseFinancialDocumentIndex",
    ),
    "OCRConfig": (".document_sources", "OCRConfig"),
    "ProgramRerankerBundle": (
        ".program_reranker",
        "ProgramRerankerBundle",
    ),
    "FinancialQAEngine": (".engine", "FinancialQAEngine"),
    "NeuroSymbolicHRMEngine": (".hrm", "NeuroSymbolicHRMEngine"),
    "NeuralModelBundle": (".neural", "NeuralModelBundle"),
    "HierarchicalReasoningBundle": (
        ".reasoning",
        "HierarchicalReasoningBundle",
    ),
    "HierarchicalReasoningModel": (
        ".reasoning",
        "HierarchicalReasoningModel",
    ),
    "HierarchicalReasoningQAEngine": (
        ".reasoning_engine",
        "HierarchicalReasoningQAEngine",
    ),
    "FinancialRepository": (".repository", "FinancialRepository"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
