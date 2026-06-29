"""
embeddings.py — Sentence embedding and FAISS similarity search.

Responsibilities:
  - Load all-MiniLM-L6-v2 once (singleton pattern).
  - Batch-encode texts efficiently on CPU.
  - Build a FAISS index for fast cosine similarity search.
  - Provide cosine similarity computation between JD and candidates.
"""

import os
os.environ["USE_TF"] = "OFF"
os.environ["USE_KERAS"] = "OFF"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from pathlib import Path
from loguru import logger

from utils import load_config, log_memory, timed

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed; falling back to numpy cosine similarity.")


# ── Singleton Model ───────────────────────────────────────────────────────────

class ONNXSentenceTransformer:
    def __init__(self, model_dir: str) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        logger.info(f"Initializing ONNXSentenceTransformer from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        
        # Locate the ONNX file
        onnx_dir = Path(model_dir) / "onnx"
        onnx_path = None
        if onnx_dir.exists():
            for name in ["model_qint8_avx512.onnx", "model_qint8_avx512_vnni.onnx", "model_qint8_arm64.onnx"]:
                if (onnx_dir / name).exists():
                    onnx_path = str((onnx_dir / name).resolve())
                    break
            if not onnx_path:
                onnx_files = list(onnx_dir.glob("*.onnx"))
                if onnx_files:
                    onnx_path = str(onnx_files[0].resolve())
                    
        if not onnx_path:
            raise FileNotFoundError(f"No ONNX model file found in {onnx_dir}")
            
        logger.info(f"Loading ONNX session for: {onnx_path}")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.dimension = 384

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=256,
                return_tensors="np"
            )
            onnx_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64)
            }
            if "token_type_ids" in encoded:
                onnx_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
                
            outputs = self.session.run(None, onnx_inputs)
            token_embeddings = outputs[0]
            
            attention_mask = encoded["attention_mask"]
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask
            
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-9)
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings).astype(np.float32)


_model_instance: ONNXSentenceTransformer | None = None


def get_model() -> ONNXSentenceTransformer:
    global _model_instance
    if _model_instance is None:
        cfg = load_config()
        cache_dir = cfg["model"]["cache_dir"]
        
        # Locate the snapshot directory recursively
        path = Path(cache_dir)
        if not path.exists() and Path(f"../{cache_dir}").exists():
            path = Path(f"../{cache_dir}")
            
        snapshots_pattern = "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        snapshots_dir = path / snapshots_pattern
        
        model_dir = None
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                model_dir = str(snapshots[0].resolve())
                
        if not model_dir:
            model_dir = str(path.resolve())
            
        logger.info(f"Loading ONNX model from localized cache: {model_dir}")
        _model_instance = ONNXSentenceTransformer(model_dir)
        log_memory("after-model-load")
    return _model_instance


# ── Embedding Engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps SentenceTransformer with batched encoding and FAISS indexing.

    Usage:
        engine = EmbeddingEngine()
        jd_embedding = engine.encode_single(jd_text)
        candidate_embeddings = engine.encode_batch(candidate_texts)
        similarities = engine.cosine_similarity_batch(jd_embedding, candidate_embeddings)
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._model = get_model()
        self._batch_size: int = self._cfg["performance"]["embedding_batch_size"]
        self._normalize: bool = self._cfg["model"]["normalize_embeddings"]
        self._logger = logger.bind(module="EmbeddingEngine")

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string to a normalized embedding vector.

        Args:
            text: Input text string.

        Returns:
            1-D float32 numpy array of shape (dim,).
        """
        if not text:
            dim = self._model.get_sentence_embedding_dimension()
            return np.zeros(dim, dtype=np.float32)

        embedding = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embedding[0].astype(np.float32)

    @timed("batch-encode")
    def encode_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Batch-encode a list of texts into a 2-D embedding matrix with optimized batching.

        Args:
            texts: List of input strings.
            show_progress: Show tqdm progress bar during encoding.

        Returns:
            2-D float32 array of shape (n, embedding_dim).
        """
        if not texts:
            dim = self._model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        # Replace empty strings with a placeholder to avoid model errors
        sanitized = [t if t else " " for t in texts]

        # Use dynamic batch sizing for better performance
        effective_batch_size = min(self._batch_size, 512, len(sanitized))

        self._logger.debug(
            f"Encoding {len(sanitized)} texts "
            f"(batch_size={effective_batch_size}, normalize={self._normalize})"
        )

        embeddings = self._model.encode(
            sanitized,
            batch_size=effective_batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def cosine_similarity_batch(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between one query vector and a corpus matrix.

        If embeddings are already L2-normalized (normalize_embeddings=True),
        cosine similarity = dot product.

        Args:
            query: 1-D array of shape (dim,).
            corpus: 2-D array of shape (n, dim).

        Returns:
            1-D float32 array of similarity scores in [−1, 1], shape (n,).
        """
        if corpus.shape[0] == 0:
            return np.array([], dtype=np.float32)

        if self._normalize:
            # Fast path: normalized vectors → dot product = cosine similarity
            sims = corpus @ query
        else:
            # General cosine similarity
            query_norm = np.linalg.norm(query)
            corpus_norms = np.linalg.norm(corpus, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = np.where(
                    (query_norm * corpus_norms) > 0,
                    (corpus @ query) / (query_norm * corpus_norms),
                    0.0,
                )

        return sims.astype(np.float32)


# ── FAISS Index ───────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    FAISS-backed similarity index for large-scale candidate search.

    Supports:
      - IndexFlatIP (exact inner product = cosine with normalized vectors)
      - Add embeddings incrementally (chunk-by-chunk)
      - Retrieve top-K candidates by similarity score
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._index: faiss.Index | None = None
        self._candidate_ids: list[str] = []
        self._dim: int = 0
        self._logger = logger.bind(module="FAISSIndex")

    def build(self, embeddings: np.ndarray, candidate_ids: list[str]) -> None:
        """
        Build the FAISS index from a full embedding matrix.

        Args:
            embeddings: 2-D float32 array of shape (n, dim).
            candidate_ids: List of candidate IDs corresponding to each row.
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is not installed.")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build FAISS index from empty embeddings.")

        n, dim = embeddings.shape
        self._dim = dim
        self._candidate_ids = list(candidate_ids)
        self._logger.info(f"Building FAISS IndexFlatIP: {n} vectors, dim={dim}")

        self._index = faiss.IndexFlatIP(dim)
        # FAISS requires contiguous C-order float32
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._logger.info(f"FAISS index built with {self._index.ntotal} vectors.")

    def add_batch(self, embeddings: np.ndarray, candidate_ids: list[str]) -> None:
        """
        Incrementally add a batch of embeddings to an existing index.

        Args:
            embeddings: 2-D float32 array of shape (batch, dim).
            candidate_ids: IDs for this batch.
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is not installed.")

        n, dim = embeddings.shape
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)
            self._logger.info(f"Created FAISS IndexFlatIP (dim={dim})")

        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._candidate_ids.extend(candidate_ids)
        self._logger.debug(
            f"Added {n} vectors; total={self._index.ntotal}"
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-K most similar candidates to the query vector.

        Args:
            query: 1-D float32 embedding of the JD.
            top_k: Number of results to return. Defaults to config value.

        Returns:
            List of (candidate_id, similarity_score) sorted descending.
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is not installed.")
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k or self._cfg["performance"]["top_k"], self._index.ntotal)
        query_2d = query.reshape(1, -1).astype(np.float32)

        scores, indices = self._index.search(query_2d, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._candidate_ids[idx], float(score)))

        return results

    @property
    def total_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0
