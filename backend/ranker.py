"""
ranker.py — Orchestrates the full candidate ranking pipeline.

Pipeline:
  1. Parse JD → ParsedJD
  2. Generate JD embedding
  3. Stream candidates in chunks from JSONL
  4. For each chunk:
      a. Build candidate embedding texts
      b. Batch-encode with SentenceTransformer
      c. Add embeddings to FAISS index
      d. Compute all sub-scores
      e. Accumulate ScoreBreakdown results
  5. Sort all results by final_score (desc), then candidate_id (asc) on tie
  6. Return top-100 ranked results

Progress is reported via an optional async callback for SSE streaming.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import numpy as np
from loguru import logger

import gc

from candidate_parser import CandidateRecord, CandidateParser
from csv_export import generate_reasoning, export_csv
from embeddings import EmbeddingEngine, FAISSIndex
from jd_parser import ParsedJD, parse_job_description
from scoring import ScoreBreakdown, ScoringEngine
from utils import ensure_output_dir, load_config, log_memory


# ── Progress Model ────────────────────────────────────────────────────────────

@dataclass
class RankingProgress:
    """Real-time progress information emitted during ranking."""

    stage: str                           # e.g. "parsing_jd", "embedding", "scoring"
    candidates_processed: int = 0
    total_candidates: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    memory_mb: float = 0.0
    message: str = ""

    @property
    def percent(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return min(100.0, self.candidates_processed / self.total_candidates * 100.0)


@dataclass
class RankingResult:
    """Final result returned after ranking completes."""

    ranked_candidates: list[ScoreBreakdown]
    jd: ParsedJD
    total_candidates_evaluated: int
    elapsed_seconds: float
    csv_path: str
    top_k: int = 100


# ── Ranker ────────────────────────────────────────────────────────────────────

class CandidateRanker:
    """
    Orchestrates the end-to-end candidate ranking pipeline.

    Designed for:
      - Memory efficiency: streams candidates in chunks, never holds all in RAM.
      - CPU performance: vectorized numpy + batched SentenceTransformer encoding.
      - Observability: emits progress callbacks at each chunk boundary.
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._embedding_engine = EmbeddingEngine()
        self._scoring_engine = ScoringEngine()
        self._parser = CandidateParser()
        self._logger = logger.bind(module="CandidateRanker")

    # ── Sync API (for CLI / scripts) ──────────────────────────────────────────

    def rank(
        self,
        jd_path: str | Path,
        candidates_path: str | Path,
        progress_callback: Optional[Callable[[RankingProgress], None]] = None,
    ) -> RankingResult:
        """
        Run the full ranking pipeline synchronously.

        Args:
            jd_path: Path to job_description.docx.
            candidates_path: Path to candidates.jsonl.
            progress_callback: Optional callable receiving RankingProgress updates.

        Returns:
            RankingResult with sorted top-K candidates.
        """
        start_time = time.perf_counter()

        def _emit(stage: str, processed: int, total: int, msg: str = "") -> None:
            elapsed = time.perf_counter() - start_time
            speed = processed / elapsed if elapsed > 0 else 1
            remaining = (total - processed) / speed if speed > 0 and total > 0 else 0
            prog = RankingProgress(
                stage=stage,
                candidates_processed=processed,
                total_candidates=total,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=remaining,
                memory_mb=_get_mem(),
                message=msg,
            )
            self._logger.info(f"[{stage}] {processed}/{total} ({prog.percent:.1f}%) — {msg}")
            if progress_callback:
                progress_callback(prog)

        # ── Step 1: Parse JD ─────────────────────────────────────────────────
        _emit("parsing_jd", 0, 1, "Parsing job description...")
        jd = parse_job_description(jd_path)
        _emit("parsing_jd", 1, 1, f"JD parsed: {len(jd.required_skills)} required skills")

        # ── Step 2: Embed JD ─────────────────────────────────────────────────
        _emit("embedding_jd", 0, 1, "Generating JD embedding...")
        jd_embedding = self._embedding_engine.encode_single(jd.embedding_text)
        _emit("embedding_jd", 1, 1, f"JD embedding shape: {jd_embedding.shape}")

        # ── Step 3: Count total candidates (for progress) ────────────────────
        _emit("counting", 0, 0, "Counting candidates...")
        total = self._parser.count_candidates(candidates_path)
        _emit("counting", total, total, f"Total candidates: {total:,}")

        # ── Step 4: Stream, embed, score in chunks ───────────────────────────
        chunk_size = self._cfg["performance"]["candidate_chunk_size"]
        all_scores: list[ScoreBreakdown] = []
        faiss_index = FAISSIndex()
        processed = 0
        candidate_index: list[CandidateRecord] = []  # keep in-chunk memory only

        for chunk in self._parser.stream(candidates_path, chunk_size):
            _emit("embedding", processed, total, f"Embedding chunk ({len(chunk)} candidates)...")

            # Build texts and encode
            texts = [c.embedding_text for c in chunk]
            chunk_embeddings = self._embedding_engine.encode_batch(texts)

            # Compute cosine similarity with JD embedding (vectorized)
            semantic_scores = self._embedding_engine.cosine_similarity_batch(
                jd_embedding, chunk_embeddings
            )

            # Score all signals
            _emit("scoring", processed, total, f"Scoring {len(chunk)} candidates...")
            breakdowns = self._scoring_engine.score_batch(chunk, jd, semantic_scores)
            all_scores.extend(breakdowns)

            processed += len(chunk)
            _emit("scoring", processed, total, f"Scored {processed:,}/{total:,}")
            log_memory(f"chunk-{processed}")

        # ── Step 5: Sort and select top-K ────────────────────────────────────
        _emit("sorting", total, total, "Sorting results...")
        top_k = self._cfg["performance"]["top_k"]

        # Sort: descending score, then ascending candidate_id on tie
        all_scores.sort(key=lambda x: (-x.final_score, x.candidate_id))
        top_results = all_scores[:top_k]

        # ── Step 6: Export CSV ───────────────────────────────────────────────
        _emit("exporting", total, total, "Exporting CSV...")
        out_dir = ensure_output_dir()
        
        # Build lookup for top-K candidates to get profile data for recruiter CSV
        top_ids = {bd.candidate_id for bd in top_results}
        top_lookup = {}
        try:
            import json
            with Path(candidates_path).open("r", encoding="utf-8") as fh:
                first_char = ""
                for line in fh:
                    line_strip = line.strip()
                    if line_strip:
                        first_char = line_strip[0]
                        break
                fh.seek(0)
                if first_char == "[":
                    data = json.load(fh)
                    if isinstance(data, list):
                        for item in data:
                            cid = item.get("candidate_id")
                            if cid in top_ids:
                                top_lookup[cid] = item
                else:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            cid = item.get("candidate_id")
                            if cid in top_ids:
                                top_lookup[cid] = item
                        except Exception:
                            continue
        except Exception as exc:
            logger.warning(f"Could not build top candidate details lookup: {exc}")
            
        csv_path = export_csv(top_results, out_dir, candidate_lookup=top_lookup)
        _emit("done", total, total, f"Export complete → {csv_path}")

        elapsed = time.perf_counter() - start_time
        self._logger.info(
            f"Ranking complete: {total:,} candidates in {elapsed:.1f}s, "
            f"top-{len(top_results)} exported to {csv_path}"
        )

        return RankingResult(
            ranked_candidates=top_results,
            jd=jd,
            total_candidates_evaluated=total,
            elapsed_seconds=elapsed,
            csv_path=str(csv_path),
            top_k=top_k,
        )

    # ── Async API (for FastAPI SSE) ───────────────────────────────────────────

    async def rank_async(
        self,
        jd_path: str | Path,
        candidates_path: str | Path,
    ) -> AsyncGenerator[RankingProgress | RankingResult, None]:
        """
        Async generator that yields RankingProgress objects during ranking,
        followed by the final RankingResult.

        Runs the CPU-bound pipeline in a thread pool to avoid blocking the
        FastAPI event loop.
        """
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        result_holder: list[RankingResult] = []

        def _callback(prog: RankingProgress) -> None:
            loop.call_soon_threadsafe(progress_queue.put_nowait, prog)

        async def _run_in_thread():
            result = await loop.run_in_executor(
                None,
                lambda: self.rank(jd_path, candidates_path, _callback),
            )
            result_holder.append(result)
            await progress_queue.put(None)  # sentinel

        task = asyncio.create_task(_run_in_thread())

        while True:
            item = await progress_queue.get()
            if item is None:
                break
            yield item

        await task

        if result_holder:
            yield result_holder[0]


# ── Convenience Function ─────────────────────────────────────────────────────

def run_ranking(
    jd_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
    progress_callback: Optional[Callable[[RankingProgress], None]] = None,
) -> RankingResult:
    """
    Convenience function to run the ranking pipeline with config defaults.

    Args:
        jd_path: Override JD path (defaults to config value).
        candidates_path: Override candidates path (defaults to config value).
        progress_callback: Optional progress update callback.

    Returns:
        RankingResult with top-K scored candidates.
    """
    cfg = load_config()

    jd = Path(jd_path) if jd_path else (
        Path(__file__).parent / cfg["data"]["job_description_docx"]
    )
    candidates = Path(candidates_path) if candidates_path else (
        Path(__file__).parent / cfg["data"]["candidates_jsonl"]
    )

    return CandidateRanker().rank(jd, candidates, progress_callback)


def _get_mem() -> float:
    """Return current process memory in MB."""
    import os
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
