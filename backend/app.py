"""
app.py — FastAPI application for the Redrob AI Candidate Ranking System.

Endpoints:
  POST /api/upload-jd          — Upload & parse a .docx job description
  POST /api/start-ranking      — Kick off the async ranking pipeline
  GET  /api/ranking-status     — SSE stream for live progress updates
  GET  /api/results            — Paginated ranked results with filtering
  GET  /api/results/top100     — Full top-100 list
  GET  /api/download-csv       — Download the submission CSV
  GET  /api/analytics          — Score distribution and breakdown statistics
  GET  /api/jd-info            — Current parsed JD details
  GET  /api/system-info        — Architecture and system metadata
  GET  /health                 — Health check
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import psutil
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from csv_export import export_csv
from jd_parser import ParsedJD, parse_job_description
from ranker import CandidateRanker, RankingProgress, RankingResult
from scoring import ScoreBreakdown, ScoringEngine
from candidate_parser import CandidateParser
from embeddings import EmbeddingEngine
from utils import ensure_output_dir, get_memory_usage_mb, load_config, setup_logging
import numpy as np


# ── Application State ────────────────────────────────────────────────────────

class AppState:
    """Mutable application state shared across requests."""

    def __init__(self) -> None:
        self.jd: ParsedJD | None = None
        self.jd_path: Path | None = None
        self.ranking_result: RankingResult | None = None
        self.ranking_in_progress: bool = False
        self.progress_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.ranking_task: asyncio.Task | None = None


_state = AppState()
_cfg = load_config()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    setup_logging(_cfg["api"].get("log_level", "info"))
    logger.info("Starting Redrob Ranking API...")
    ensure_output_dir()

    # Pre-warm the embedding model on startup (avoids cold start on first request)
    try:
        from embeddings import get_model
        logger.info("Pre-warming SentenceTransformer model...")
        get_model()
        logger.info("Model ready.")
    except Exception as exc:
        logger.warning(f"Model pre-warm failed (will load on first request): {exc}")

    # Load previously saved top100 results if available
    try:
        out_dir = ensure_output_dir()
        json_path = out_dir / "top100.json"
        if json_path.exists():
            logger.info(f"Loading pre-computed ranking results from: {json_path}")
            with json_path.open("r", encoding="utf-8") as jf:
                data = json.load(jf)
            
            # Reconstruct ScoreBreakdowns
            reconstructed = []
            for item in data:
                bd = ScoreBreakdown(
                    candidate_id=item["candidate_id"],
                    semantic_similarity=item.get("semantic_similarity", 0.0),
                    skill_match=item.get("skill_match", 0.0),
                    experience_match=item.get("experience_match", 0.0),
                    behavior_score=item.get("behavior_score", 0.0),
                    location_bonus=item.get("location_bonus", 0.0),
                    education_score=item.get("education_score", 0.0),
                    final_score=item.get("final_score", 0.0),
                    reasoning=item.get("reasoning", ""),
                    recommendation_tier=item.get("recommendation_tier", "Consider"),
                    confidence_level=item.get("confidence_level", "Medium"),
                )
                reconstructed.append(bd)
            
            # Populate state
            _state.ranking_result = RankingResult(
                ranked_candidates=reconstructed,
                jd=ParsedJD(),  # Placeholder
                total_candidates_evaluated=100000,
                elapsed_seconds=0.0,
                csv_path=str(out_dir / "submission.csv"),
                top_k=100,
            )
            logger.info("Successfully loaded pre-computed ranking results into app state.")
    except Exception as exc:
        logger.warning(f"Could not load pre-computed ranking results: {exc}")

    yield

    logger.info("Shutting down Redrob Ranking API.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Redrob AI Candidate Ranking API",
    description=(
        "Production-ready intelligent candidate ranking engine for the "
        "Redrob India Runs Hackathon. Ranks candidates against a Job "
        "Description using local Sentence Transformers — no hosted LLM APIs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "memory_mb": round(get_memory_usage_mb(), 1),
        "jd_loaded": _state.jd is not None,
        "ranking_in_progress": _state.ranking_in_progress,
        "ranking_complete": _state.ranking_result is not None,
    }


# ── JD Upload ─────────────────────────────────────────────────────────────────

@app.post("/api/upload-jd", tags=["Pipeline"])
async def upload_jd(file: UploadFile = File(...)) -> dict:
    """
    Upload and parse a Job Description (.docx) file.

    The parsed fields (skills, experience range, location, etc.) are cached
    in application state and used during ranking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".docx", ".doc"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Please upload a .docx file.",
        )

    # Save uploaded file to output dir
    out_dir = ensure_output_dir()
    jd_path = out_dir / "uploaded_jd.docx"
    content = await file.read()
    jd_path.write_bytes(content)

    try:
        jd = parse_job_description(jd_path)
        _state.jd = jd
        _state.jd_path = jd_path
        logger.info(f"JD uploaded and parsed: {file.filename}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"JD parsing failed: {exc}") from exc

    return {
        "message": "Job description parsed successfully.",
        "filename": file.filename,
        "fields": jd.to_dict(),
    }


@app.get("/api/jd-info", tags=["Pipeline"])
async def get_jd_info() -> dict:
    """Return the currently parsed JD details."""
    if _state.jd is None:
        raise HTTPException(status_code=404, detail="No JD loaded. Upload one first.")
    return {
        "loaded": True,
        "fields": _state.jd.to_dict(),
        "embedding_text_preview": _state.jd.embedding_text[:300],
    }


# ── Ranking Pipeline ──────────────────────────────────────────────────────────

@app.post("/api/start-ranking", tags=["Pipeline"])
async def start_ranking(
    candidates_path: str | None = Query(
        default=None,
        description="Override path to candidates.jsonl. Defaults to config.",
    )
) -> dict:
    """
    Start the async ranking pipeline.

    Requires a JD to be uploaded first.
    Returns immediately; use /api/ranking-status for live progress.
    """
    if _state.jd is None:
        raise HTTPException(status_code=400, detail="No JD loaded. POST /api/upload-jd first.")

    if _state.ranking_in_progress:
        raise HTTPException(status_code=409, detail="Ranking is already in progress.")

    # Resolve paths
    cfg = load_config()
    jd_path = _state.jd_path or Path(cfg["data"]["job_description_docx"])
    cand_path = Path(candidates_path) if candidates_path else (
        Path(__file__).parent / cfg["data"]["candidates_jsonl"]
    )

    if not cand_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Candidates file not found: {cand_path}",
        )

    _state.ranking_in_progress = True
    _state.ranking_result = None
    # Clear old progress
    while not _state.progress_queue.empty():
        _state.progress_queue.get_nowait()

    async def _run():
        try:
            ranker = CandidateRanker()
            async for item in ranker.rank_async(jd_path, cand_path):
                if isinstance(item, RankingProgress):
                    try:
                        _state.progress_queue.put_nowait(item)
                    except asyncio.QueueFull:
                        pass
                elif isinstance(item, RankingResult):
                    _state.ranking_result = item
        except Exception as exc:
            logger.exception(f"Ranking pipeline error: {exc}")
            # Push error marker
            _state.progress_queue.put_nowait({"error": str(exc)})
        finally:
            _state.ranking_in_progress = False

    _state.ranking_task = asyncio.create_task(_run())

    return {"message": "Ranking started.", "stream_url": "/api/ranking-status"}


@app.get("/api/ranking-status", tags=["Pipeline"])
async def ranking_status() -> EventSourceResponse:
    """
    Server-Sent Events stream delivering RankingProgress updates.

    Connect with EventSource in the browser; each event is a JSON-encoded
    RankingProgress object. A final event with stage='done' signals completion.
    """

    async def _generator() -> AsyncGenerator[dict, None]:
        while _state.ranking_in_progress or not _state.progress_queue.empty():
            try:
                item = await asyncio.wait_for(_state.progress_queue.get(), timeout=1.0)
                if isinstance(item, dict) and "error" in item:
                    yield {"event": "error", "data": json.dumps(item)}
                    break
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "stage": item.stage,
                        "percent": round(item.percent, 1),
                        "candidates_processed": item.candidates_processed,
                        "total_candidates": item.total_candidates,
                        "elapsed_seconds": round(item.elapsed_seconds, 1),
                        "estimated_remaining_seconds": round(item.estimated_remaining_seconds, 1),
                        "memory_mb": round(item.memory_mb, 1),
                        "message": item.message,
                    }),
                }
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive
                yield {"event": "heartbeat", "data": "{}"}

        if _state.ranking_result is not None:
            yield {
                "event": "done",
                "data": json.dumps({
                    "stage": "done",
                    "total_evaluated": _state.ranking_result.total_candidates_evaluated,
                    "elapsed_seconds": round(_state.ranking_result.elapsed_seconds, 1),
                    "csv_path": _state.ranking_result.csv_path,
                }),
            }

    return EventSourceResponse(_generator())


# ── Results API ───────────────────────────────────────────────────────────────

@app.get("/api/results", tags=["Results"])
async def get_results(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    max_score: float = Query(default=1.0, ge=0.0, le=1.0),
    sort_by: str = Query(default="rank", pattern="^(rank|score|skill_match|semantic_similarity)$"),
) -> dict:
    """
    Return paginated and filtered ranked candidates.
    """
    if _state.ranking_result is None:
        raise HTTPException(status_code=404, detail="No ranking results available yet.")

    candidates = _state.ranking_result.ranked_candidates

    # Filter by score range
    filtered = [c for c in candidates if min_score <= c.final_score <= max_score]

    # Sort
    if sort_by == "skill_match":
        filtered.sort(key=lambda c: -c.skill_match)
    elif sort_by == "semantic_similarity":
        filtered.sort(key=lambda c: -c.semantic_similarity)
    else:
        # rank / score — already in order
        pass

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = filtered[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [c.to_dict() for c in page_data],
    }


@app.get("/api/results/top100", tags=["Results"])
async def get_top100() -> dict:
    """Return the full top-100 ranked candidates with all score details."""
    if _state.ranking_result is None:
        raise HTTPException(status_code=404, detail="No ranking results available yet.")

    top = _state.ranking_result.ranked_candidates[:100]
    return {
        "count": len(top),
        "total_evaluated": _state.ranking_result.total_candidates_evaluated,
        "elapsed_seconds": round(_state.ranking_result.elapsed_seconds, 1),
        "results": [
            {
                "rank": i + 1,
                **c.to_dict(),
            }
            for i, c in enumerate(top)
        ],
    }


# ── CSV Download ──────────────────────────────────────────────────────────────

@app.get("/api/download-csv", tags=["Results"])
async def download_csv() -> FileResponse:
    """Download the submission-ready CSV file."""
    out_dir = ensure_output_dir()
    csv_path = out_dir / "submission.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="CSV not generated yet. Run /api/start-ranking first.",
        )

    return FileResponse(
        path=str(csv_path),
        filename="submission.csv",
        media_type="text/csv",
    )


@app.get("/api/download-recruiter-csv", tags=["Results"])
async def download_recruiter_csv() -> FileResponse:
    """Download the detailed Recruiter Report CSV file."""
    out_dir = ensure_output_dir()
    csv_path = out_dir / "recruiter_report.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Recruiter Report CSV not generated yet. Run /api/start-ranking first.",
        )

    return FileResponse(
        path=str(csv_path),
        filename="recruiter_report.csv",
        media_type="text/csv",
    )


@app.get("/api/results/{candidate_id}", tags=["Results"])
async def get_candidate_details(candidate_id: str) -> dict:
    """
    Search candidates.jsonl line-by-line for candidate_id and return detailed profile
    along with its score breakdown and skill matching analyses.
    """
    cfg = load_config()
    
    # Check both backend folder relative and absolute config paths
    backend_dir = Path(__file__).parent
    candidates_raw_path = Path(cfg["data"]["candidates_jsonl"])
    if not candidates_raw_path.exists() and (backend_dir / candidates_raw_path).exists():
        cand_path = (backend_dir / candidates_raw_path).resolve()
    else:
        cand_path = candidates_raw_path.resolve()

    if not cand_path.exists():
        raise HTTPException(status_code=404, detail="Candidate database source file not found.")

    found_raw = None
    with cand_path.open("r", encoding="utf-8") as fh:
        first_char = ""
        for line in fh:
            line_strip = line.strip()
            if line_strip:
                first_char = line_strip[0]
                break
        fh.seek(0)
        
        if first_char == "[":
            try:
                import json
                data = json.load(fh)
                if isinstance(data, list):
                    for item in data:
                        if item.get("candidate_id") == candidate_id:
                            found_raw = item
                            break
            except Exception as e:
                logger.error(f"Error parsing candidate JSON array: {e}")
        else:
            import json
            for line in fh:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if item.get("candidate_id") == candidate_id:
                        found_raw = item
                        break
                except Exception:
                    continue

    if not found_raw:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found in database.")

    # Parse candidate record
    parser = CandidateParser()
    candidate_record = parser._parse_candidate(found_raw)
    if not candidate_record:
        raise HTTPException(status_code=500, detail="Failed to parse candidate record.")

    # Score candidate
    breakdown_dict = {}
    matched_skills = []
    missing_skills = []
    
    if _state.jd is not None:
        sem_score = 0.5
        if _state.ranking_result is not None:
            for c in _state.ranking_result.ranked_candidates:
                if c.candidate_id == candidate_id:
                    sem_score = c.semantic_similarity
                    break
            else:
                try:
                    engine = EmbeddingEngine()
                    jd_emb = engine.encode_single(_state.jd.embedding_text)
                    cand_emb = engine.encode_single(candidate_record.embedding_text)
                    sem_score = float(engine.cosine_similarity_batch(jd_emb, np.expand_dims(cand_emb, axis=0))[0])
                except Exception as exc:
                    logger.warning(f"Could not compute semantic similarity on fly: {exc}")
        
        scoring_engine = ScoringEngine()
        breakdowns = scoring_engine.score_batch([candidate_record], _state.jd, np.array([sem_score]))
        if breakdowns:
            bd = breakdowns[0]
            breakdown_dict = bd.to_dict()
            matched_skills = bd.matched_skills
            missing_skills = bd.missing_skills
    else:
        breakdown_dict = {
            "final_score": 0.5,
            "semantic_similarity": 0.5,
            "skill_match": 0.5,
            "experience_match": 0.5,
            "behavior_score": 0.5,
            "location_bonus": 0.5,
            "education_score": 0.5,
            "recommendation_tier": "Consider",
            "confidence_level": "Medium",
            "reasoning": "Job specification not uploaded yet."
        }

    # Strengths / Weaknesses
    strengths = []
    weaknesses = []
    
    expert_skills = [s.name for s in candidate_record.skills if s.proficiency in {"expert", "advanced"}]
    if expert_skills:
        strengths.append(f"Expertise in key tech: {', '.join(expert_skills[:3])}")
    if candidate_record.years_of_experience >= 5:
        strengths.append(f"Experienced professional with {candidate_record.years_of_experience:.1f} years of tenure")
    if candidate_record.signals and candidate_record.signals.github_activity_score >= 70:
        strengths.append("High GitHub active contribution score")
    if candidate_record.signals and candidate_record.signals.profile_completeness_score >= 90:
        strengths.append("Thoroughly documented professional profile")

    if not expert_skills:
        weaknesses.append("Lack of expert/advanced level skills in profile history")
    if candidate_record.years_of_experience < 2:
        weaknesses.append("Early career stage, might require additional mentorship")
    if candidate_record.signals and candidate_record.signals.profile_completeness_score < 60:
        weaknesses.append("Incomplete platform profile details")
    if missing_skills:
        weaknesses.append(f"Missing core JD required skills: {', '.join(missing_skills[:3])}")

    return {
        "candidate_id": candidate_id,
        "headline": candidate_record.headline,
        "summary": candidate_record.summary,
        "location": candidate_record.location,
        "country": candidate_record.country,
        "years_of_experience": candidate_record.years_of_experience,
        "current_title": candidate_record.current_title,
        "current_company": candidate_record.current_company,
        "current_company_size": candidate_record.current_company_size,
        "current_industry": candidate_record.current_industry,
        "skills": [
            {
                "name": s.name,
                "proficiency": s.proficiency,
                "endorsements": s.endorsements,
                "duration_months": s.duration_months
            }
            for s in candidate_record.skills
        ],
        "career_history": [
            {
                "company": c.company,
                "title": c.title,
                "duration_months": c.duration_months,
                "is_current": c.is_current,
                "industry": c.industry,
                "company_size": c.company_size,
                "description": c.description
            }
            for c in candidate_record.career_history
        ],
        "education": [
            {
                "institution": e.institution,
                "degree": e.degree,
                "field_of_study": e.field_of_study,
                "start_year": e.start_year,
                "end_year": e.end_year,
                "tier": e.tier
            }
            for e in candidate_record.education
        ],
        "certifications": [
            {
                "name": cert.name,
                "issuer": cert.issuer,
                "year": cert.year
            }
            for cert in candidate_record.certifications
        ],
        "signals": {
            "profile_completeness_score": candidate_record.signals.profile_completeness_score if candidate_record.signals else 0,
            "open_to_work_flag": candidate_record.signals.open_to_work_flag if candidate_record.signals else False,
            "recruiter_response_rate": candidate_record.signals.recruiter_response_rate if candidate_record.signals else 0,
            "avg_response_time_hours": candidate_record.signals.avg_response_time_hours if candidate_record.signals else 999,
            "github_activity_score": candidate_record.signals.github_activity_score if candidate_record.signals else -1,
            "connection_count": candidate_record.signals.connection_count if candidate_record.signals else 0,
            "willing_to_relocate": candidate_record.signals.willing_to_relocate if candidate_record.signals else False,
            "preferred_work_mode": candidate_record.signals.preferred_work_mode if candidate_record.signals else "on-site",
        },
        "score_breakdown": breakdown_dict,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "strengths": strengths,
        "weaknesses": weaknesses
    }


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/api/analytics", tags=["Analytics"])
async def get_analytics() -> dict:
    """
    Return aggregated analytics for dashboard charts.

    Includes:
      - Score distribution histogram (20 bins)
      - Average sub-score breakdown
      - Score component correlations
    """
    if _state.ranking_result is None:
        raise HTTPException(status_code=404, detail="No ranking results available yet.")

    candidates = _state.ranking_result.ranked_candidates
    if not candidates:
        return {"error": "No candidates scored."}

    import numpy as np  # local import to keep module-level clean

    scores = [c.final_score for c in candidates]
    sem = [c.semantic_similarity for c in candidates]
    skill = [c.skill_match for c in candidates]
    exp = [c.experience_match for c in candidates]
    beh = [c.behavior_score for c in candidates]
    loc = [c.location_bonus for c in candidates]
    edu = [c.education_score for c in candidates]

    # Histogram (20 bins from 0 to 1)
    hist, bin_edges = np.histogram(scores, bins=20, range=(0.0, 1.0))
    histogram = [
        {"bin": f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}", "count": int(hist[i])}
        for i in range(len(hist))
    ]

    # Average breakdown
    avg_breakdown = {
        "semantic_similarity": round(float(np.mean(sem)), 4),
        "skill_match": round(float(np.mean(skill)), 4),
        "experience_match": round(float(np.mean(exp)), 4),
        "behavior_score": round(float(np.mean(beh)), 4),
        "location_bonus": round(float(np.mean(loc)), 4),
        "education_score": round(float(np.mean(edu)), 4),
    }

    # Top-10 score distribution
    top10 = [
        {"rank": i + 1, "score": round(c.final_score, 4)}
        for i, c in enumerate(candidates[:10])
    ]

    return {
        "total_candidates": len(candidates),
        "top_score": round(max(scores), 4),
        "mean_score": round(float(np.mean(scores)), 4),
        "median_score": round(float(np.median(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "histogram": histogram,
        "avg_breakdown": avg_breakdown,
        "top10_scores": top10,
        "weights": _cfg["weights"],
    }


# ── System Info ───────────────────────────────────────────────────────────────

@app.get("/api/system-info", tags=["System"])
async def system_info() -> dict:
    """Return system architecture and runtime metadata."""
    cfg = load_config()
    return {
        "model": cfg["model"]["name"],
        "device": cfg["model"]["device"],
        "weights": cfg["weights"],
        "performance": cfg["performance"],
        "memory_mb": round(get_memory_usage_mb(), 1),
        "cpu_count": os.cpu_count(),
        "platform": {
            "python": f"{os.sys.version}",
        },
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "app:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=False,
        log_level=cfg["api"]["log_level"],
    )
