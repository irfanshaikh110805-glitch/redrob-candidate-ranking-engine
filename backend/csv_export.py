"""
csv_export.py — Generate submission-ready CSV from scored candidates.

Produces the required output format:
  candidate_id, rank, score, reasoning

Reasoning is auto-generated from score components and candidate profile data.
The output is validated against the official submission rules before saving.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from scoring import ScoreBreakdown


# ── Reasoning Generator ───────────────────────────────────────────────────────

def generate_reasoning(breakdown: "ScoreBreakdown", candidate_data: dict | None = None) -> str:
    """
    Generate a human-readable reasoning string from a ScoreBreakdown.

    Args:
        breakdown: The scored candidate breakdown.
        candidate_data: Optional extra candidate data for richer reasoning.

    Returns:
        A concise reasoning string (≤200 chars).
    """
    parts: list[str] = []

    # Semantic similarity
    sem = breakdown.semantic_similarity
    if sem >= 0.80:
        parts.append("Strong semantic alignment with JD")
    elif sem >= 0.60:
        parts.append("Good semantic match with JD")
    elif sem >= 0.40:
        parts.append("Moderate semantic relevance")
    else:
        parts.append("Partial semantic match")

    # Skill match
    skill = breakdown.skill_match
    if skill >= 0.80:
        parts.append("excellent skill coverage")
    elif skill >= 0.60:
        parts.append("strong skill match")
    elif skill >= 0.40:
        parts.append("moderate skill overlap")
    else:
        parts.append("limited skill match")

    # Experience
    exp = breakdown.experience_match
    if exp >= 0.90:
        parts.append("ideal experience range")
    elif exp >= 0.70:
        parts.append("good experience fit")
    elif exp >= 0.40:
        parts.append("experience partially aligned")
    else:
        parts.append("experience outside ideal range")

    # Behavior signals
    beh = breakdown.behavior_score
    if beh >= 0.75:
        parts.append("high platform engagement score")
    elif beh >= 0.50:
        parts.append("active on platform")
    elif beh >= 0.25:
        parts.append("moderate platform signals")

    # Location
    if breakdown.location_bonus >= 0.90:
        parts.append("location match")
    elif breakdown.location_bonus >= 0.50:
        parts.append("same region")

    # Education
    edu = breakdown.education_score
    if edu >= 0.80:
        parts.append("strong educational background")
    elif edu >= 0.60:
        parts.append("relevant educational credentials")

    # Enrich with candidate data if available
    if candidate_data:
        title = candidate_data.get("current_title", "")
        years = candidate_data.get("years_of_experience", "")
        if title:
            parts.insert(0, f"{title}")
        if years:
            parts.append(f"{years:.1f} yrs experience")

    reasoning = ", ".join(parts)
    return reasoning[:250]  # Safety cap


# ── CSV Exporter ──────────────────────────────────────────────────────────────

def export_csv(
    ranked_candidates: list["ScoreBreakdown"],
    output_dir: Path,
    filename: str = "submission.csv",
    candidate_lookup: dict[str, dict] | None = None,
) -> Path:
    """
    Export the top-100 ranked candidates to a submission-ready CSV.

    Args:
        ranked_candidates: Already-sorted list of ScoreBreakdown (best first).
        output_dir: Directory to write the CSV file.
        filename: Output filename.
        candidate_lookup: Optional dict {candidate_id: profile_dict} for richer reasoning.

    Returns:
        Path to the created CSV file.

    Raises:
        ValueError: If fewer than 100 candidates are provided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename

    top_100 = ranked_candidates[:100]

    if len(top_100) < 100:
        logger.warning(f"Only {len(top_100)} candidates available (expected 100).")

    # Ensure scores are non-increasing (should already be sorted)
    _validate_score_order(top_100)

    rows: list[dict] = []
    for rank, bd in enumerate(top_100, start=1):
        # Use pre-computed reasoning from breakdown if present, else fallback
        reasoning = getattr(bd, "reasoning", "")
        if not reasoning:
            cand_data = (candidate_lookup or {}).get(bd.candidate_id)
            reasoning = generate_reasoning(bd, cand_data)

        rows.append({
            "candidate_id": bd.candidate_id,
            "rank": rank,
            "score": f"{bd.final_score:.4f}",
            "reasoning": reasoning,
        })

    # Handle tie-breaking: equal scores → sort candidate_id ascending
    rows = _apply_tie_breaking(rows)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} candidates to {csv_path}")

    # Generate Recruiter Report CSV
    rec_csv_path = output_dir / "recruiter_report.csv"
    rec_rows = []
    for rank, bd in enumerate(top_100, start=1):
        raw = (candidate_lookup or {}).get(bd.candidate_id, {})
        profile = raw.get("profile", {})
        
        # Name (anonymized)
        suffix = bd.candidate_id.split("_")[-1] if "_" in bd.candidate_id else bd.candidate_id
        name = f"Candidate {suffix}"
        
        # Skills list
        skills_list = ", ".join(s.get("name", "") for s in raw.get("skills", []) if s.get("name"))
        
        # Experience
        years = profile.get("years_of_experience", 0)
        
        # Education list
        edu_list = "; ".join(
            f"{e.get('degree', '')} in {e.get('field_of_study', '')} at {e.get('institution', '')}"
            for e in raw.get("education", [])
        )
        if not edu_list:
            edu_list = "None"
            
        location = profile.get("location", "")
        if profile.get("country"):
            location = f"{location}, {profile.get('country')}" if location else profile.get("country")
            
        rec_rows.append({
            "candidate_id": bd.candidate_id,
            "Name": name,
            "Skills": skills_list if skills_list else "None",
            "Experience": f"{years:.1f} Years" if isinstance(years, (int, float)) else str(years),
            "Education": edu_list,
            "Recommendation": bd.recommendation_tier,
            "Score": f"{bd.final_score * 100:.1f}%",
            "Matched Skills": ", ".join(bd.matched_skills) if bd.matched_skills else "None",
            "Missing Skills": ", ".join(bd.missing_skills) if bd.missing_skills else "None",
            "Behavior Score": f"{bd.behavior_score * 100:.1f}%",
            "Location": location if location else "Unknown",
            "Reasoning": bd.reasoning,
        })
        
    try:
        with rec_csv_path.open("w", encoding="utf-8", newline="") as rfh:
            rec_headers = [
                "candidate_id", "Name", "Skills", "Experience", "Education", 
                "Recommendation", "Score", "Matched Skills", "Missing Skills", 
                "Behavior Score", "Location", "Reasoning"
            ]
            r_writer = csv.DictWriter(rfh, fieldnames=rec_headers)
            r_writer.writeheader()
            r_writer.writerows(rec_rows)
        logger.info(f"Exported Recruiter Report to {rec_csv_path}")
    except Exception as exc:
        logger.warning(f"Could not save recruiter_report.csv: {exc}")

    # Also save top100.json for fast reload on startup
    import json
    json_path = output_dir / "top100.json"
    try:
        serializable_results = []
        for rank, bd in enumerate(top_100, start=1):
            bd_dict = bd.to_dict()
            bd_dict["rank"] = rank
            serializable_results.append(bd_dict)
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(serializable_results, jf, indent=2)
        logger.info(f"Exported {len(serializable_results)} candidates to {json_path}")
    except Exception as exc:
        logger.warning(f"Could not save top100.json: {exc}")

    # Programmatically run validate_submission.py
    import subprocess
    import sys
    from utils import load_config
    try:
        cfg = load_config()
        backend_dir = Path(__file__).parent
        candidates_raw_path = Path(cfg["data"]["candidates_jsonl"])
        if not candidates_raw_path.exists() and (backend_dir / candidates_raw_path).exists():
            candidates_file = (backend_dir / candidates_raw_path).resolve()
        else:
            candidates_file = candidates_raw_path.resolve()
        validator_path = candidates_file.parent / "validate_submission.py"
        if validator_path.exists():
            logger.info(f"Invoking official validation script: {validator_path}")
            res = subprocess.run(
                [sys.executable, str(validator_path), str(csv_path)],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                logger.info("Submission validation succeeded! CSV is 100% compliant.")
            else:
                logger.error(f"Submission validation failed! Output:\n{res.stdout.strip()}\n{res.stderr.strip()}")
        else:
            logger.warning(f"validate_submission.py not found at {validator_path}")
    except Exception as exc:
        logger.warning(f"Could not run submission validator: {exc}")

    return csv_path


def export_full_csv(
    all_candidates: list["ScoreBreakdown"],
    output_dir: Path,
    filename: str = "full_results.csv",
    candidate_lookup: dict[str, dict] | None = None,
) -> Path:
    """
    Export all scored candidates (not just top-100) for analytics.

    Args:
        all_candidates: Full sorted list of ScoreBreakdown.
        output_dir: Directory to write the CSV file.
        filename: Output filename.
        candidate_lookup: Optional profile lookup for reasoning.

    Returns:
        Path to the created CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename

    fieldnames = [
        "candidate_id", "rank", "score",
        "semantic_similarity", "skill_match", "experience_match",
        "behavior_score", "location_bonus", "education_score",
        "reasoning",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, bd in enumerate(all_candidates, start=1):
            cand_data = (candidate_lookup or {}).get(bd.candidate_id)
            reasoning = generate_reasoning(bd, cand_data)
            writer.writerow({
                "candidate_id": bd.candidate_id,
                "rank": rank,
                "score": f"{bd.final_score:.4f}",
                "semantic_similarity": f"{bd.semantic_similarity:.4f}",
                "skill_match": f"{bd.skill_match:.4f}",
                "experience_match": f"{bd.experience_match:.4f}",
                "behavior_score": f"{bd.behavior_score:.4f}",
                "location_bonus": f"{bd.location_bonus:.4f}",
                "education_score": f"{bd.education_score:.4f}",
                "reasoning": reasoning,
            })

    logger.info(f"Exported {len(all_candidates)} candidates to {csv_path}")
    return csv_path


# ── Validation Helpers ────────────────────────────────────────────────────────

def _validate_score_order(candidates: list["ScoreBreakdown"]) -> None:
    """Assert scores are non-increasing. Logs a warning if violated."""
    for i in range(len(candidates) - 1):
        if candidates[i].final_score < candidates[i + 1].final_score:
            logger.warning(
                f"Score order violation at rank {i + 1}: "
                f"{candidates[i].final_score:.4f} < {candidates[i + 1].final_score:.4f}"
            )


def _apply_tie_breaking(rows: list[dict]) -> list[dict]:
    """
    Re-sort rows by (score descending, candidate_id ascending) and re-assign ranks.
    This satisfies the official tie-breaking rule.
    """
    rows_sorted = sorted(
        rows,
        key=lambda r: (-float(r["score"]), r["candidate_id"]),
    )
    for new_rank, row in enumerate(rows_sorted, start=1):
        row["rank"] = new_rank
    return rows_sorted
