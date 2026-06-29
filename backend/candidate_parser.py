"""
candidate_parser.py — Stream and parse candidates from candidates.jsonl.

The JSONL file is 487MB (~50K candidates). This module:
  1. Streams it line-by-line to keep memory usage low.
  2. Extracts all fields needed by the scoring engine.
  3. Builds a composite text blob per candidate for embedding.
  4. Yields CandidateRecord dataclasses in configurable chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator

from loguru import logger

from utils import clean_text_for_embedding, coerce_float, load_config, normalize_text


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class SkillRecord:
    name: str
    proficiency: str          # beginner | intermediate | advanced | expert
    endorsements: int
    duration_months: int


@dataclass
class CareerEntry:
    company: str
    title: str
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class EducationEntry:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    tier: str                 # tier_1 | tier_2 | tier_3 | tier_4 | unknown


@dataclass
class CertificationRecord:
    name: str
    issuer: str
    year: int


@dataclass
class RedrobSignals:
    profile_completeness_score: float
    open_to_work_flag: bool
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
    last_active_date: str


@dataclass
class CandidateRecord:
    """Fully parsed candidate, ready for scoring."""

    candidate_id: str
    # Profile
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    # Structured lists
    skills: list[SkillRecord] = field(default_factory=list)
    career_history: list[CareerEntry] = field(default_factory=list)
    education: list[EducationEntry] = field(default_factory=list)
    certifications: list[CertificationRecord] = field(default_factory=list)
    # Behavioral signals
    signals: RedrobSignals | None = None
    # Embedding text (built from all fields)
    embedding_text: str = ""

    @property
    def skill_names(self) -> list[str]:
        return [s.name for s in self.skills]

    @property
    def all_career_text(self) -> str:
        return " ".join(
            f"{e.title} at {e.company}: {e.description}" for e in self.career_history
        )


# ── Parser ────────────────────────────────────────────────────────────────────

class CandidateParser:
    """
    Streams candidates.jsonl and yields parsed CandidateRecord objects.

    Memory strategy: process one JSON object at a time.
    Chunk strategy: accumulate into batches for embedding efficiency.
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._logger = logger.bind(module="CandidateParser")

    # ── Public API ─────────────────────────────────────────────────────────────

    def stream(
        self,
        jsonl_path: str | Path,
        chunk_size: int | None = None,
    ) -> Generator[list[CandidateRecord], None, None]:
        """
        Stream candidates from a JSONL file in chunks.

        Args:
            jsonl_path: Path to candidates.jsonl.
            chunk_size: Number of candidates per chunk. Defaults to config value.

        Yields:
            Lists of CandidateRecord of length <= chunk_size.
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Candidates JSONL not found: {path}")

        chunk_size = chunk_size or self._cfg["performance"]["candidate_chunk_size"]
        self._logger.info(f"Streaming candidates from {path.name} (chunk_size={chunk_size})")

        chunk: list[CandidateRecord] = []
        total = 0

        for raw in self._iter_lines(path):
            record = self._parse_candidate(raw)
            if record is None:
                continue
            chunk.append(record)
            total += 1

            if len(chunk) >= chunk_size:
                self._logger.debug(f"Yielding chunk of {len(chunk)} (total so far: {total})")
                yield chunk
                chunk = []

        if chunk:
            self._logger.debug(f"Yielding final chunk of {len(chunk)} (total: {total})")
            yield chunk

        self._logger.info(f"Finished streaming {total} candidates")

    def count_candidates(self, jsonl_path: str | Path) -> int:
        """Count total candidates in the JSON or JSONL file without loading the whole thing if it is JSONL."""
        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as fh:
            first_char = ""
            for line in fh:
                line_strip = line.strip()
                if line_strip:
                    first_char = line_strip[0]
                    break
        
        if first_char == "[":
            with path.open("r", encoding="utf-8") as fh:
                try:
                    data = json.load(fh)
                    if isinstance(data, list):
                        return len(data)
                    return 1
                except Exception:
                    return 0
        else:
            count = 0
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        count += 1
            return count

    # ── Private Parsing ───────────────────────────────────────────────────────

    def _iter_lines(self, path: Path) -> Iterator[dict]:
        """Yield parsed JSON objects from a JSON or JSONL file."""
        with path.open("r", encoding="utf-8") as fh:
            first_char = ""
            for line in fh:
                line_strip = line.strip()
                if line_strip:
                    first_char = line_strip[0]
                    break
        
        if first_char == "[":
            with path.open("r", encoding="utf-8") as fh:
                try:
                    data = json.load(fh)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                    else:
                        yield data
                except Exception as exc:
                    self._logger.error(f"Failed to parse standard JSON file: {exc}")
        else:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._logger.warning(f"Skipping malformed JSON at line {lineno}: {exc}")

    def _parse_candidate(self, raw: dict[str, Any]) -> CandidateRecord | None:
        """
        Parse a single raw candidate dict into a CandidateRecord.

        Returns None if the candidate_id is missing or invalid.
        """
        cid = raw.get("candidate_id", "").strip()
        if not cid:
            return None

        profile = raw.get("profile", {})

        # Core profile fields
        record = CandidateRecord(
            candidate_id=cid,
            headline=profile.get("headline", ""),
            summary=profile.get("summary", ""),
            location=profile.get("location", ""),
            country=profile.get("country", ""),
            years_of_experience=coerce_float(profile.get("years_of_experience", 0)),
            current_title=profile.get("current_title", ""),
            current_company=profile.get("current_company", ""),
            current_company_size=profile.get("current_company_size", ""),
            current_industry=profile.get("current_industry", ""),
        )

        # Skills
        record.skills = [
            SkillRecord(
                name=s.get("name", ""),
                proficiency=s.get("proficiency", "beginner"),
                endorsements=int(s.get("endorsements", 0)),
                duration_months=int(s.get("duration_months", 0)),
            )
            for s in raw.get("skills", [])
            if s.get("name")
        ]

        # Career history
        record.career_history = [
            CareerEntry(
                company=c.get("company", ""),
                title=c.get("title", ""),
                duration_months=int(c.get("duration_months", 0)),
                is_current=bool(c.get("is_current", False)),
                industry=c.get("industry", ""),
                company_size=c.get("company_size", ""),
                description=c.get("description", ""),
            )
            for c in raw.get("career_history", [])
        ]

        # Education
        record.education = [
            EducationEntry(
                institution=e.get("institution", ""),
                degree=e.get("degree", ""),
                field_of_study=e.get("field_of_study", ""),
                start_year=int(e.get("start_year", 0)),
                end_year=int(e.get("end_year", 0)),
                tier=e.get("tier", "unknown"),
            )
            for e in raw.get("education", [])
        ]

        # Certifications
        record.certifications = [
            CertificationRecord(
                name=cert.get("name", ""),
                issuer=cert.get("issuer", ""),
                year=int(cert.get("year", 0)),
            )
            for cert in raw.get("certifications", [])
        ]

        # Redrob behavioral signals
        sig = raw.get("redrob_signals", {})
        if sig:
            record.signals = RedrobSignals(
                profile_completeness_score=coerce_float(sig.get("profile_completeness_score", 0)),
                open_to_work_flag=bool(sig.get("open_to_work_flag", False)),
                recruiter_response_rate=coerce_float(sig.get("recruiter_response_rate", 0)),
                avg_response_time_hours=coerce_float(sig.get("avg_response_time_hours", 999)),
                skill_assessment_scores=sig.get("skill_assessment_scores", {}),
                connection_count=int(sig.get("connection_count", 0)),
                endorsements_received=int(sig.get("endorsements_received", 0)),
                notice_period_days=int(sig.get("notice_period_days", 90)),
                preferred_work_mode=sig.get("preferred_work_mode", ""),
                willing_to_relocate=bool(sig.get("willing_to_relocate", False)),
                github_activity_score=coerce_float(sig.get("github_activity_score", -1)),
                search_appearance_30d=int(sig.get("search_appearance_30d", 0)),
                saved_by_recruiters_30d=int(sig.get("saved_by_recruiters_30d", 0)),
                interview_completion_rate=coerce_float(sig.get("interview_completion_rate", 0)),
                offer_acceptance_rate=coerce_float(sig.get("offer_acceptance_rate", -1)),
                verified_email=bool(sig.get("verified_email", False)),
                verified_phone=bool(sig.get("verified_phone", False)),
                linkedin_connected=bool(sig.get("linkedin_connected", False)),
                last_active_date=sig.get("last_active_date", ""),
            )

        # Build embedding text
        record.embedding_text = self._build_embedding_text(record)
        return record

    def _build_embedding_text(self, c: CandidateRecord) -> str:
        """
        Construct a rich text representation for semantic embedding.
        Ordered by semantic relevance to typical JD queries.
        """
        parts: list[str] = []

        # Title + headline is the strongest signal
        if c.current_title:
            parts.append(c.current_title)
        if c.headline:
            parts.append(c.headline)

        # Professional summary
        if c.summary:
            parts.append(c.summary[:300])

        # Skills list (expert/advanced weighted implicitly by position)
        expert_skills = [
            s.name for s in c.skills if s.proficiency in {"expert", "advanced"}
        ]
        all_skills = [s.name for s in c.skills]
        if expert_skills:
            parts.append("Expert in: " + ", ".join(expert_skills[:10]))
        elif all_skills:
            parts.append("Skills: " + ", ".join(all_skills[:10]))

        # Career history titles
        for entry in c.career_history[:2]:
            if entry.title:
                parts.append(f"Worked as {entry.title}")

        # Education
        for edu in c.education[:1]:
            parts.append(f"{edu.degree} in {edu.field_of_study}")

        return clean_text_for_embedding(parts)


# ── Convenience Function ─────────────────────────────────────────────────────

def stream_candidates(
    jsonl_path: str | Path,
    chunk_size: int | None = None,
) -> Generator[list[CandidateRecord], None, None]:
    """
    Convenience wrapper to stream candidates in chunks.

    Args:
        jsonl_path: Path to candidates.jsonl.
        chunk_size: Candidates per chunk (default from config).

    Yields:
        Chunks of CandidateRecord.
    """
    return CandidateParser().stream(jsonl_path, chunk_size)
