"""
jd_parser.py — Job Description parser for the Redrob ranking engine.

Reads a .docx file and extracts structured fields:
  - required_skills
  - preferred_skills
  - experience_range
  - education_requirements
  - location
  - soft_skills
  - responsibilities
  - raw_text (for embedding)

The parser is keyword-section aware and falls back to full-text
extraction if structured sections are not found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from docx import Document
from loguru import logger

from utils import normalize_text


# ── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class ParsedJD:
    """Structured representation of a parsed Job Description."""

    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    experience_min_years: float = 0.0
    experience_max_years: float = 15.0
    education_requirements: list[str] = field(default_factory=list)
    location: str = ""
    soft_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    raw_text: str = ""
    embedding_text: str = ""

    def to_dict(self) -> dict:
        return {
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "experience_min_years": self.experience_min_years,
            "experience_max_years": self.experience_max_years,
            "education_requirements": self.education_requirements,
            "location": self.location,
            "soft_skills": self.soft_skills,
            "responsibilities": self.responsibilities,
        }


# ── Section Detection Patterns ────────────────────────────────────────────────

# Headings that signal sections
_EXPERIENCE_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*(\d+(?:\.\d+)?))?\s*(?:\+)?\s*years?",
    re.IGNORECASE,
)


# ── Core Parser ──────────────────────────────────────────────────────────────

class JDParser:
    """
    Parses a .docx Job Description into a structured ParsedJD object.

    Strategy:
      1. Extract all paragraphs and their heading levels.
      2. Identify sections by heading keywords.
      3. Collect bullet points / text under each section.
      4. Fallback to full-text regex parsing where sections are absent.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(module="JDParser")

    def parse(self, docx_path: str | Path) -> ParsedJD:
        """
        Parse a .docx job description file.

        Args:
            docx_path: Absolute or relative path to the .docx file.

        Returns:
            ParsedJD with all extracted fields populated.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid .docx.
        """
        path = Path(docx_path)
        if not path.exists():
            raise FileNotFoundError(f"JD file not found: {path}")
        if path.suffix.lower() not in {".docx", ".doc"}:
            raise ValueError(f"Expected a .docx file, got: {path.suffix}")

        self._logger.info(f"Parsing JD from {path.name}")
        doc = Document(str(path))

        # Extract all paragraphs with metadata
        paragraphs = self._extract_paragraphs(doc)
        raw_text = "\n".join(p["text"] for p in paragraphs if p["text"])

        # Section-aware extraction
        sections = self._split_into_sections(paragraphs)
        jd = ParsedJD(raw_text=raw_text)

        # Tech keywords for structured extraction from sections
        TECH_KEYWORDS = [
            "Python", "PyTorch", "TensorFlow", "scikit-learn", "FastAPI", "Docker", "Kubernetes",
            "AWS", "GCP", "Azure", "SQL", "NoSQL", "FAISS", "Pinecone", "Weaviate", "Qdrant",
            "Milvus", "OpenSearch", "Elasticsearch", "Spark", "Airflow", "Kafka", "Git",
            "sentence-transformers", "OpenAI embeddings", "BGE", "E5", "LangChain", "LlamaIndex",
            "XGBoost", "Machine Learning", "Deep Learning", "NLP", "LLM", "RAG", "Embeddings",
            "Fine-tuning", "MLOps", "Data Engineering", "Vector Database", "Evaluation",
            "NDCG", "MRR", "MAP", "LoRA", "QLoRA", "PEFT", "learning-to-rank", "distributed systems",
            "search", "retrieval", "ranking", "A/B testing"
        ]

        def extract_skills_from_bullets(bullets: list[str]) -> list[str]:
            skills: list[str] = []
            for b in bullets:
                for kw in TECH_KEYWORDS:
                    pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                    if re.search(pattern, b.lower()):
                        if kw not in skills:
                            skills.append(kw)
            return skills

        # Extract structured required and preferred skills
        jd.required_skills = extract_skills_from_bullets(sections.get("required_skills", []))
        jd.preferred_skills = extract_skills_from_bullets(sections.get("preferred_skills", []))

        # Fallback to general keyword match if empty
        if not jd.required_skills:
            jd.required_skills = self._extract_skills_from_text(raw_text)
            self._logger.warning(
                "No structured 'Required Skills' section found — using full-text fallback."
            )

        jd.responsibilities = self._extract_items(sections.get("responsibilities", []))
        jd.soft_skills = self._extract_items(sections.get("soft_skills", []))
        jd.education_requirements = self._extract_items(sections.get("education", []))
        jd.location = self._extract_location(sections.get("location", []), raw_text)

        # Extract experience from full text paragraphs
        paragraph_texts = [p["text"] for p in paragraphs if p["text"]]
        jd.experience_min_years, jd.experience_max_years = self._extract_experience(paragraph_texts)

        # Build embedding text (ordered by importance)
        jd.embedding_text = self._build_embedding_text(jd)

        self._logger.info(
            f"JD parsed: {len(jd.required_skills)} required skills, "
            f"{len(jd.preferred_skills)} preferred skills, "
            f"exp {jd.experience_min_years}–{jd.experience_max_years} yrs, "
            f"location={jd.location!r}"
        )
        return jd

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _extract_paragraphs(self, doc: Document) -> list[dict]:
        """Return list of {text, style, is_heading, level} dicts."""
        result = []
        for para in doc.paragraphs:
            text = para.text.strip()
            is_heading = para.style.name.startswith("Heading")
            level = 0
            if is_heading:
                try:
                    level = int(para.style.name.split()[-1])
                except (ValueError, IndexError):
                    level = 1
            result.append({"text": text, "style": para.style.name, "is_heading": is_heading, "level": level})
        return result

    def _split_into_sections(self, paragraphs: list[dict]) -> dict[str, list[str]]:
        """
        Group paragraph texts under their nearest heading section.

        Returns a dict mapping section_key → list of text lines.
        """
        sections: dict[str, list[str]] = {}
        current_section: str | None = None

        for para in paragraphs:
            text = para["text"]
            if not text:
                continue

            if para["is_heading"] or self._looks_like_heading(text):
                current_section = self._classify_heading(text)
                if current_section not in sections:
                    sections[current_section] = []
            elif current_section:
                sections[current_section].append(text)
            else:
                # Before any heading — might contain location / overview
                sections.setdefault("overview", []).append(text)

        return sections

    def _looks_like_heading(self, text: str) -> bool:
        """Heuristic: short ALL-CAPS or Title Case lines may be headings."""
        if len(text) > 80:
            return False
        words = text.split()
        if not words:
            return False
        capitalized = sum(1 for w in words if w and w[0].isupper())
        return capitalized / len(words) >= 0.7 and len(words) <= 8

    def _classify_heading(self, text: str) -> str | None:
        """Return the section key for a heading text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["absolutely need", "required", "mandatory", "must have", "core"]):
            return "required_skills"
        if any(w in text_lower for w in ["like you to have", "preferred", "nice to have", "bonus", "desired", "plus"]):
            return "preferred_skills"
        if any(w in text_lower for w in ["responsibilities", "role", "duties", "what you'll do", "key responsibilities", "what we mean by"]):
            return "responsibilities"
        if any(w in text_lower for w in ["soft skills", "interpersonal", "communication", "behavioral"]):
            return "soft_skills"
        if any(w in text_lower for w in ["education", "qualification", "degree", "academic"]):
            return "education"
        if any(w in text_lower for w in ["experience", "years", "work history"]):
            return "experience"
        if any(w in text_lower for w in ["location", "office", "based in", "work location", "logistics"]):
            return "location"
        return "other"

    def _extract_items(self, lines: list[str]) -> list[str]:
        """
        Convert a list of raw paragraph lines into clean skill/item strings.
        Handles bullet points, numbered lists, and comma-separated values.
        """
        items: list[str] = []
        for line in lines:
            # Strip common bullet/list characters
            cleaned = re.sub(r"^[\s•●▪▸◦\-\*\d+\.\)]+", "", line).strip()
            if not cleaned:
                continue
            # Split on commas if line looks like a comma list
            if "," in cleaned and len(cleaned.split(",")) > 2:
                items.extend(s.strip() for s in cleaned.split(",") if s.strip())
            else:
                items.append(cleaned)
        return [i for i in items if len(i) > 1]

    def _extract_experience(self, paragraphs_text_list: list[str]) -> tuple[float, float]:
        """
        Extract experience year range from doc paragraphs.
        Looks for the first range match in a paragraph containing "experience" or "exp".
        """
        for line in paragraphs_text_list:
            if "experience" in line.lower() or "exp" in line.lower():
                matches = _EXPERIENCE_RANGE_RE.findall(line)
                for m in matches:
                    if m[0] and m[1]:  # Both min and max are present
                        return float(m[0]), float(m[1])

        # Try any range match in the document
        for line in paragraphs_text_list:
            matches = _EXPERIENCE_RANGE_RE.findall(line)
            for m in matches:
                if m[0] and m[1]:
                    return float(m[0]), float(m[1])

        # Fallback to single value
        for line in paragraphs_text_list:
            if "experience" in line.lower() or "exp" in line.lower():
                matches = _EXPERIENCE_RANGE_RE.findall(line)
                for m in matches:
                    if m[0]:
                        val = float(m[0])
                        return val, val + 5.0

        return 3.0, 8.0  # Default fallback

    def _extract_location(self, location_lines: list[str], fallback_text: str) -> str:
        """Extract the primary location string."""
        if location_lines:
            return " ".join(location_lines[:2]).strip()

        # Try to find location from full text
        loc_match = re.search(
            r"(location|based\s+in|office)[:\s]+([A-Za-z,\s]+?)(?:\n|$)",
            fallback_text,
            re.IGNORECASE,
        )
        if loc_match:
            return loc_match.group(2).strip()

        # Look for Indian city names
        cities = [
            "Bengaluru", "Bangalore", "Mumbai", "Delhi", "Gurgaon", "Gurugram",
            "Hyderabad", "Chennai", "Pune", "Kolkata", "Noida", "Remote",
        ]
        for city in cities:
            if city.lower() in fallback_text.lower():
                return city

        return ""

    def _extract_skills_from_text(self, text: str) -> list[str]:
        """
        Fallback: extract potential skills from full JD text using
        a curated list of technology keywords.
        """
        tech_keywords = [
            "Python", "Machine Learning", "Deep Learning", "NLP", "LLM",
            "Transformers", "PyTorch", "TensorFlow", "Scikit-learn", "FastAPI",
            "Docker", "Kubernetes", "AWS", "GCP", "Azure", "SQL", "NoSQL",
            "RAG", "Vector Database", "FAISS", "Embeddings", "Fine-tuning",
            "MLOps", "Data Engineering", "Spark", "Airflow", "Kafka",
            "REST API", "Microservices", "Git", "Linux",
        ]
        found = [kw for kw in tech_keywords if kw.lower() in text.lower()]
        return found

    def _build_embedding_text(self, jd: ParsedJD) -> str:
        """
        Build a single rich text string optimized for semantic embedding.
        Orders content by importance for the model's attention.
        """
        parts = []

        if jd.required_skills:
            parts.append("Required skills: " + ", ".join(jd.required_skills))

        if jd.responsibilities:
            parts.append("Responsibilities: " + " ".join(jd.responsibilities[:5]))

        if jd.preferred_skills:
            parts.append("Preferred skills: " + ", ".join(jd.preferred_skills))

        if jd.soft_skills:
            parts.append("Soft skills: " + ", ".join(jd.soft_skills))

        if jd.education_requirements:
            parts.append("Education: " + ", ".join(jd.education_requirements))

        if jd.location:
            parts.append(f"Location: {jd.location}")

        exp_str = (
            f"Experience: {jd.experience_min_years}–{jd.experience_max_years} years"
        )
        parts.append(exp_str)

        # Pad with raw text excerpt for broader semantic coverage
        if jd.raw_text:
            parts.append(jd.raw_text[:500])

        return " ".join(parts)[:2048]


# ── Convenience Function ─────────────────────────────────────────────────────

def parse_job_description(docx_path: str | Path) -> ParsedJD:
    """
    Convenience function to parse a JD file.

    Args:
        docx_path: Path to the .docx file.

    Returns:
        ParsedJD object.
    """
    return JDParser().parse(docx_path)
