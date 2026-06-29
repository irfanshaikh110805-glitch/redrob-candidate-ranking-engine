"""
scoring.py — Multi-signal scoring engine for the Redrob ranking system.

Computes six independent sub-scores (all in [0, 1]) per candidate:
  1. semantic_similarity  — cosine sim between JD & candidate embeddings
  2. skill_match          — RapidFuzz token matching with proficiency weighting
  3. experience_match     — Gaussian proximity to JD's ideal experience range
  4. behavior_score       — Composite of Redrob platform signals
  5. location_bonus       — City / country / relocation match
  6. education_score      — Degree level + field relevance + institution tier

All scoring is fully vectorized (numpy) for throughput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from rapidfuzz import fuzz, process

from candidate_parser import CandidateRecord, RedrobSignals
from jd_parser import ParsedJD
from utils import clamp, load_config, normalize_text


# ── Score Breakdown Container ─────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    """Per-candidate score components before weighted aggregation."""

    candidate_id: str
    semantic_similarity: float
    skill_match: float
    experience_match: float
    behavior_score: float
    location_bonus: float
    education_score: float
    final_score: float
    reasoning: str = ""
    recommendation_tier: str = "Consider"
    confidence_level: str = "Medium"
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "skill_match": round(self.skill_match, 4),
            "experience_match": round(self.experience_match, 4),
            "behavior_score": round(self.behavior_score, 4),
            "location_bonus": round(self.location_bonus, 4),
            "education_score": round(self.education_score, 4),
            "final_score": round(self.final_score, 4),
            "reasoning": self.reasoning,
            "recommendation_tier": self.recommendation_tier,
            "confidence_level": self.confidence_level,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
        }


# ── Scoring Engine ────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Computes all sub-scores for a batch of candidates against a parsed JD.

    Design philosophy:
      - All computations are stateless and side-effect-free.
      - Heavy numpy operations are vectorized over the batch.
      - Per-candidate Python logic (skill matching, education) runs in a tight loop
        but is O(skills_per_candidate × jd_skills) ≈ very fast.
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._weights = self._cfg["weights"]
        self._exp_cfg = self._cfg["experience"]
        self._skill_cfg = self._cfg["skills"]
        self._edu_cfg = self._cfg["education"]
        self._loc_cfg = self._cfg["location"]
        self._beh_cfg = self._cfg["behavior"]
        self._logger = logger.bind(module="ScoringEngine")

    # ── Honeypot Detection ──────────────────────────────────────────────────

    def _is_honeypot(self, candidate: CandidateRecord) -> bool:
        """
        Identify honeypot (trap) candidates with impossible profiles.
        """
        # 1. Expert or Advanced skills with exactly 0 duration
        expert_0 = [
            s for s in candidate.skills 
            if s.proficiency in ("expert", "advanced") and s.duration_months == 0
        ]
        if len(expert_0) >= 2:
            return True
            
        # 2. Single job duration in career history (in years) exceeds years of experience
        for h in candidate.career_history:
            dur_years = h.duration_months / 12.0
            if dur_years > candidate.years_of_experience + 0.5:
                return True
                
        # 3. Sum of all job durations in career history exceeds years of experience by a lot
        total_months = sum(h.duration_months for h in candidate.career_history)
        total_years = total_months / 12.0
        if total_years > candidate.years_of_experience + 3.0:
            return True
            
        return False

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def score_batch(
        self,
        candidates: list[CandidateRecord],
        jd: ParsedJD,
        semantic_scores: np.ndarray,
    ) -> list[ScoreBreakdown]:
        """
        Score a batch of candidates against the parsed JD.

        Args:
            candidates: List of CandidateRecord objects.
            jd: ParsedJD with extracted fields.
            semantic_scores: Pre-computed cosine similarities (one per candidate).

        Returns:
            List of ScoreBreakdown, one per candidate.
        """
        if len(candidates) != len(semantic_scores):
            raise ValueError(
                f"Mismatch: {len(candidates)} candidates vs {len(semantic_scores)} scores."
            )

        # Normalize semantic scores to [0, 1] (they come in as cosine similarity [-1, 1])
        sem_normalized = ((semantic_scores + 1.0) / 2.0).clip(0.0, 1.0)

        jd_skills_normalized = [normalize_text(s) for s in (jd.required_skills + jd.preferred_skills)]
        jd_location_normalized = normalize_text(jd.location)

        breakdowns: list[ScoreBreakdown] = []

        for i, (candidate, sem_score) in enumerate(zip(candidates, sem_normalized)):
            # Honeypot Check
            if self._is_honeypot(candidate):
                self._logger.warning(f"Flagged honeypot candidate: {candidate.candidate_id}. Dropping score to 0.0.")
                breakdowns.append(
                    ScoreBreakdown(
                        candidate_id=candidate.candidate_id,
                        semantic_similarity=0.0,
                        skill_match=0.0,
                        experience_match=0.0,
                        behavior_score=0.0,
                        location_bonus=0.0,
                        education_score=0.0,
                        final_score=0.0,
                        reasoning="Disqualified: Impossible profile / honeypot candidate.",
                        recommendation_tier="Reject",
                        confidence_level="Low"
                    )
                )
                continue

            skill_score, matched_req = self._compute_skill_match(candidate, jd, jd_skills_normalized)
            exp_score = self._compute_experience_match(candidate)
            beh_score = self._compute_behavior_score(candidate.signals)
            loc_score = self._compute_location_bonus(candidate, jd_location_normalized)
            edu_score = self._compute_education_score(candidate)

            final = (
                self._weights["semantic_similarity"] * float(sem_score)
                + self._weights["skill_match"] * skill_score
                + self._weights["experience_match"] * exp_score
                + self._weights["behavior_score"] * beh_score
                + self._weights["location_bonus"] * loc_score
                + self._weights["education_score"] * edu_score
            )
            final = clamp(final)

            # Match required skills for explainable AI reasoning (reused from skill match)
            missing_req = [s for s in jd.required_skills if s not in matched_req]

            matched_str = ", ".join(matched_req[:3]) if matched_req else "None"
            missing_str = ", ".join(missing_req[:3]) if missing_req else "None"

            # Platform engagement classification
            eng_level = "High" if beh_score >= 0.75 else "Medium" if beh_score >= 0.40 else "Low"
            if candidate.signals and candidate.signals.github_activity_score >= 0:
                eng_level += f" (GitHub: {candidate.signals.github_activity_score:.0f}%)"

            # Location match status
            loc_status = (
                "Exact City Match" if loc_score >= 0.9 
                else "Same Country" if loc_score >= 0.6 
                else "Willing to Relocate" if loc_score >= 0.4 
                else "Remote/Flexible Preference" if loc_score > 0.0 
                else "Location Mismatch"
            )

            # Build explainable reasoning string conforming to plan.md
            reasoning = (
                f"Matched [{matched_str}] | Missing [{missing_str}] | "
                f"{candidate.years_of_experience:.1f} years experience | "
                f"Platform engagement: {eng_level} | Location: {loc_status} | "
                f"Final Score: {final * 100:.1f}%"
            )

            # Recommendation tier
            rec_tier = (
                "Highly Recommended" if final >= 0.85 
                else "Recommended" if final >= 0.75 
                else "Consider" if final >= 0.60 
                else "Needs Improvement" if final >= 0.40 
                else "Reject"
            )

            # Ranking confidence calculation
            conf_level = "Medium"
            if candidate.signals:
                if candidate.signals.profile_completeness_score >= 80 and candidate.signals.github_activity_score >= 0:
                    conf_level = "High"
                elif candidate.signals.profile_completeness_score < 50:
                    conf_level = "Low"

            breakdowns.append(
                ScoreBreakdown(
                    candidate_id=candidate.candidate_id,
                    semantic_similarity=float(sem_score),
                    skill_match=skill_score,
                    experience_match=exp_score,
                    behavior_score=beh_score,
                    location_bonus=loc_score,
                    education_score=edu_score,
                    final_score=final,
                    reasoning=reasoning,
                    recommendation_tier=rec_tier,
                    confidence_level=conf_level,
                    matched_skills=matched_req,
                    missing_skills=missing_req
                )
            )

        return breakdowns

    # ── Skill Match ───────────────────────────────────────────────────────────

    def _compute_skill_match(
        self,
        candidate: CandidateRecord,
        jd: ParsedJD,
        jd_skills_normalized: list[str],
    ) -> tuple[float, list[str]]:
        """
        Compute weighted skill match using RapidFuzz fuzzy matching.

        Strategy:
          - For each JD skill, find the best-matching candidate skill.
          - Weight by proficiency level and endorsement count.
          - Required skills are weighted 2× preferred skills.
          - Return normalized score [0, 1] and a list of matched required skills.
        """
        if not jd_skills_normalized or not candidate.skills:
            return 0.0, []

        prof_weights = self._skill_cfg["proficiency_weights"]
        threshold = self._skill_cfg["fuzzy_threshold"]
        endorsement_cap = self._skill_cfg["endorsement_cap"]
        duration_cap = self._skill_cfg["duration_cap_months"]
        endorsement_w = self._skill_cfg["endorsement_weight"]
        duration_w = self._skill_cfg["duration_weight"]

        # Build candidate skill map: normalized_name → SkillRecord
        candidate_skill_map = {normalize_text(s.name): s for s in candidate.skills}
        candidate_skill_names = list(candidate_skill_map.keys())

        if not candidate_skill_names:
            return 0.0, []

        n_required = len(jd.required_skills)
        total_weight = 0.0
        matched_weight = 0.0
        matched_required: list[str] = []

        for idx, jd_skill in enumerate(jd_skills_normalized):
            is_req = idx < n_required
            # Required skills count double
            jd_weight = 2.0 if is_req else 1.0
            total_weight += jd_weight

            # Exact match fast path
            if jd_skill in candidate_skill_map:
                matched_name = jd_skill
                score = 100.0
                is_match = True
            else:
                # Find best matching candidate skill via fuzzy matching
                match = process.extractOne(
                    jd_skill,
                    candidate_skill_names,
                    scorer=fuzz.token_set_ratio,
                    score_cutoff=threshold,
                )
                if match is not None:
                    matched_name, score, _ = match
                    is_match = True
                else:
                    is_match = False

            if is_match:
                if is_req:
                    # Retrieve the original un-normalized required skill name
                    matched_required.append(jd.required_skills[idx])

                skill_rec = candidate_skill_map[matched_name]

                # Base match quality (fuzzy score 75–100 → 0.5–1.0)
                fuzzy_quality = (score - threshold) / (100.0 - threshold)

                # Proficiency bonus
                prof_bonus = prof_weights.get(skill_rec.proficiency, 0.4)

                # Endorsement sub-score
                endorsement_sub = min(skill_rec.endorsements, endorsement_cap) / endorsement_cap

                # Duration sub-score
                duration_sub = min(skill_rec.duration_months, duration_cap) / duration_cap

                # Combined skill value
                skill_value = (
                    (1.0 - endorsement_w - duration_w) * prof_bonus * fuzzy_quality
                    + endorsement_w * endorsement_sub
                    + duration_w * duration_sub
                )
                matched_weight += jd_weight * clamp(skill_value)

        if total_weight == 0:
            return 0.0, []

        return clamp(matched_weight / total_weight), matched_required

    # ── Experience Match ──────────────────────────────────────────────────────

    def _compute_experience_match(self, candidate: CandidateRecord) -> float:
        """
        Score experience using a plateau + Gaussian decay model.

        - Within [ideal_min, ideal_max]: score = 1.0
        - Outside this range: score decays exponentially with sigma
        """
        years = candidate.years_of_experience
        years = clamp(years, lo=0.0, hi=self._exp_cfg["max_years_cap"])

        lo = self._exp_cfg["ideal_min_years"]
        hi = self._exp_cfg["ideal_max_years"]
        sigma = self._exp_cfg["decay_sigma"]

        if lo <= years <= hi:
            return 1.0

        # Distance from nearest edge of ideal range
        dist = (lo - years) if years < lo else (years - hi)
        score = math.exp(-(dist ** 2) / (2 * sigma ** 2))
        return clamp(score)

    # ── Behavior Score ────────────────────────────────────────────────────────

    def _compute_behavior_score(self, signals: RedrobSignals | None) -> float:
        """
        Aggregate Redrob platform behavioral signals into a single score.

        Each sub-signal is normalized to [0, 1] before weighting.
        """
        if signals is None:
            return 0.0

        cfg = self._beh_cfg

        # 1. Profile completeness (0–100 → 0–1)
        completeness = clamp(signals.profile_completeness_score / 100.0)

        # 2. Recruiter response rate (already 0–1)
        response_rate = clamp(signals.recruiter_response_rate)

        # 3. GitHub activity (−1 means no GitHub; 0–100 → 0–1)
        github = clamp(signals.github_activity_score / 100.0) if signals.github_activity_score >= 0 else 0.0

        # 4. Skill assessment average (0–100 → 0–1)
        assessments = list(signals.skill_assessment_scores.values())
        assessment_avg = (sum(assessments) / len(assessments) / 100.0) if assessments else 0.0
        assessment_avg = clamp(assessment_avg)

        # 5. Interview completion rate (already 0–1)
        interview = clamp(signals.interview_completion_rate)

        # 6. Open to work flag
        open_to_work = 1.0 if signals.open_to_work_flag else 0.4

        # 7. Engagement composite: views, saved, connections
        views_norm = clamp(signals.search_appearance_30d / 500.0)
        saved_norm = clamp(signals.saved_by_recruiters_30d / 20.0)
        connections_norm = clamp(signals.connection_count / 1000.0)
        engagement = (views_norm + saved_norm + connections_norm) / 3.0

        # Weighted sum
        score = (
            cfg["profile_completeness_weight"] * completeness
            + cfg["recruiter_response_weight"] * response_rate
            + cfg["github_activity_weight"] * github
            + cfg["assessment_score_weight"] * assessment_avg
            + cfg["interview_completion_weight"] * interview
            + cfg["open_to_work_weight"] * open_to_work
            + cfg["engagement_weight"] * engagement
        )
        return clamp(score)

    # ── Location Bonus ────────────────────────────────────────────────────────

    def _compute_location_bonus(
        self,
        candidate: CandidateRecord,
        jd_location_normalized: str,
    ) -> float:
        """
        Compute location match score.

        Scoring tiers:
          - Exact city match: 1.0
          - Same country as JD location (India): 0.6
          - Willing to relocate: 0.4
          - No match: 0.0
        """
        if not jd_location_normalized:
            return 0.5  # No location requirement → neutral

        cfg = self._loc_cfg
        candidate_location = normalize_text(candidate.location)
        candidate_country = normalize_text(candidate.country)

        # Check city match
        if jd_location_normalized in candidate_location or candidate_location in jd_location_normalized:
            return cfg["exact_city_score"]

        # Check if JD location is a known city and candidate is in same country
        jd_in_india = any(
            city in jd_location_normalized
            for city in ["bengaluru", "bangalore", "mumbai", "delhi", "gurgaon",
                         "hyderabad", "chennai", "pune", "kolkata", "noida", "india"]
        )
        if jd_in_india and candidate_country == "india":
            return cfg["same_country_score"]

        # Remote / flexible
        if candidate.signals and candidate.signals.preferred_work_mode in {"remote", "flexible"}:
            return cfg["same_country_score"]

        # Willing to relocate
        if candidate.signals and candidate.signals.willing_to_relocate:
            return cfg["willing_to_relocate_score"]

        return cfg["no_match_score"]

    # ── Education Score ───────────────────────────────────────────────────────

    def _compute_education_score(self, candidate: CandidateRecord) -> float:
        """
        Score education based on:
          1. Highest degree level
          2. Field of study relevance to tech/AI
          3. Institution tier

        Returns the best education entry score.
        """
        if not candidate.education:
            return 0.2  # No education listed → small default

        cfg = self._edu_cfg
        degree_weights = cfg["degree_weights"]
        tier_weights = cfg["tier_weights"]
        relevant_fields = [f.lower() for f in cfg["relevant_fields"]]
        field_bonus = cfg["field_relevance_bonus"]

        best_score = 0.0

        for edu in candidate.education:
            # Degree level score
            degree_norm = edu.degree.lower().strip().rstrip(".")
            deg_score = 0.5  # default for unrecognized degrees
            for key, val in degree_weights.items():
                if key in degree_norm:
                    deg_score = val
                    break

            # Institution tier score
            tier_score = tier_weights.get(edu.tier, 0.5)

            # Field relevance
            field_norm = edu.field_of_study.lower()
            is_relevant = any(f in field_norm for f in relevant_fields)
            field_score = 1.0 if is_relevant else 0.5

            # Combined score for this education entry
            entry_score = 0.5 * deg_score + 0.3 * tier_score + 0.2 * field_score
            entry_score = clamp(entry_score + (field_bonus if is_relevant else 0))

            best_score = max(best_score, entry_score)

        return clamp(best_score)
