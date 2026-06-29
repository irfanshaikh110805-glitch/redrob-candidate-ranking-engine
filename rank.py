#!/usr/bin/env python3
"""
rank.py — CLI entry point to run the Candidate Discovery and Ranking System.
Reproduces the submission CSV from candidates.jsonl in a single command.
"""

import argparse
import sys
from pathlib import Path
import os

# Add backend directory to sys.path
backend_dir = Path(__file__).parent / "backend"
sys.path.append(str(backend_dir))

# Set environment variables for Sentence Transformers before loading it
os.environ["USE_TF"] = "OFF"
os.environ["USE_KERAS"] = "OFF"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from loguru import logger
from ranker import CandidateRanker
from utils import load_config, setup_logging

def main():
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranking Engine — CLI reproducer."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        help="Path to candidates.jsonl file.",
    )
    parser.add_argument(
        "--jd",
        type=str,
        help="Path to job_description.docx file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.csv",
        help="Output CSV path (default: submission.csv).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Initializing Redrob candidate ranking CLI...")

    cfg = load_config(backend_dir / "config.yaml")

    jd_path = args.jd or (backend_dir / cfg["data"]["job_description_docx"])
    cand_path = args.candidates or (backend_dir / cfg["data"]["candidates_jsonl"])

    jd_path = Path(jd_path)
    cand_path = Path(cand_path)

    if not jd_path.exists():
        logger.error(f"Job Description file not found at: {jd_path}")
        sys.exit(1)
    if not cand_path.exists():
        logger.error(f"Candidates file not found at: {cand_path}")
        sys.exit(1)

    logger.info(f"Using Job Description: {jd_path}")
    logger.info(f"Using Candidates File: {cand_path}")

    try:
        ranker = CandidateRanker()
        out_path = Path(args.out)
        
        logger.info("Starting candidate ranking pipeline...")
        result = ranker.rank(
            jd_path=jd_path,
            candidates_path=cand_path,
        )
        
        default_exported_path = Path(result.csv_path)
        if default_exported_path.exists() and default_exported_path.resolve() != out_path.resolve():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(default_exported_path, out_path)
            logger.info(f"Copied final CSV to requested destination: {out_path}")
            
        logger.info(f"Successfully ranked {result.total_candidates_evaluated:,} candidates in {result.elapsed_seconds:.1f}s.")
        logger.info(f"Top 100 CSV saved to: {out_path}")
        
    except Exception as e:
        logger.exception(f"Fatal error running candidate ranking: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
