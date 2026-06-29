import sys
import csv
import json
from pathlib import Path
from loguru import logger

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from jd_parser import parse_job_description
from candidate_parser import CandidateParser
from embeddings import EmbeddingEngine
from scoring import ScoringEngine
from utils import ensure_output_dir, load_config

def main():
    logger.info("Initializing reconstruction of top100.json...")
    cfg = load_config()
    
    backend_dir = Path(__file__).parent
    jd_path = (backend_dir / cfg["data"]["job_description_docx"]).resolve()
    candidates_path = (backend_dir / cfg["data"]["candidates_jsonl"]).resolve()
    out_dir = ensure_output_dir()
    
    csv_path = out_dir / "submission.csv"
    if not csv_path.exists():
        logger.error(f"submission.csv not found at {csv_path}")
        return
        
    # 1. Read top 100 candidate IDs
    top_ids = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            top_ids.append(row["candidate_id"])
            
    logger.info(f"Loaded {len(top_ids)} candidate IDs from CSV.")
    
    # 2. Parse JD
    jd = parse_job_description(jd_path)
    
    # 3. Read candidate records from JSONL
    logger.info("Scanning candidates.jsonl for target profiles...")
    parser = CandidateParser()
    target_records = {}
    
    for raw in parser._iter_lines(candidates_path):
        cid = raw.get("candidate_id", "").strip()
        if cid in top_ids:
            rec = parser._parse_candidate(raw)
            if rec:
                target_records[cid] = rec
            if len(target_records) >= len(top_ids):
                break
                
    logger.info(f"Loaded {len(target_records)} candidate profiles.")
    
    # Order them according to top_ids
    ordered_candidates = [target_records[cid] for cid in top_ids if cid in target_records]
    
    # 4. Generate embeddings
    logger.info("Generating embeddings for top candidates...")
    emb_engine = EmbeddingEngine()
    texts = [c.embedding_text for c in ordered_candidates]
    candidate_embeddings = emb_engine.encode_batch(texts)
    jd_embedding = emb_engine.encode_single(jd.embedding_text)
    
    semantic_scores = emb_engine.cosine_similarity_batch(jd_embedding, candidate_embeddings)
    
    # 5. Score them
    logger.info("Scoring top candidates...")
    scoring_engine = ScoringEngine()
    breakdowns = scoring_engine.score_batch(ordered_candidates, jd, semantic_scores)
    
    # 6. Save top100.json
    json_path = out_dir / "top100.json"
    serializable_results = []
    for rank, bd in enumerate(breakdowns, start=1):
        bd_dict = bd.to_dict()
        bd_dict["rank"] = rank
        serializable_results.append(bd_dict)
        
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(serializable_results, jf, indent=2)
        
    logger.info(f"Successfully saved reconstructed top 100 to {json_path}")

if __name__ == "__main__":
    main()
