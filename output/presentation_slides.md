# Redrob AI Candidate Ranking Engine — Pitch Deck Outline

This document contains the slide structure, visual layout guides, and presenter speaker notes for the **Redrob India Runs Hackathon** pitch presentation.

---

### Slide 1: Title & Overview [IMPROVED]
* **Slide Title**: AI-Powered Candidate Discovery & Ranking Engine
* **Subtitle**: High-Accuracy Contextual Candidate Matching Designed for Large Datasets
* **Key Details**:
  * **Problem**: Traditional resume screens fail to identify qualified candidates due to semantic phrasing gaps and keyword limits.
  * **Solution**: Localized embedding-based retrieval coupled with multi-signal heuristic scoring.
  * **Key Tech**: Sentence Transformers, FAISS Vector Search, RapidFuzz String Alignments, and FastAPI.
  * **Participant registered ID**: Team Candidate Ranker
  * **Hackathon Name**: Redrob India Runs Data & AI Hackathon
* **Suggested Visual**: Large title card in Slate Gray background with Bright Blue accent highlights, a clean minimalist AI system emblem, and clear metadata boxes.
* **Icons**: `Cpu`, `Layers`, `ShieldCheck`
* **Layout**: Centered minimal hero block with two columns for problem/solution.
* **Speaker Notes (45s)**: 
  > "Hello judges. Today we are presenting our AI-Powered Candidate Discovery and Ranking Engine, designed to process large candidate datasets. Standard search tools filter candidates based on strict keywords, meaning highly qualified profiles are often missed due to differences in phrasing. Our solution solves this problem by using localized embedding models to match candidates contextually, combined with a robust multi-signal heuristic scoring system. We leverage Sentence Transformers, FAISS, and RapidFuzz to execute this entire pipeline efficiently on a local CPU."

---

### Slide 2: Problem Statement [IMPROVED]
* **Slide Title**: Why Keyword Matching Fails Recruiter Workflows
* **Key Bullet Points**:
  * **High-Volume Overload**: Sifting through thousands of sparse candidate records manually is slow and prone to fatigue.
  * **Semantic Expression Gap**: Keyword filters miss synonyms (e.g., searching 'Embeddings' misses candidates listing 'Vector Representations').
  * **Proficiency Blind Spot**: Raw keyword searches fail to weigh skill duration, endorsements, or real proficiency levels.
  * **Location & Relocation Friction**: Location preferences and willingness to relocate are rarely computed as weighted indicators.
  * **Degree & Field Mismatches**: Hard education filters reject relevant fields and equivalent degrees.
* **Suggested Visual**: A comparative table or split view. Left side: "Keyword Filters" (Red warning borders, lists failures). Right side: "Contextual Matching" (Green success borders, lists advantages).
* **Icons**: `XCircle`, `AlertTriangle`, `TrendingDown`
* **Layout**: Two-column split grid.
* **Speaker Notes (50s)**:
  > "Recruiters face massive overload when reviewing candidate pools. Unfortunately, simple keyword filters do not work well. If a recruiter searches for 'Embeddings', they will miss candidates listing 'Vector Representations' or 'Semantic Retrieval' due to phrasing differences. Furthermore, keyword searches cannot assess experience decay, platform engagement, or location willingness in a unified score. Our system addresses these gaps by evaluating candidate quality across six distinct signals, scoring candidates based on contextual alignment rather than simple string matching."

---

### Slide 3: System Architecture [IMPROVED]
* **Slide Title**: Modular Candidate Processing Architecture
* **Key Bullet Points**:
  * **Section-Aware JD Parsing**: Extracts skills, location limits, and experience sweet-spots from Job Descriptions.
  * **Streamed Data Pipeline**: Reads large JSONL datasets in chunks of 5,000 to keep memory usage flat.
  * **Batch Embeddings Generator**: Vectorizes candidate profiles on local CPU threads.
  * **Multi-Heuristic Scoring**: Combines FAISS semantic search scores with 5 other heuristic scoring filters.
  * **Lexicographical Tie-Breaking**: Sorts final candidate listings by score, breaking ties alphabetically by ID.
* **Suggested Visual**: Unified data flow block diagram tracing: Job Description -> Parser -> Embeddings -> Candidate Streaming -> FAISS Indexing -> Scoring -> Explainable AI -> Export.
* **Icons**: `GitMerge`, `Server`, `Shuffle`, `Database`
* **Layout**: Horizontal architectural pipeline blocks from left to right.
* **Speaker Notes (50s)**:
  > "This diagram details our system architecture. It is built as a modular pipeline to guarantee stability and performance. First, the JD parser extracts required criteria from the job description. The candidate profiles are streamed in chunks from a JSONL database. Candidate texts are encoded in batches to maximize CPU vectorization, then matched against the JD vector using FAISS. The results are scored across multiple signals, sorted with strict alphabetical tie-breaking, and validated programmatically against the official CSV submission format."

---

### Slide 4: AI & Vector Embedding Pipeline [IMPROVED]
* **Slide Title**: Local Semantic Search & Dense Retrieval
* **Key Bullet Points**:
  * **Compact Embeddings Context**: Compresses candidate profile texts to key semantic summaries, preventing model latency.
  * **SentenceTransformers (all-MiniLM-L6-v2)**: Computes dense vectors offline for the JD and candidate profiles.
  * **FAISS Flat Index Search**: Builds an in-memory FAISS flat index to compute exact cosine similarities.
  * **ONNX Runtime Engine**: Runs inference locally on CPU, ensuring low-latency execution.
  * **Pre-Warmed Lifecycle**: Pre-loads model weights on FastAPI startup to prevent cold-start latency.
* **Suggested Visual**: Diagram illustrating a Job Description vector and Candidate vectors mapping to an Index Space, showing cosine distance.
* **Icons**: `Activity`, `Compass`, `Network`
* **Layout**: Left column for technical explanation, right column showing vector search diagram.
* **Speaker Notes (45s)**:
  > "For our AI core, we avoid costly external APIs and run everything locally. We utilize the 'all-MiniLM-L6-v2' SentenceTransformer model loaded via ONNX Runtime for CPU-optimized inference. To make this fast, we compact the candidate profile representation strings to focus strictly on relevant skills and titles. We then build an in-memory FAISS flat index of candidate vectors, calculating the exact cosine similarity against the job description vector in a fraction of a second."

---

### Slide 5: The Multi-Signal Scoring Heuristics [IMPROVED]
* **Slide Title**: Balanced Multi-Signal Candidate Scoring
* **Key Bullet Points**:
  * **Configurable Model Weights**: Scoring weights are fully adjustable via `config.yaml` to suit different recruiter preferences.
  * **Core Signals (65%)**: Semantic match (40%) and fuzzy/exact skill match (25%) adjusted for proficiency and duration.
  * **Experience Sweet-Spot (15%)**: Gaussian decay curve centered around ideal experience years.
  * **Behavioral Activity (10%)**: Weighs profile completeness, response rates, and platform assessments.
  * **Academic & Location Alignment (10%)**: Degree tiers (5%) and geographic proximity/relocation bonus (5%).
* **Suggested Visual**: A clean, modern stacked bar chart or horizontal bar chart representing the weight distribution (40% Blue, 25% Green, 15% Indigo, 10% Purple, 5% Teal, 5% Gray).
* **Icons**: `Percent`, `Sliders`, `UserCheck`, `MapPin`
* **Layout**: Large graph on the left, bulleted weights breakdown on the right.
* **Speaker Notes (55s)**:
  > "Our scoring engine is not limited to semantic similarity alone. It evaluates candidate relevance across six dimensions. We allocate 40% of the weight to semantic matching, and 25% to skills. Experience fit represents 15%, calculated using a Gaussian decay curve to prioritize candidates within the ideal experience range without harshly rejecting minor variations. The remaining 20% evaluates behavioral activity, education quality, and location alignment. Crucially, all these weights are configurable via a central YAML file, letting recruiters adapt the scoring logic in seconds."

---

### Slide 6: Interactive AI Recruiter Dashboard [IMPROVED]
* **Slide Title**: Recruiter Dashboard UI/UX
* **Key Bullet Points**:
  * **Clean SaaS Design Language**: Styled with a clean, premium light slate background, crisp borders, and custom inline SVG icons.
  * **Interactive Specifications**: Simple drag-and-drop job description upload.
  * **Live Stream Tracking**: Visual progress bars powered by server-sent events.
  * **Top Candidates List**: Sortable results showing score breakdowns and recommendation tiers.
  * **Explainable AI Integration**: Clickable candidate rows revealing matching reasoning and interactive radar charts.
* **Suggested Visual**: Grid of clean mock screenshots representing: (1) Upload Panel, (2) Progress Bar, (3) Candidate Grid, and (4) Explainable Radar Chart.
* **Icons**: `DesktopComputer`, `Table`, `Save`, `Download`
* **Layout**: Four-box grid of interface layouts.
* **Speaker Notes (50s)**:
  > "Here is our interactive Recruiter Dashboard, built with React and Tailwind CSS. It is designed using a clean, modern SaaS design language. Recruiters can upload a JD in DOCX format, start the candidate ranking run, and monitor execution progress via a live SSE progress bar. The results are displayed in a clean grid showing candidate details. Clicking any candidate row opens a detailed view showing the exact signal breakdown, a radar chart, and the explainable AI reasoning summary."

---

### Slide 7: Candidate Profile Validation [IMPROVED]
* **Slide Title**: Data Quality & Profile Validation
* **Key Bullet Points**:
  * **Timeline Consistency**: Inspects career histories for overlapping dates or invalid duration records.
  * **Skill Authenticity Verification**: Checks if candidates list expert-level skills but show zero duration or endorsements.
  * **Filter Action**: Disqualifies candidate profiles containing inconsistent data by dropping their score to 0.0.
  * **Confidence Level Scoring**: Assigns High, Medium, or Low confidence based on completeness of signals.
  * **Tie-Breaking Rule**: Guarantees deterministic rankings by sorting ties alphabetically by candidate ID.
* **Suggested Visual**: Flow chart showing a candidate record entering the validation filter, checking dates and profiles, and branching into 'Valid Score' or 'Flagged (Disqualified)'.
* **Icons**: `CheckSquare`, `Info`, `Lock`
* **Layout**: Left column showing validation logic, right column showing validation flow.
* **Speaker Notes (45s)**:
  > "To ensure the system's recommendations are reliable, we built a Candidate Profile Validation layer. It inspects candidate timelines for overlapping dates or inconsistent career histories. It also checks for skill authenticity, making sure candidates who claim high proficiencies have corresponding durations or endorsements. Disqualified candidates are filtered out, and valid candidates are scored with a confidence rating (High, Medium, or Low). Deterministic rankings are guaranteed using strict alphabetical sorting for tied scores."

---

### Slide 8: Performance Optimizations [IMPROVED]
* **Slide Title**: Optimized Engine for CPU Execution
* **Key Bullet Points**:
  * **Memory-Efficient Streaming**: Streams JSONL datasets line-by-line, maintaining a flat memory footprint.
  * **Batch Embeddings encoding**: Groups sentences in batches of 256 to utilize PyTorch vectorization.
  * **Vectorized NumPy Operations**: scoring matrices are processed concurrently using NumPy, avoiding slow python loops.
  * **In-Memory FAISS Index**: Vector comparisons are computed instantly on CPU.
  * **State Restoration**: Saves results to `top100.json` to reload the dashboard in less than 1 second upon server restarts.
* **Suggested Visual**: Bar chart comparing "Un-optimized Python Loops" vs "Optimized Vectorized NumPy + FAISS" execution times.
* **Icons**: `Sparkles`, `Clock`, `Database`
* **Layout**: Modern two-column layout highlighting CPU performance metrics.
* **Speaker Notes (50s)**:
  > "Because this engine runs locally on a CPU, we implemented several performance optimizations. We avoid loading large candidate databases into memory by streaming the JSONL file. We vectorize our embedding calculations in batches of 256, and use NumPy vectorized matrix math for candidate scoring to avoid slow Python loops. We also save our final results to a local JSON file. If the FastAPI server restarts, it restores the memory state instantly, making the web dashboard load immediately."

---

### Slide 9: Technology Stack [NEW]
* **Slide Title**: Localized & Containerized Stack
* **Key Bullet Points**:
  * **Backend**: FastAPI, Uvicorn, sse-starlette.
  * **Data & Math**: NumPy, Pandas, python-docx, PyMuPDF.
  * **AI & Search**: Sentence Transformers (all-MiniLM-L6-v2), FAISS, RapidFuzz.
  * **Frontend**: React, Vite, Tailwind CSS, Recharts.
  * **DevOps**: Docker, docker-compose.
* **Suggested Visual**: Tech logos grouped into cards: Backend, AI & Search, Frontend, and Containerization.
* **Icons**: `Code`, `Terminal`, `Cube`, `Collection`
* **Layout**: Four-quadrant grid mapping categories.
* **Speaker Notes (40s)**:
  > "Our technology stack is completely containerized. The backend runs on FastAPI, which handles API requests and server-sent progress events. Data manipulation is handled by NumPy and Pandas, while Sentence Transformers, FAISS, and RapidFuzz drive our search and scoring. The frontend is built on React, Vite, and Tailwind CSS. The entire application is containerized using Docker, allowing judges or developers to boot the system with a single command."

---

### Slide 10: Workflow [NEW]
* **Slide Title**: Candidate Data Transformation Workflow
* **Key Bullet Points**:
  * **Step 1: Extract**: Parse job description structure and define target variables.
  * **Step 2: Stream**: Read candidate JSONL profiles in chunks from disk.
  * **Step 3: Embed**: Generate dense vector representations for the text.
  * **Step 4: Score**: Calculate all 6 heuristic scores via vectorized math.
  * **Step 5: Rank & Output**: Keep top-100 matches in a min-heap and export.
* **Suggested Visual**: A clean, numbered chevron step workflow diagram (1. Parse -> 2. Stream -> 3. Embed -> 4. Score -> 5. Rank -> 6. Export).
* **Icons**: `Collection`, `TrendingUp`, `ChartBar`, `DocumentDownload`
* **Layout**: Horizontal chevron progress flow.
* **Speaker Notes (45s)**:
  > "Let's walk through the candidate data transformation workflow. It starts with extracting structured variables from the job description. The system then streams candidate records in chunks. For each chunk, the profile data is converted into vectors and matched via FAISS. Next, the scoring engine calculates the heuristic matches. The top 100 candidates are retained in a min-heap, sorted, and exported to a compliant CSV format."

---

### Slide 11: Explainable AI in Practice [NEW]
* **Slide Title**: Transparency Through Explainable AI
* **Key Bullet Points**:
  * **Candidate Score Example**: CAND_0071974 (Match Score: 70.1%)
  * **Skill Gap Analysis**: Matched: [sentence-transformers, OpenAI embeddings, Embeddings] | Missing: [BGE, E5, FAISS]
  * **Heuristics Summary**: 7.8 years experience | Platform activity: High (GitHub: 83%) | Location: Noida (Match)
  * **Auto-Generated Reasoning**: `"Matched [sentence-transformers, OpenAI embeddings] | Missing [BGE, E5] | 7.8 yrs exp | Location: Same Country | Score: 70.1%"`
  * **Recommendation Tier**: Highly Recommended
* **Suggested Visual**: Mock card representing a candidate profile. Left: Core metrics, Right: Radar chart showing the 6 signal scores, Bottom: Text reasoning box.
* **Icons**: `Eye`, `BookOpen`, `Heart`
* **Layout**: Left column for details, right column for radar chart.
* **Speaker Notes (50s)**:
  > "Transparency is a key focus of our engine. Here is a real candidate profile ranked by the system. Candidate CAND_0071974 achieved a final score of 70.1%. The engine details which skills matched the JD and which ones were missing. It notes that the candidate has 7.8 years of experience, high platform activity, and is in a matching location. This is summarized in a clean, human-readable text explanation, giving recruiters clear context on why the candidate was ranked at the top."

---

### Slide 12: Results & Verification [NEW]
* **Slide Title**: Submission Compliance & Results
* **Key Bullet Points**:
  * **Submission Format**: Exports a 4-column CSV file (`candidate_id, rank, score, reasoning`).
  * **Programmatic Validation**: Automatically invokes `validate_submission.py` upon exporting results.
  * **Compliance Result**: `Submission validation succeeded! CSV is 100% compliant.`
  * **Data Integrity**: Exactly 100 candidate rows, sorted with tie-breakers.
* **Suggested Visual**: Split view: Left showing the top rows of `submission.csv`, Right showing the green console message of the successful validator script.
* **Icons**: `DocumentReport`, `BadgeCheck`, `Terminal`
* **Layout**: Two-column layout showcasing CSV contents and validation log.
* **Speaker Notes (45s)**:
  > "We verified our engine's output against the hackathon rules. The system exports a CSV containing exactly 100 candidate records. To guarantee compliance, the exporter automatically triggers the official validation script. The results are fully compliant, verifying that the candidate IDs are valid, the ranks are correct, and identical scores are sorted alphabetically. The system is ready to output compliant submissions."

---

### Slide 13: Future Scope [IMPROVED]
* **Slide Title**: Engine Roadmap & Future Extensions
* **Key Bullet Points**:
  * **Resume PDF Parsing**: Support direct parsing of raw PDF and Word candidate files.
  * **LLM-Based Candidate Summaries**: Integrate a local LLM to generate rich candidate profiles and summaries.
  * **Multi-JD Ranking**: Support matching and ranking candidates against multiple JDs concurrently.
  * **Cloud Deployment**: Scale search and scoring services using Kubernetes and distributed memory databases.
  * **Recruiter Feedback Learning**: Dynamically update signal weights based on user choice history and rejection/acceptance actions.
* **Suggested Visual**: Timeline arrow with nodes indicating PDF Parsing -> LLM Summaries -> Multi-JD -> Recruiter Feedback.
* **Icons**: `Calendar`, `CloudUpload`, `Adjustments`
* **Layout**: Clean timeline flow chart.
* **Speaker Notes (50s)**:
  > "Looking ahead, we have planned several extensions for our engine. We want to support PDF and Word document parsing for candidates, so recruiters can upload raw resumes. We plan to integrate a local LLM to generate rich candidate profile summaries, and support multi-JD matching. We will also scale the search using Kubernetes, and implement a feedback loop that updates signal weights based on recruiter actions. Thank you, and we are ready for your questions."
