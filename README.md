# AI Research Paper Pipeline

An evidence-aware, human-in-the-loop pipeline for turning a research question into a verified, submission-ready paper package.

The application coordinates nine stages with [LangGraph](https://langchain-ai.github.io/langgraph/), uses an OpenRouter-compatible chat model for language-model tasks, stores research state in SQLite, and exports LaTeX/PDF/BibTeX artifacts. It is designed for survey and literature-based research workflows, not for unsupervised publication.

> **Important:** Generated text and citations still require expert review. The verification stage is a quality gate, not a guarantee of factual correctness, originality, or acceptance by a venue.

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Pipeline stages](#pipeline-stages)
- [Control flow and approval gates](#control-flow-and-approval-gates)
- [Data flow and storage](#data-flow-and-storage)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Project structure](#project-structure)
- [Development and validation](#development-and-validation)
- [Troubleshooting](#troubleshooting)
- [Security and responsible use](#security-and-responsible-use)
- [Limitations and roadmap](#limitations-and-roadmap)
- [License](#license)

## What it does

Given a topic, the pipeline:

1. Defines a research scope and screening criteria.
2. Searches for relevant literature and builds a local corpus.
3. Extracts structured notes and evidence from sources.
4. Identifies research gaps.
5. Proposes a paper outline and contribution map.
6. Drafts the paper from the approved research state.
7. Audits claims, citations, numbers, attribution, consistency, overlap, and placeholders.
8. Revises failed or weak sections through bounded fix loops.
9. Produces a formatted export package after final approval.

The stages are connected by explicit state transitions. Human approval gates allow the researcher to inspect results before downstream work begins.

## Architecture

### System overview

```mermaid
flowchart LR
    User[Researcher] --> CLI[main.py CLI]
    CLI --> Graph[LangGraph workflow]
    Graph --> Agents[Pipeline agents]
    Agents --> LLM[OpenRouter chat model]
    Agents --> Sources[Web and scholarly sources]
    Agents --> Corpus[(corpus.db)]
    Agents --> Notes[(extraction.db)]
    Graph --> Checkpoints[Human approval checkpoints]
    Graph --> Export[ExportAgent]
    Export --> Files[export/ PDF, LaTeX, BibTeX, bundle]
    Config[.env] -. configuration .-> CLI
    Config -. credentials .-> LLM
```

### Runtime interaction

```mermaid
sequenceDiagram
    actor Researcher
    participant CLI as main.py
    participant Graph as LangGraph
    participant Model as OpenRouter model
    participant DB as SQLite stores
    participant Export as ExportAgent

    Researcher->>CLI: Enter research topic
    CLI->>Model: Create configured chat client
    CLI->>Graph: Invoke initial state
    loop Stages 1-5
        Graph->>Model: Generate stage result
        Graph->>DB: Persist corpus or extracted state
        Graph-->>CLI: Interrupt for approval
        CLI-->>Researcher: Display options
        Researcher->>CLI: approve, rerun, or reject
        CLI->>Graph: Resume with decision
    end
    Graph->>Model: Draft and verify
    Graph->>Graph: Route to revision when verification fails
    Graph->>Export: Build approved export
    Export->>DB: Read verified content and citations
    Export-->>CLI: Return export report
    CLI-->>Researcher: Show PDF and submission status
```

### Component responsibilities

| Component | Responsibility |
| --- | --- |
| `main.py` | Loads configuration, creates the model and agents, starts/resumes the graph |
| LangGraph graph | Orchestrates stages, interrupts for approval, and routes verification/revision loops |
| Stage agents | Perform scoping, retrieval, extraction, analysis, drafting, revision, verification, and export |
| OpenRouter | Provides the configured chat model through an OpenAI-compatible endpoint |
| `corpus.db` | Stores source metadata, screening state, and corpus records |
| `extraction.db` | Stores extracted notes and evidence used by analysis and drafting |
| Export layer | Performs preflight checks and creates submission artifacts without intentionally changing verified content |

## Pipeline stages

```mermaid
flowchart TD
    Start([Research topic]) --> S1[1. Scope]
    S1 --> G1{Approve scope}
    G1 -->|approve| S2[2. Retrieve literature]
    G1 -->|rerun| S1
    G1 -->|reject| Stop([Stop])
    S2 --> G2{Approve corpus}
    G2 -->|approve| S3[3. Extract notes]
    G2 -->|rerun| S2
    G2 -->|reject| Stop
    S3 --> G3{Approve extraction}
    G3 -->|approve| S4[4. Analyze gaps]
    G3 -->|rerun| S3
    G3 -->|reject| Stop
    S4 --> G4{Approve gap analysis}
    G4 -->|approve| S5[5. Generate outline]
    G4 -->|rerun| S4
    G4 -->|reject| Stop
    S5 --> G5{Approve outline}
    G5 -->|approve| S6[6. Draft paper]
    G5 -->|rerun| S5
    G5 -->|reject| Stop
    S6 --> S7[7. Verify draft]
    S7 --> V{Verification gate}
    V -->|passed| S9[9. Export]
    V -->|failed| S8[8. Revise and fix]
    S8 --> G8{Review revision route}
    G8 -->|stage2| S2
    G8 -->|stage7| S7
    G8 -->|stage9| S9
    S9 --> G9{Approve export}
    G9 -->|approve| Done([Submission package])
    G9 -->|rerun| S9
    G9 -->|reject| Stop
```

### Stage details

| Stage | Module | Inputs | Main result |
| ---: | --- | --- | --- |
| 1 | `scope.py` | Topic, paper type, venue, deadline | Scope, research questions, inclusion/exclusion criteria |
| 2 | `retrieval.py` | Approved scope and search criteria | Screened literature corpus and source metadata |
| 3 | `extraction.py` | Corpus PDFs and metadata | Structured notes, quotations, methods, results, and limitations |
| 4 | `gap_analysis.py` | Extracted notes and corpus | Candidate gaps ranked for relevance and evidence |
| 5 | `outline.py` | Approved gaps and notes | Section plan, claims, contributions, and source mapping |
| 6 | `drafting.py` | Outline and extracted evidence | Draft sections with citation keys |
| 7 | `verification.py` | Draft, outline, corpus, notes | `VerificationReport` with verdicts, issues, pass rate, and fix list |
| 8 | `revision.py` | Draft and verification report | Revised draft and a route back to verification or earlier stages |
| 9 | `export.py` | Approved revised draft and corpus | Preflight report and formatted output package |

## Control flow and approval gates

Stages 1–5 and stage 9 pause for an explicit decision. The CLI displays the available options and accepts a decision at the prompt:

```text
approve
rerun
reject
```

Stage 8 can route the workflow to:

```text
approve  -> export
stage2   -> retrieve literature again
stage7   -> verify the revised draft again
stage9   -> export again
```

Verification is intentionally stricter than a simple model review. The implementation checks:

- Citation existence and screening status
- DOI metadata and title/year consistency where available
- Sentence-level support and supporting quotations
- Citation attribution and possible citation swaps
- Hedging calibration for weak evidence
- Numeric and cross-section consistency
- Contribution delivery
- New citations introduced in conclusions
- N-gram overlap with source text
- Remaining placeholders or unsupported first-person results

The report records supported, partial, unsupported, and contradicted sentence verdicts, plus a gate status and actionable fixes.

## Data flow and storage

```mermaid
flowchart TB
    Topic[Topic and .env settings] --> Scope[Scope state]
    Scope --> CorpusDB[(corpus.db)]
    CorpusDB --> Retrieval[Retrieval and screening]
    Retrieval --> PDFs[corpus/pdfs/]
    PDFs --> Extract[PDF and optional GROBID extraction]
    Extract --> NotesDB[(extraction.db)]
    NotesDB --> Gap[Gap analysis]
    NotesDB --> Outline[Outline]
    Gap --> Outline
    Outline --> Draft[Draft]
    Draft --> Verify[Verification report]
    CorpusDB --> Verify
    NotesDB --> Verify
    Verify --> Revise[Revision loop]
    Revise --> Verify
    Verify --> Export[Export preflight]
    CorpusDB --> Export
    Export --> Output[export/]
```

### Local data policy

The following are runtime artifacts and are ignored by Git:

- `corpus.db`
- `extraction.db`
- `revision_versions.db`
- `.cache/`
- `corpus/pdfs/`
- `export/`
- `venv/`

Deleting these files can remove local research state, downloaded papers, or generated outputs. Back them up separately if they are valuable.

## Requirements

- Python 3.10 or newer
- Windows PowerShell, macOS/Linux shell, or an equivalent Python environment
- An [OpenRouter API key](https://openrouter.ai/)
- Internet access for model calls and literature/metadata services
- Optional: a [Semantic Scholar API key](https://www.semanticscholar.org/product/api)
- Optional: a local [GROBID](https://github.com/kermitt2/grobid) server for enhanced PDF extraction
- Optional: a working LaTeX installation if the export implementation compiles LaTeX locally

## Installation

### Windows PowerShell

```powershell
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, either allow scripts for the current user or invoke the interpreter directly:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` before starting the pipeline. Never commit `.env`.

## Configuration

All settings are read from environment variables. The complete starter configuration is in [.env.example](.env.example).

### Model and service settings

| Variable | Default | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | required | Credential for the configured chat model |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible model endpoint |
| `RESEARCH_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter model identifier |
| `RESEARCH_EMAIL` | `research-agent@localhost` in code | Contact value used by retrieval services |
| `SEMANTIC_SCHOLAR_API_KEY` | empty | Optional Semantic Scholar credential |
| `USE_GROBID` | `false` | Enable GROBID-backed extraction |
| `GROBID_URL` | `http://localhost:8070` | GROBID service URL |

### Storage settings

| Variable | Default | Description |
| --- | --- | --- |
| `CORPUS_DB_PATH` | `corpus.db` | SQLite corpus database |
| `EXTRACTION_DB_PATH` | `extraction.db` | SQLite extraction database |
| `PDF_DIR` | `corpus/pdfs` | Downloaded/source PDF directory |
| `OUTPUT_DIR` | `export` | Generated output directory |

### Research and export settings

| Variable | Default | Description |
| --- | --- | --- |
| `PAPER_TYPE` | `survey` | Paper type passed into the pipeline |
| `TARGET_VENUE` | empty | Target venue or publication context |
| `PAGE_LIMIT` | `0` | Page limit; zero means no configured limit |
| `DEADLINE` | empty | Optional project deadline |
| `TOP_N_GAPS` | `3` | Number of top gaps to carry forward |
| `TOTAL_WORD_LIMIT` | `8000` | Draft word budget |
| `MAX_REVISION_ROUNDS` | `3` | Maximum revision rounds |
| `MAX_FIX_LOOPS` | `2` | Maximum verification/fix loops |
| `VENUE_TEMPLATE` | `generic` | Export template, such as `ieee`, `acm`, `springer_lncs`, `elsevier`, `arxiv`, or `generic` |
| `PAPER_TITLE` | empty | Optional title |
| `AUTHORS` | empty | Comma-separated author names |
| `CORRESPONDING_EMAIL` | empty | Optional corresponding-author email |
| `GENERATE_DOCX` | `false` | Request DOCX generation |

## Running the pipeline

Start the interactive workflow:

```powershell
python main.py
```

Then enter a research topic:

```text
What should the research paper investigate? How do retrieval-augmented language models reduce factual errors?
```

At each approval checkpoint, inspect the displayed stage result and enter one of the allowed decisions. The process creates a new in-memory workflow thread for each run and resumes interrupted graph execution using LangGraph commands.

### Example `.env`

```dotenv
OPENROUTER_API_KEY=replace-with-a-new-key
RESEARCH_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
PAPER_TYPE=survey
TARGET_VENUE=arxiv
TOTAL_WORD_LIMIT=6000
MAX_REVISION_ROUNDS=2
GENERATE_DOCX=false
```

Do not copy real credentials into documentation, issue reports, screenshots, or commits.

## Outputs

The export stage is designed to produce a submission package containing some or all of the following, depending on available data and local tooling:

- Compiled PDF
- LaTeX source
- BibTeX bibliography
- Supplementary/reproducibility bundle
- Optional DOCX
- Export and compilation reports

Export can be blocked by preflight issues such as missing citations, unresolved references, missing figures, or compilation failures. Treat `ready_to_submit` as an application status, not a substitute for a final author review.

## Project structure

```text
.
├── main.py                 # CLI entry point and graph execution
├── scope.py                # Stage 1: scope definition
├── retrieval.py            # Stage 2: literature retrieval and screening
├── extraction.py            # Stage 3: source/note extraction
├── gap_analysis.py         # Stage 4: gap analysis
├── outline.py              # Stage 5: outline generation
├── drafting.py             # Stage 6: draft generation
├── verification.py         # Stage 7: evidence and quality gate
├── revision.py             # Stage 8: revision and fix loops
├── export.py               # Stage 9: formatting and export
├── tools.py                # Shared helpers
├── requirements.txt        # Python dependencies
├── .env.example            # Safe configuration template
├── .gitignore              # Secret and runtime-artifact exclusions
├── corpus.db               # Local runtime database; ignored
├── extraction.db           # Local runtime database; ignored
├── corpus/pdfs/            # Local source PDFs; ignored
└── export/                 # Generated artifacts; ignored
```

## License

No license has been selected for this project yet. Add a license before publishing if you want others to use, modify, or redistribute the code.
