"""Command-line entry point for the complete research pipeline."""

import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.types import Command

from drafting import DraftingAgent
from export import ExportAgent, ExportConfig
from extraction import NoteExtractionAgent
from gap_analysis import GapAnalysisAgent
from outline import OutlineGenerator
from revision import RevisionAgent
from retrieval import LiteratureRetrievalAgent
from scope import ScopingAgent
from verification import VerificationAgent, build_research_graph


def build_pipeline_agents(llm):
    """Create the concrete agents used by the nine-stage graph."""
    corpus_db_path = os.getenv("CORPUS_DB_PATH", "corpus.db")
    extraction_db_path = os.getenv("EXTRACTION_DB_PATH", "extraction.db")
    email = os.getenv("RESEARCH_EMAIL") or "research-agent@localhost"

    verification_agent = VerificationAgent(
        llm=llm,
        corpus_db_path=corpus_db_path,
        extraction_db_path=extraction_db_path,
    )

    return {
        "stage1": ScopingAgent(
            llm=llm,
            scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            auto_approve=True,
        ),
        "stage2": LiteratureRetrievalAgent(
            llm=llm,
            email=email,
            scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            corpus_db_path=corpus_db_path,
            pdf_dir=os.getenv("PDF_DIR", "corpus/pdfs"),
        ),
        "stage3": NoteExtractionAgent(
            llm=llm,
            db_path=extraction_db_path,
            use_grobid=os.getenv("USE_GROBID", "").lower() == "true",
            grobid_url=os.getenv("GROBID_URL", "http://localhost:8070"),
        ),
        "stage4": GapAnalysisAgent(
            llm=llm,
            extraction_db_path=extraction_db_path,
            corpus_db_path=corpus_db_path,
        ),
        "stage5": OutlineGenerator(
            llm=llm,
            extraction_db_path=extraction_db_path,
        ),
        "stage6": DraftingAgent(
            llm=llm,
            extraction_db_path=extraction_db_path,
        ),
        "stage7": verification_agent,
        "stage8": RevisionAgent(
            llm=llm,
            corpus_db_path=corpus_db_path,
            extraction_db_path=extraction_db_path,
            verification_agent=verification_agent,
        ),
        "stage9": ExportAgent(corpus_db_path=corpus_db_path),
    }


def _create_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY before running the research pipeline.")

    return ChatOpenAI(
        model=os.getenv(
            "RESEARCH_MODEL",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        ),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )


def _resume_checkpoint(result: dict) -> dict:
    """Display a checkpoint and collect the decision used to resume the graph."""
    checkpoint = result["__interrupt__"][0].value
    stage = checkpoint["stage"]
    options = checkpoint["options"]
    print(f"\nStage {stage} is awaiting human approval.")
    print("Options: " + ", ".join(options))

    while True:
        decision = input("Decision: ").strip().lower()
        if decision in options:
            if decision == "reject":
                raise SystemExit(f"Pipeline rejected after Stage {stage}.")
            return {"decision": decision}
        print("Choose one of: " + ", ".join(options))


def run_pipeline(topic: str) -> dict:
    """Run the pipeline, pausing and resuming at each human checkpoint."""
    llm = _create_llm()
    agents = build_pipeline_agents(llm)
    graph = build_research_graph(agents)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic": topic,
        "paper_type": os.getenv("PAPER_TYPE", "survey"),
        "venue": os.getenv("TARGET_VENUE"),
        "page_limit": int(os.getenv("PAGE_LIMIT", "0")) or None,
        "deadline": os.getenv("DEADLINE"),
        "top_n_gaps": int(os.getenv("TOP_N_GAPS", "3")),
        "total_word_limit": int(os.getenv("TOTAL_WORD_LIMIT", "8000")),
        "max_revision_rounds": int(os.getenv("MAX_REVISION_ROUNDS", "3")),
        "max_fix_loops": int(os.getenv("MAX_FIX_LOOPS", "2")),
        "output_dir": os.getenv("OUTPUT_DIR", "export"),
        "export_config": ExportConfig(
            venue_template=os.getenv("VENUE_TEMPLATE", "generic"),
            title=os.getenv("PAPER_TITLE") or None,
            authors=[
                author.strip()
                for author in os.getenv("AUTHORS", "").split(",")
                if author.strip()
            ],
            corresponding_email=os.getenv("CORRESPONDING_EMAIL") or None,
            generate_docx=os.getenv("GENERATE_DOCX", "").lower() == "true",
        ),
    }

    result = graph.invoke(initial_state, config=config)
    while result.get("__interrupt__"):
        result = graph.invoke(
            Command(resume=_resume_checkpoint(result)),
            config=config,
        )

    print("\nResearch pipeline completed.")
    export_report = result.get("export_report")
    if export_report is not None:
        print(f"PDF: {export_report.pdf_path}")
        print(f"Ready to submit: {export_report.ready_to_submit}")
    return result


if __name__ == "__main__":
    requested_topic = input("What should the research paper investigate? ").strip()
    if not requested_topic:
        raise SystemExit("A research topic is required.")
    try:
        run_pipeline(requested_topic)
    except RuntimeError as exc:
        message = str(exc)
        if "OpenRouter rate limit reached" in message:
            raise SystemExit(
                f"\n{message}\n"
                "To continue, set RESEARCH_MODEL in .env to a model your "
                "OpenRouter account can use, or wait for the quota reset."
            ) from None
        raise
