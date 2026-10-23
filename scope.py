"""
Stage 1: Scoping Module

Turns a vague topic into a precise, checkable research question with boundaries.
Implements the scoping pipeline with human checkpoint approval.
"""

import requests
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContributionType(str, Enum):
    SURVEY = "survey"
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    POSITION = "position"


class ResearchQuestionCandidate(BaseModel):
    """A candidate research question with scoring."""
    question: str = Field(..., description="The research question")
    rationale: str = Field(..., description="One-line rationale for this question")
    novelty_score: int = Field(default=0, ge=1, le=5, description="Novelty score (1-5)")
    feasibility_score: int = Field(default=0, ge=1, le=5, description="Feasibility score (1-5)")
    answerability_score: int = Field(default=0, ge=1, le=5, description="Answerability score (1-5)")
    scope_fit_score: int = Field(default=0, ge=1, le=5, description="Scope fit score (1-5)")
    is_already_answered: bool = Field(default=False, description="Flagged by novelty check")
    similarity_score: Optional[float] = Field(default=None, description="Cosine similarity from novelty check")


class ScopeDocument(BaseModel):
    """The validated output of Stage 1: Scoping."""
    # Input
    original_topic: str
    paper_type: Optional[str] = None
    target_venue: Optional[str] = None
    page_limit: Optional[int] = None
    deadline: Optional[str] = None
    year_range: tuple[int, int] = Field(default=(2015, 2026), description="Year range for sources")

    # Decomposition
    subtopics: list[str] = Field(default_factory=list, description="3-6 subtopics extracted from landscape probe")
    keywords: list[str] = Field(default_factory=list, description="Keywords and synonyms for search")
    common_methods: list[str] = Field(default_factory=list, description="Common methods found in probe")
    common_datasets: list[str] = Field(default_factory=list, description="Common datasets found in probe")

    # Selected question
    research_question: str
    contribution_type: ContributionType
    question_rationale: str

    # Boundaries
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default=["English"])
    venue_types: list[str] = Field(default_factory=list)

    # Success criteria
    hypothesis: Optional[str] = Field(default=None, description="For empirical papers")
    required_coverage: list[str] = Field(default_factory=list, description="For surveys: taxonomy, comparison axes")

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved: bool = False
    approval_notes: Optional[str] = None


class SemanticScholarClient:
    """Wrapper for Semantic Scholar API searches."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def search(
        self,
        query: str,
        year_range: Optional[tuple[int, int]] = None,
        limit: int = 50,
        fields: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Search Semantic Scholar for papers.

        Args:
            query: Search query string
            year_range: Optional (start_year, end_year) tuple
            limit: Maximum number of results (default 50)
            fields: Fields to return (default: title, abstract, year, venue, authors)

        Returns:
            List of paper dictionaries
        """
        if fields is None:
            fields = ["title", "abstract", "year", "venue", "authors", "citationCount"]

        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields)
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            print(f"Semantic Scholar API error: {e}")
            return []

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """Get a single paper by ID."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={"fields": "title,abstract,year,venue,authors,citationCount,embedding"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None


class LandscapeProbe:
    """Performs landscape probe to ground scoping in actual literature."""

    def __init__(self, client: Optional[SemanticScholarClient] = None):
        self.client = client or SemanticScholarClient()

    def probe(
        self,
        topic: str,
        year_range: tuple[int, int] = (2015, 2026),
        num_queries: int = 3
    ) -> list[dict]:
        """
        Run broad searches to understand the landscape.

        Args:
            topic: Raw topic string
            year_range: Year range for papers
            num_queries: Number of query variations to try

        Returns:
            Combined list of papers from all queries
        """
        # Generate query variations
        queries = [
            topic,
            f"{topic} survey review",
            f"{topic} recent advances"
        ][:num_queries]

        all_papers = []
        seen_ids = set()

        for query in queries:
            papers = self.client.search(query, year_range=year_range, limit=50)
            for paper in papers:
                paper_id = paper.get("paperId")
                if paper_id and paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    all_papers.append(paper)

        return all_papers

    def probe_openalex(
        self,
        topic: str,
        year_range: tuple[int, int],
        limit: int = 50,
    ) -> list[dict]:
        """Fallback academic search when Semantic Scholar is rate-limited."""
        params = {
            "search": topic,
            "filter": (
                f"from_publication_date:{year_range[0]}-01-01,"
                f"to_publication_date:{year_range[1]}-12-31,"
                "type:article"
            ),
            "per-page": limit,
        }
        try:
            response = requests.get(
                "https://api.openalex.org/works",
                params=params,
                headers={"User-Agent": "Research-Agent/1.0"},
                timeout=30,
            )
            response.raise_for_status()
            records = response.json().get("results", [])
        except (requests.RequestException, ValueError) as exc:
            print(f"OpenAlex fallback error: {exc}")
            return []

        papers = []
        for record in records:
            abstract_index = record.get("abstract_inverted_index") or {}
            abstract_tokens = []
            for word, positions in abstract_index.items():
                if isinstance(positions, list):
                    abstract_tokens.extend(
                        (position, str(word))
                        for position in positions
                        if isinstance(position, int)
                    )
            doi = record.get("doi")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi.removeprefix("https://doi.org/")
            papers.append({
                "paperId": record.get("id", ""),
                "title": record.get("title", ""),
                "abstract": " ".join(
                    word for _, word in sorted(abstract_tokens)
                ),
                "year": record.get("publication_year"),
                "venue": (
                    (record.get("primary_location") or {})
                    .get("source") or {}
                ).get("display_name"),
                "authors": [
                    (authorship.get("author") or {}).get("display_name", "")
                    for authorship in record.get("authorships", [])
                    if (authorship.get("author") or {}).get("display_name")
                ],
                "doi": doi,
                "externalIds": {"DOI": doi} if doi else {},
            })
        return papers

    def extract_subtopics(self, papers: list[dict], min_count: int = 3) -> list[str]:
        """
        Extract subtopics from paper titles and abstracts.

        Simple keyword-based extraction (can be enhanced with NLP).
        """
        # Collect all text
        text_blocks = []
        for paper in papers[:30]:
            if paper.get("title"):
                text_blocks.append(paper["title"])
            if paper.get("abstract"):
                text_blocks.append(paper["abstract"])

        # Simple subtopic extraction based on common phrases
        # In production, you'd use topic modeling or LLM extraction
        subtopics = []
        common_patterns = [
            "personalized learning",
            "intelligent tutoring",
            "automated assessment",
            "learning analytics",
            "adaptive systems",
            "natural language processing",
            "question answering",
            "content generation",
            "student modeling",
            "educational chatbots"
        ]

        combined_text = " ".join(text_blocks).lower()
        for pattern in common_patterns:
            if pattern in combined_text:
                subtopics.append(pattern.title())

        return subtopics[:6] if subtopics else ["General Applications", "Methodology", "Evaluation"]

    def extract_methods_and_datasets(
        self,
        papers: list[dict]
    ) -> tuple[list[str], list[str]]:
        """Extract common methods and datasets from papers."""
        methods = set()
        datasets = set()

        common_methods = [
            "fine-tuning", "prompt engineering", "RAG", "reinforcement learning",
            "transfer learning", "few-shot learning", "zero-shot learning"
        ]

        common_datasets = [
            "MATH", "GSM8K", "ARC", "SQuAD", "MMLU", "HumanEval",
            "Student essays", "Educational logs"
        ]

        combined_text = " ".join([
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in papers[:30]
        ]).lower()

        for method in common_methods:
            if method.lower() in combined_text:
                methods.add(method.title())

        for dataset in common_datasets:
            if dataset.lower() in combined_text:
                datasets.add(dataset)

        return list(methods), list(datasets)


class NoveltyChecker:
    """Checks if a research question has already been answered."""

    def __init__(
        self,
        scholar_client: Optional[SemanticScholarClient] = None,
        embedding_client = None  # Placeholder for embedding model
    ):
        self.scholar = scholar_client or SemanticScholarClient()
        self.embedding_client = embedding_client
        self.similarity_threshold = 0.85

    def check_novelty(
        self,
        question: str,
        year_range: tuple[int, int] = (2015, 2026)
    ) -> tuple[bool, Optional[float]]:
        """
        Check if a research question has already been answered.

        Args:
            question: The research question to check
            year_range: Year range for papers

        Returns:
            Tuple of (is_already_answered, max_similarity_score)
        """
        # Search for papers matching the question
        papers = self.scholar.search(question, year_range=year_range, limit=10)

        if not papers:
            return False, None

        if not self.embedding_client:
            # Fallback: simple keyword overlap check
            question_words = set(question.lower().split())
            max_overlap = 0

            for paper in papers[:5]:
                if not paper.get("abstract"):
                    continue
                abstract_words = set(paper["abstract"].lower().split())
                overlap = len(question_words & abstract_words) / len(question_words)
                max_overlap = max(max_overlap, overlap)

            # If > 70% keyword overlap, likely answered
            return max_overlap > 0.7, max_overlap

        # With embeddings: compute cosine similarity
        # This would require an embedding model (e.g., sentence-transformers)
        # Placeholder for actual implementation
        return False, None


class ScopingAgent:
    """
    Main agent for Stage 1: Scoping.

    Orchestrates the scoping pipeline and produces a validated ScopeDocument.
    """

    def __init__(
        self,
        llm,  # LangChain LLM instance
        scholar_api_key: Optional[str] = None,
        auto_approve: bool = False
    ):
        self.llm = llm
        self.scholar_client = SemanticScholarClient(api_key=scholar_api_key)
        self.probe = LandscapeProbe(self.scholar_client)
        self.novelty_checker = NoveltyChecker(self.scholar_client)
        self.auto_approve = auto_approve

    def parse_constraints(
        self,
        topic: str,
        paper_type: Optional[str] = None,
        venue: Optional[str] = None,
        page_limit: Optional[int] = None,
        deadline: Optional[str] = None,
        year_range: Optional[tuple[int, int]] = None
    ) -> dict:
        """Step 1: Parse and validate constraints."""
        return {
            "original_topic": topic,
            "paper_type": paper_type or "survey",
            "target_venue": venue,
            "page_limit": page_limit,
            "deadline": deadline,
            "year_range": year_range or (2015, datetime.now().year)
        }

    def run_landscape_probe(
        self,
        topic: str,
        year_range: tuple[int, int]
    ) -> tuple[list[dict], dict]:
        """
        Step 2: Run landscape probe to ground scoping in actual literature.

        Returns:
            Tuple of (papers, decomposition_results)
        """
        print(f"[Stage 1] Running landscape probe for: {topic}")
        papers = self.probe.probe(topic, year_range)
        if not papers:
            print("[Stage 1] Semantic Scholar unavailable; trying OpenAlex...")
            papers = self.probe.probe_openalex(topic, year_range)
        print(f"[Stage 1] Found {len(papers)} papers")
        if not papers:
            raise RuntimeError(
                "Stage 1 could not retrieve academic landscape papers from "
                "Semantic Scholar or OpenAlex. Check network access, wait for "
                "rate limits to reset, or set SEMANTIC_SCHOLAR_API_KEY in .env."
            )

        # Step 3: Decompose topic
        subtopics = self.probe.extract_subtopics(papers)
        keywords = self._extract_keywords(topic, papers)
        methods, datasets = self.probe.extract_methods_and_datasets(papers)

        decomposition = {
            "subtopics": subtopics,
            "keywords": keywords,
            "common_methods": methods,
            "common_datasets": datasets
        }

        return papers, decomposition

    def _extract_keywords(self, topic: str, papers: list[dict]) -> list[str]:
        """Extract keywords and synonyms from topic and papers."""
        keywords = [topic]

        # Add topic variations
        topic_lower = topic.lower()
        if "llm" in topic_lower:
            keywords.extend(["large language model", "GPT", "language model"])
        if "education" in topic_lower:
            keywords.extend(["educational", "learning", "teaching"])

        # Add venue names from papers
        venues = set()
        for paper in papers[:20]:
            if paper.get("venue"):
                venues.add(paper["venue"])

        keywords.extend(list(venues)[:5])
        return keywords

    def generate_candidates(
        self,
        constraints: dict,
        decomposition: dict
    ) -> list[ResearchQuestionCandidate]:
        """
        Steps 4-5: Generate and score candidate research questions.

        Uses LLM to generate candidates and score them.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        class CandidateList(BaseModel):
            candidates: list[ResearchQuestionCandidate]

        parser = PydanticOutputParser(pydantic_object=CandidateList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research advisor. Generate 5 candidate research questions
for the given topic, based on the landscape analysis provided.

For each question:
1. Provide a clear, specific research question
2. Give a one-line rationale
3. Score each criterion (1-5) based on:
   - Novelty: Has this exact question been answered? (5 = completely novel)
   - Feasibility: Can this be answered with available resources? (5 = highly feasible)
   - Answerability: Will enough sources exist? (5 = many sources available)
   - Scope fit: Not too broad, not too narrow (5 = perfect scope)

Consider the subtopics and common methods identified.

{format_instructions}"""),
            ("human", """Topic: {topic}

Subtopics found: {subtopics}
Common methods: {methods}
Paper type: {paper_type}

Generate 5 candidate research questions with scores.""")
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "topic": constraints["original_topic"],
                "subtopics": ", ".join(decomposition["subtopics"]),
                "methods": ", ".join(decomposition["common_methods"]) or "Various methods",
                "paper_type": constraints["paper_type"]
            })
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429 or "Rate limit exceeded" in str(exc):
                raise RuntimeError(
                    "OpenRouter rate limit reached during Stage 1 scoping. "
                    "Wait for the quota reset, add credits, or set RESEARCH_MODEL "
                    "to a model available on your account."
                ) from exc
            raise

        return result.candidates

    def check_candidate_novelty(
        self,
        candidates: list[ResearchQuestionCandidate],
        year_range: tuple[int, int]
    ) -> list[ResearchQuestionCandidate]:
        """
        Step 6: Automated novelty check for each candidate.
        """
        print(f"[Stage 1] Checking novelty for {len(candidates)} candidates...")

        for candidate in candidates:
            is_answered, similarity = self.novelty_checker.check_novelty(
                candidate.question,
                year_range
            )
            candidate.is_already_answered = is_answered
            candidate.similarity_score = similarity

        return candidates

    def select_question(
        self,
        candidates: list[ResearchQuestionCandidate]
    ) -> tuple[ResearchQuestionCandidate, ContributionType]:
        """
        Step 7: Select the top-scored question and contribution type.
        """
        # Filter out already-answered questions
        valid_candidates = [c for c in candidates if not c.is_already_answered]

        if not valid_candidates:
            print("[Stage 1] Warning: All candidates flagged as already answered")
            valid_candidates = candidates  # Fall back to all candidates

        # Score by weighted sum
        def total_score(c: ResearchQuestionCandidate) -> float:
            return (
                c.novelty_score * 0.3 +
                c.feasibility_score * 0.25 +
                c.answerability_score * 0.25 +
                c.scope_fit_score * 0.2
            )

        best = max(valid_candidates, key=total_score)

        # Determine contribution type based on question nature
        question_lower = best.question.lower()
        if any(word in question_lower for word in ["survey", "review", "taxonomy", "landscape"]):
            contribution_type = ContributionType.SURVEY
        elif any(word in question_lower for word in ["experiment", "empirical", "evaluate", "measure"]):
            contribution_type = ContributionType.EMPIRICAL
        elif any(word in question_lower for word in ["theory", "framework", "model"]):
            contribution_type = ContributionType.THEORETICAL
        else:
            contribution_type = ContributionType.POSITION

        return best, contribution_type

    def define_boundaries(
        self,
        constraints: dict,
        decomposition: dict,
        contribution_type: ContributionType
    ) -> dict:
        """
        Step 8: Define inclusion/exclusion criteria and boundaries.
        """
        inclusion = [
            "Peer-reviewed publications",
            f"Year range: {constraints['year_range'][0]}-{constraints['year_range'][1]}",
            "English language publications"
        ]

        exclusion = [
            "Non-peer-reviewed sources (blogs, preprints excluded unless seminal)",
            "Papers without clear methodology"
        ]

        if contribution_type == ContributionType.SURVEY:
            inclusion.append("Systematic reviews and meta-analyses")
            exclusion.append("Papers outside the defined subtopics")

        # Add topic-specific criteria
        if decomposition.get("common_methods"):
            inclusion.append(f"Studies using: {', '.join(decomposition['common_methods'][:3])}")

        return {
            "inclusion_criteria": inclusion,
            "exclusion_criteria": exclusion,
            "languages": ["English"],
            "venue_types": ["Journal", "Conference", "Workshop"]
        }

    def define_success_criteria(
        self,
        question: str,
        contribution_type: ContributionType,
        subtopics: list[str]
    ) -> dict:
        """
        Step 9: Define success criteria based on contribution type.
        """
        if contribution_type == ContributionType.EMPIRICAL:
            return {
                "hypothesis": f"Primary hypothesis derived from: {question}",
                "required_coverage": []
            }
        else:
            # Survey or other types
            return {
                "hypothesis": None,
                "required_coverage": [
                    f"Taxonomy of {subtopics[0] if subtopics else 'approaches'}",
                    "Comparison of methodologies across papers",
                    "Identification of research gaps",
                    "Future research directions"
                ]
            }

    def validate_scope(
        self,
        decomposition: dict,
        papers: list[dict]
    ) -> tuple[bool, str]:
        """
        Check for common failure modes.
        """
        # Too broad: more than 6 subtopics or thousands of papers
        if len(decomposition["subtopics"]) > 6:
            return False, "Too broad: More than 6 subtopics identified. Narrow the focus."

        # Too narrow: fewer than 15 relevant papers
        if len(papers) < 15:
            return False, f"Too narrow: Only {len(papers)} relevant papers found. Expand the scope."

        return True, "Scope validated successfully"

    def run(
        self,
        topic: str,
        paper_type: Optional[str] = None,
        venue: Optional[str] = None,
        page_limit: Optional[int] = None,
        deadline: Optional[str] = None,
        year_range: Optional[tuple[int, int]] = None
    ) -> ScopeDocument:
        """
        Execute the full scoping pipeline.

        Steps 1-9 automated, Step 10 requires human approval.
        """
        # Step 1: Parse constraints
        constraints = self.parse_constraints(
            topic, paper_type, venue, page_limit, deadline, year_range
        )

        # Step 2-3: Landscape probe and decomposition
        papers, decomposition = self.run_landscape_probe(
            constraints["original_topic"],
            constraints["year_range"]
        )

        # Validate scope
        is_valid, validation_msg = self.validate_scope(decomposition, papers)
        if not is_valid:
            print(f"[Stage 1] Validation failed: {validation_msg}")
            print("[Stage 1] Attempting to adjust scope...")
            # Could loop back to step 4 here with adjusted parameters

        # Steps 4-5: Generate and score candidates
        candidates = self.generate_candidates(constraints, decomposition)

        # Step 6: Novelty check
        candidates = self.check_candidate_novelty(
            candidates,
            constraints["year_range"]
        )

        # Step 7: Select question and contribution type
        selected, contribution_type = self.select_question(candidates)

        # Step 8: Define boundaries
        boundaries = self.define_boundaries(constraints, decomposition, contribution_type)

        # Step 9: Define success criteria
        success = self.define_success_criteria(
            selected.question,
            contribution_type,
            decomposition["subtopics"]
        )

        # Create scope document
        scope_doc = ScopeDocument(
            original_topic=constraints["original_topic"],
            paper_type=constraints["paper_type"],
            target_venue=constraints["target_venue"],
            page_limit=constraints["page_limit"],
            deadline=constraints["deadline"],
            year_range=constraints["year_range"],
            subtopics=decomposition["subtopics"],
            keywords=decomposition["keywords"],
            common_methods=decomposition["common_methods"],
            common_datasets=decomposition["common_datasets"],
            research_question=selected.question,
            contribution_type=contribution_type,
            question_rationale=selected.rationale,
            inclusion_criteria=boundaries["inclusion_criteria"],
            exclusion_criteria=boundaries["exclusion_criteria"],
            languages=boundaries["languages"],
            venue_types=boundaries["venue_types"],
            hypothesis=success["hypothesis"],
            required_coverage=success["required_coverage"],
            approved=self.auto_approve
        )

        return scope_doc

    def request_approval(self, scope_doc: ScopeDocument) -> ScopeDocument:
        """
        Step 10: Human checkpoint for approval.

        Presents the scope document and collects user approval.
        """
        print("\n" + "="*70)
        print("SCOPE DOCUMENT - AWAITING APPROVAL")
        print("="*70)
        print(f"\nTopic: {scope_doc.original_topic}")
        print(f"\nResearch Question: {scope_doc.research_question}")
        print(f"Contribution Type: {scope_doc.contribution_type.value}")
        print(f"\nRationale: {scope_doc.question_rationale}")

        print(f"\nSubtopics ({len(scope_doc.subtopics)}):")
        for i, subtopic in enumerate(scope_doc.subtopics, 1):
            print(f"  {i}. {subtopic}")

        print(f"\nInclusion Criteria:")
        for criterion in scope_doc.inclusion_criteria:
            print(f"  ✓ {criterion}")

        print(f"\nExclusion Criteria:")
        for criterion in scope_doc.exclusion_criteria:
            print(f"  ✗ {criterion}")

        if scope_doc.required_coverage:
            print(f"\nRequired Coverage:")
            for item in scope_doc.required_coverage:
                print(f"  • {item}")

        print("\n" + "-"*70)

        while True:
            response = input("\nApprove this scope? (yes/edit/reject): ").strip().lower()

            if response in ["yes", "y"]:
                scope_doc.approved = True
                print("\n✓ Scope approved. Proceeding to Stage 2...")
                break
            elif response == "edit":
                print("\nEditing options:")
                print("  1. Change research question")
                print("  2. Modify subtopics")
                print("  3. Adjust boundaries")
                print("  4. Add notes")

                edit_choice = input("Select option (1-4): ").strip()

                if edit_choice == "1":
                    new_question = input("Enter new research question: ").strip()
                    scope_doc.research_question = new_question
                elif edit_choice == "2":
                    print("Current subtopics:")
                    for i, st in enumerate(scope_doc.subtopics, 1):
                        print(f"  {i}. {st}")
                    changes = input("Enter changes (e.g., 'remove 2, add Topic X'): ")
                    # Parse and apply changes (simplified)
                elif edit_choice == "3":
                    criterion = input("Add inclusion criterion: ").strip()
                    if criterion:
                        scope_doc.inclusion_criteria.append(criterion)
                elif edit_choice == "4":
                    notes = input("Enter notes: ").strip()
                    scope_doc.approval_notes = notes
            elif response in ["reject", "r"]:
                scope_doc.approved = False
                scope_doc.approval_notes = input("Reason for rejection: ").strip()
                print("\n✗ Scope rejected. Returning to candidate selection...")
                break

        return scope_doc


def demo_scope():
    """Demo function showing the scoping pipeline."""
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    agent = ScopingAgent(llm=llm)

    scope_doc = agent.run(
        topic="LLMs for education",
        paper_type="survey"
    )

    scope_doc = agent.request_approval(scope_doc)

    # Save to file
    with open("scope_document.json", "w") as f:
        f.write(scope_doc.model_dump_json(indent=2))

    print(f"\nScope document saved to scope_document.json")
    return scope_doc


if __name__ == "__main__":
    demo_scope()
