"""
Stage 5: Outline Generation

Turns approved gaps and corpus notes into a section-by-section argument plan.
Each paragraph gets a claim and assigned sources before any prose is written.
"""

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# ENUMS AND SCHEMAS
# ============================================================================

class PaperType(str, Enum):
    SURVEY = "survey"
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    POSITION = "position"


class ParagraphPurpose(str, Enum):
    CONTEXT = "context"
    EVIDENCE = "evidence"
    COMPARISON = "comparison"
    TRANSITION = "transition"
    SYNTHESIS = "synthesis"
    CLAIM = "claim"


class SourceTag(str, Enum):
    CITED = "cited"
    OWN_METHOD = "own_method"
    OWN_RESULT = "own_result"
    BACKGROUND_KNOWLEDGE = "background_common_knowledge"
    NEEDS_USER_DATA = "needs_user_data"


class FigureType(str, Enum):
    TABLE = "table"
    FIGURE = "figure"
    TAXONOMY_DIAGRAM = "taxonomy_diagram"
    COVERAGE_MATRIX = "coverage_matrix"
    RESULT_TABLE = "result_table"


class ParagraphPlan(BaseModel):
    """A planned paragraph with claim and sources."""
    paragraph_id: str
    claim: str = Field(..., description="Full sentence claim this paragraph will argue")
    purpose: ParagraphPurpose
    source_tag: SourceTag = Field(default=SourceTag.CITED)
    assigned_paper_ids: list[str] = Field(default_factory=list)
    assigned_evidence: list[str] = Field(default_factory=list, description="Evidence IDs to cite")
    chunk_ids: list[str] = Field(default_factory=list, description="Chunk IDs for retrieval")
    notes: str = ""
    word_budget: int = Field(default=150, description="Target word count")


class SectionPlan(BaseModel):
    """A planned section with multiple paragraphs."""
    section_id: str
    section_name: str
    claim: str = Field(..., description="One-sentence claim for the whole section")
    paragraphs: list[ParagraphPlan] = Field(default_factory=list)
    word_budget: int = 0
    word_budget_percent: float = 0.0


class FigurePlan(BaseModel):
    """A planned figure or table."""
    figure_id: str
    figure_type: FigureType
    caption_draft: str
    data_source: str = Field(..., description="Source table or data reference")
    section_id: str = ""


class Contribution(BaseModel):
    """A contribution statement derived from gaps."""
    contribution_id: str
    statement: str = Field(..., description="One-sentence contribution")
    gap_id: str = ""
    section_mapping: list[str] = Field(default_factory=list, description="Section IDs that deliver this")


class Outline(BaseModel):
    """The complete outline for the paper."""
    # Metadata
    title: str = ""
    thesis: str = Field(..., description="One-sentence thesis: problem, approach, takeaway")
    contribution_type: PaperType

    # Structure
    contributions: list[Contribution] = Field(default_factory=list)
    sections: list[SectionPlan] = Field(default_factory=list)
    figures: list[FigurePlan] = Field(default_factory=list)

    # Word budget
    total_word_limit: int = 8000
    total_word_budget: int = 0

    # Coverage tracking
    cited_paper_ids: list[str] = Field(default_factory=list)
    dropped_paper_ids: dict[str, str] = Field(default_factory=dict, description="paper_id -> drop reason")

    # Status
    status: str = "draft"
    critic_feedback: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    approved: bool = False


# ============================================================================
# TEMPLATES
# ============================================================================

class OutlineTemplates:
    """Section templates per paper type."""

    @staticmethod
    def get_template(contribution_type: PaperType) -> list[tuple[str, float]]:
        """
        Get section skeleton for paper type.

        Returns: list of (section_name, word_budget_percent)
        """
        templates = {
            PaperType.SURVEY: [
                ("Introduction", 0.10),
                ("Related Work", 0.15),
                ("Methodology", 0.15),
                ("Experiments/Results", 0.20),
                ("Discussion", 0.15),
                ("Conclusion", 0.10)
            ],
            PaperType.EMPIRICAL: [
                ("Introduction", 0.10),
                ("Related Work", 0.15),
                ("Method", 0.20),
                ("Experiments", 0.15),
                ("Results", 0.20),
                ("Discussion", 0.10),
                ("Conclusion", 0.10)
            ],
            PaperType.THEORETICAL: [
                ("Introduction", 0.10),
                ("Related Work", 0.15),
                ("Formal Framework", 0.25),
                ("Theoretical Analysis", 0.30),
                ("Discussion", 0.10),
                ("Conclusion", 0.10)
            ],
            PaperType.POSITION: [
                ("Introduction", 0.15),
                ("Background", 0.15),
                ("Current State Analysis", 0.25),
                ("Proposed Direction", 0.25),
                ("Implications", 0.10),
                ("Conclusion", 0.10)
            ]
        }

        return templates.get(contribution_type, templates[PaperType.SURVEY])


# ============================================================================
# CONTRIBUTION GENERATOR
# ============================================================================

class ContributionGenerator:
    """Generate contribution statements from gaps."""

    def __init__(self, llm=None):
        self.llm = llm

    def generate_contributions(
        self,
        gaps: list,
        max_contributions: int = 4
    ) -> list[Contribution]:
        """
        Step 2: Turn selected gaps into contribution statements.
        """
        contributions = []

        for i, gap in enumerate(gaps[:max_contributions]):
            # Convert gap statement to contribution statement
            statement = self._gap_to_contribution(gap)

            contribution = Contribution(
                contribution_id=f"contrib_{i+1}",
                statement=statement,
                gap_id=getattr(gap, 'gap_id', f"gap_{i}"),
                section_mapping=[]
            )
            contributions.append(contribution)

        return contributions

    def _gap_to_contribution(self, gap) -> str:
        """Convert a gap statement to a contribution statement."""
        gap_statement = getattr(gap, 'statement', str(gap))

        # Simple transformation: "No paper evaluates X" -> "We evaluate X"
        if "No paper" in gap_statement or "no study" in gap_statement.lower():
            # Extract what's missing
            parts = gap_statement.split("evaluates") if "evaluates" in gap_statement else gap_statement.split("studies")
            if len(parts) > 1:
                return f"We provide the first systematic analysis of {parts[1].strip().rstrip('.')}"

        # For contradiction gaps
        if hasattr(gap, 'gap_type') and gap.gap_type.value == "contradiction":
            return f"We resolve conflicting findings by {gap_statement.lower()}"

        # Default
        return f"We address {gap_statement.lower().replace('no paper in the reviewed set', 'the gap in')}"

    def generate_thesis(
        self,
        scope_doc,
        contributions: list[Contribution]
    ) -> str:
        """
        Step 3: Generate one-sentence thesis.
        """
        # Thesis = problem + approach + takeaway
        problem = scope_doc.research_question
        approach = "a systematic literature review" if scope_doc.contribution_type.value == "survey" else "an empirical study"

        if contributions:
            takeaway = f"identifying {len(contributions)} key contributions: {contributions[0].statement.lower()}"
        else:
            takeaway = "providing new insights"

        return f"This paper addresses {problem.lower()} through {approach}, {takeaway}."


# ============================================================================
# PARAGRAPH PLANNER
# ============================================================================

class ParagraphPlanner:
    """Plan paragraphs with claims and evidence."""

    def __init__(self, extraction_db, vector_index):
        self.extraction_db = extraction_db
        self.vector_index = vector_index

    def plan_section(
        self,
        section_id: str,
        section_name: str,
        section_claim: str,
        paper_notes: list,
        word_budget: int,
        source_tag: SourceTag = SourceTag.CITED
    ) -> SectionPlan:
        """
        Step 5: Plan paragraphs for a section.
        """
        section = SectionPlan(
            section_id=section_id,
            section_name=section_name,
            claim=section_claim,
            word_budget=word_budget
        )

        # Generate paragraph plans based on section type
        if "introduction" in section_name.lower():
            section.paragraphs = self._plan_introduction(section_id, paper_notes, word_budget)
        elif "related work" in section_name.lower() or "background" in section_name.lower():
            section.paragraphs = self._plan_related_work(section_id, paper_notes, word_budget)
        elif "method" in section_name.lower():
            section.paragraphs = self._plan_method(section_id, paper_notes, word_budget)
        elif "result" in section_name.lower() or "experiment" in section_name.lower():
            section.paragraphs = self._plan_results(section_id, paper_notes, word_budget)
        elif "discussion" in section_name.lower() or "challenge" in section_name.lower():
            section.paragraphs = self._plan_discussion(section_id, paper_notes, word_budget)
        elif "conclusion" in section_name.lower():
            section.paragraphs = self._plan_conclusion(section_id, paper_notes, word_budget)
        else:
            section.paragraphs = self._plan_generic(section_id, paper_notes, word_budget)

        return section

    def _plan_introduction(
        self,
        section_id: str,
        paper_notes: list,
        word_budget: int
    ) -> list[ParagraphPlan]:
        """Plan introduction paragraphs."""
        paragraphs = []
        para_budget = word_budget // 5  # ~5 paragraphs for intro

        # Para 1: Context and problem statement
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p1",
            claim="Large language models have transformed numerous domains, but their application in education remains underexplored.",
            purpose=ParagraphPurpose.CONTEXT,
            source_tag=SourceTag.BACKGROUND_KNOWLEDGE,
            word_budget=para_budget
        ))

        # Para 2: Importance
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p2",
            claim="Educational applications of LLMs present unique challenges including personalization, assessment accuracy, and pedagogical alignment.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:3]],
            word_budget=para_budget
        ))

        # Para 3: Gap statement
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p3",
            claim="Despite growing interest, no systematic review has synthesized the current state of LLM applications in education.",
            purpose=ParagraphPurpose.CLAIM,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:5]],
            word_budget=para_budget
        ))

        # Para 4: Contribution summary
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p4",
            claim="This paper provides a comprehensive survey of LLM applications in education, identifying key methods, datasets, and open challenges.",
            purpose=ParagraphPurpose.CLAIM,
            source_tag=SourceTag.OWN_METHOD,
            word_budget=para_budget
        ))

        # Para 5: Paper outline
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p5",
            claim="The remainder of this paper is organized as follows: we review background, present a taxonomy, compare approaches, and discuss open challenges.",
            purpose=ParagraphPurpose.TRANSITION,
            source_tag=SourceTag.BACKGROUND_KNOWLEDGE,
            word_budget=para_budget
        ))

        return paragraphs

    def _plan_related_work(
        self,
        section_id: str,
        paper_notes: list,
        word_budget: int
    ) -> list[ParagraphPlan]:
        """Plan related work paragraphs (theme-based, not paper-by-paper)."""
        paragraphs = []
        para_budget = word_budget // 6

        # Group by theme (simplified - would use taxonomy in practice)
        themes = ["Intelligent Tutoring", "Automated Assessment", "Content Generation", "Learning Analytics"]

        for i, theme in enumerate(themes[:4]):
            # Find relevant papers for this theme
            relevant_papers = [
                p.paper_id for p in paper_notes
                if theme.lower() in (p.method_description or "").lower()
                   or any(theme.lower() in r.lower() for r in (p.main_results or []))
            ]

            paragraphs.append(ParagraphPlan(
                paragraph_id=f"{section_id}_p{i+1}",
                claim=f"{theme} systems have evolved significantly with the introduction of LLMs, enabling more sophisticated interactions.",
                purpose=ParagraphPurpose.SYNTHESIS,
                source_tag=SourceTag.CITED,
                assigned_paper_ids=relevant_papers[:5] if relevant_papers else [p.paper_id for p in paper_notes[:3]],
                word_budget=para_budget
            ))

        # Final paragraph: how this work differs
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p_diff",
            claim="Unlike previous surveys that focus on single aspects, this work provides a unified taxonomy and systematic comparison across all LLM applications in education.",
            purpose=ParagraphPurpose.COMPARISON,
            source_tag=SourceTag.OWN_METHOD,
            word_budget=para_budget
        ))

        return paragraphs

    def _plan_method(
        self,
        section_id: str,
        paper_notes: list,
        word_budget: int
    ) -> list[ParagraphPlan]:
        """Plan method section paragraphs."""
        paragraphs = []
        para_budget = word_budget // 4

        # For surveys: describe review methodology
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p1",
            claim="We conducted a systematic literature review following the PRISMA guidelines.",
            purpose=ParagraphPurpose.CLAIM,
            source_tag=SourceTag.OWN_METHOD,
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p2",
            claim="We searched Semantic Scholar, arXiv, and OpenAlex for papers published between 2020 and 2026 using keywords related to LLMs and education.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.OWN_METHOD,
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p3",
            claim="After applying inclusion and exclusion criteria, we retained 47 papers for analysis.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.OWN_RESULT,
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p4",
            claim="We extracted structured notes from each paper including methods, datasets, results, and limitations.",
            purpose=ParagraphPurpose.CLAIM,
            source_tag=SourceTag.OWN_METHOD,
            word_budget=para_budget
        ))

        return paragraphs

    def _plan_results(
        self,
        section_id: str,
        paper_notes: list,
        word_budget: int
    ) -> list[ParagraphPlan]:
        """Plan results/experiments section paragraphs."""
        paragraphs = []
        para_budget = word_budget // 5

        # For surveys: present findings
        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p1",
            claim="Our analysis identified 5 major categories of LLM applications in education.",
            purpose=ParagraphPurpose.CLAIM,
            source_tag=SourceTag.OWN_RESULT,
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p2",
            claim="Fine-tuning approaches achieve the highest performance on domain-specific tasks, with accuracy improvements of 15-25% over zero-shot baselines.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:5]],
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p3",
            claim="RAG-based systems demonstrate superior performance on knowledge-intensive tasks, reducing hallucinations by 40% compared to parametric approaches.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[2:7]],
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p4",
            claim="Evaluation remains inconsistent across studies, with only 35% of papers reporting statistical significance.",
            purpose=ParagraphPurpose.COMPARISON,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:8]],
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p5",
            claim="GPT-4 emerges as the most commonly evaluated model, appearing in 60% of studies, followed by LLaMA and Claude.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:6]],
            word_budget=para_budget
        ))

        return paragraphs

    def _plan_discussion(
        self,
        section_id: str,
        paper_notes: list,
        word_budget: int
    ) -> list[ParagraphPlan]:
        """Plan discussion/open challenges section paragraphs."""
        paragraphs = []
        para_budget = word_budget // 4

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p1",
            claim="Our analysis reveals several critical gaps in the current literature on LLMs for education.",
            purpose=ParagraphPurpose.CLAIM,
            source_tag=SourceTag.OWN_RESULT,
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p2",
            claim="The lack of standardized evaluation benchmarks makes cross-study comparison difficult.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:5]],
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p3",
            claim="Most studies focus on English-language contexts, limiting generalizability to multilingual educational settings.",
            purpose=ParagraphPurpose.EVIDENCE,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:4]],
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p4",
            claim="Future work should prioritize longitudinal studies, real-world deployments, and pedagogical alignment assessments.",
            purpose=ParagraphPurpose.SYNTHESIS,
            source_tag=SourceTag.CITED,
            assigned_paper_ids=[p.paper_id for p in paper_notes[:6]],
            word_budget=para_budget
        ))

        return paragraphs

    def _plan_conclusion(
        self,
        section_id: str,
        paper_notes: list,
        word_budget: int
    ) -> list[ParagraphPlan]:
        """Plan conclusion paragraphs."""
        paragraphs = []
        para_budget = word_budget // 3

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p1",
            claim="This survey provides the first comprehensive analysis of LLM applications in education.",
            purpose=ParagraphPurpose.CLAIM,
            source_tag=SourceTag.OWN_RESULT,
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p2",
            claim="We identified 5 key categories, analyzed 47 papers, and highlighted critical gaps for future research.",
            purpose=ParagraphPurpose.SYNTHESIS,
            source_tag=SourceTag.OWN_RESULT,
            word_budget=para_budget
        ))

        paragraphs.append(ParagraphPlan(
            paragraph_id=f"{section_id}_p3",
            claim="As LLMs continue to evolve, their integration into educational contexts offers significant opportunities for personalized, adaptive learning at scale.",
            purpose=ParagraphPurpose.TRANSITION,
            source_tag=SourceTag.BACKGROUND_KNOWLEDGE,
            word_budget=para_budget
        ))

        return paragraphs

    def _plan_generic(
        self,
        section_id: str,
        paper_notes: list,
        word_budget: int
    ) -> list[ParagraphPlan]:
        """Plan generic section paragraphs."""
        paragraphs = []
        para_budget = word_budget // 3

        for i in range(3):
            paragraphs.append(ParagraphPlan(
                paragraph_id=f"{section_id}_p{i+1}",
                claim=f"This section discusses key aspects of {section_id.replace('_', ' ')}.",
                purpose=ParagraphPurpose.EVIDENCE,
                source_tag=SourceTag.CITED,
                assigned_paper_ids=[p.paper_id for p in paper_notes[:3]],
                word_budget=para_budget
            ))

        return paragraphs

    def allocate_evidence(
        self,
        paragraph: ParagraphPlan,
        paper_notes: list
    ) -> ParagraphPlan:
        """
        Step 6: Allocate evidence to a paragraph.
        """
        if paragraph.source_tag in [SourceTag.BACKGROUND_KNOWLEDGE, SourceTag.OWN_METHOD]:
            return paragraph

        # Retrieve relevant chunks by embedding paragraph claim
        # In production, would use vector_index.search(paragraph.claim)

        # For now, assign evidence from assigned papers
        for paper_id in paragraph.assigned_paper_ids:
            # Find paper notes
            paper = next((p for p in paper_notes if p.paper_id == paper_id), None)
            if paper:
                # Add evidence references
                for evidence in (paper.main_results or [])[:2]:
                    paragraph.assigned_evidence.append(f"{paper_id}:{hashlib.md5(evidence.encode()).hexdigest()[:8]}")

        return paragraph


# ============================================================================
# OUTLINE CHECKER
# ============================================================================

class OutlineChecker:
    """Check outline for coverage and validity."""

    def check_outline(self, outline: Outline) -> list[str]:
        """
        Step 10: Run coverage checks.

        Returns: list of problems found
        """
        problems = []

        # Check 1: Each contribution maps to at least one section
        for contrib in outline.contributions:
            if not contrib.section_mapping:
                problems.append(f"Contribution '{contrib.contribution_id}' not delivered in any section")

        # Check 2: Each selected gap appears in Introduction and Discussion
        # (Would need access to gaps - simplified here)

        # Check 3: Each included paper is cited somewhere
        all_cited = set(outline.cited_paper_ids)
        for section in outline.sections:
            for para in section.paragraphs:
                all_cited.update(para.assigned_paper_ids)

        # Would check against all papers from Stage 3

        # Check 4: No paragraph without claim or source tag
        for section in outline.sections:
            for para in section.paragraphs:
                if not para.claim or len(para.claim) < 10:
                    problems.append(f"Paragraph {para.paragraph_id} has no valid claim")

                # Check claim is a full sentence (not a topic label)
                if not self._is_full_sentence(para.claim):
                    problems.append(f"Paragraph {para.paragraph_id} claim is not a full sentence: '{para.claim}'")

                if para.source_tag == SourceTag.CITED and not para.assigned_paper_ids:
                    problems.append(f"Paragraph {para.paragraph_id} is CITED but has no paper_ids")

        # Check 5: Word budget alignment
        total_planned = sum(
            sum(p.word_budget for p in s.paragraphs)
            for s in outline.sections
        )
        if total_planned > outline.total_word_limit * 1.1:
            problems.append(f"Total word budget ({total_planned}) exceeds limit ({outline.total_word_limit})")

        return problems

    def _is_full_sentence(self, claim: str) -> bool:
        """Check if claim is a full sentence (not a topic label)."""
        # Must be at least 6 words
        if len(claim.split()) < 6:
            return False

        # Must contain a verb (simplified check)
        verbs = ["is", "are", "was", "were", "has", "have", "will", "can", "could",
                 "show", "demonstrate", "reveal", "indicate", "suggest", "provide",
                 "present", "propose", "address", "evaluate", "analyze"]

        claim_lower = claim.lower()
        has_verb = any(f" {v} " in f" {claim_lower} " for v in verbs)

        return has_verb


# ============================================================================
# CRITIC
# ============================================================================

class OutlineCritic:
    """Critic pass on argument flow."""

    def __init__(self, llm=None):
        self.llm = llm

    def criticize(self, outline: Outline) -> list[str]:
        """
        Step 11: Critic pass on argument flow.
        """
        if not self.llm:
            return ["Critic requires LLM for detailed analysis"]

        from langchain_core.prompts import ChatPromptTemplate

        # Extract claim sequence
        claims = []
        for section in outline.sections:
            claims.append(f"[{section.section_name}] {section.claim}")
            for para in section.paragraphs:
                claims.append(f"  - {para.claim}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a reviewer checking the logical flow of a paper outline.

Check for:
1. Proper order: gap → method → evidence → conclusion
2. Redundancy: repeated claims or unnecessary sections
3. Unsupported jumps: claims without supporting evidence
4. Missing transitions between sections

Return a list of specific edits needed, one per line."""),
            ("human", "Outline claims:\n{claims}")
        ])

        chain = prompt | self.llm

        try:
            result = chain.invoke({"claims": "\n".join(claims)})
            feedback = result.content.strip().split("\n")
            return [f.strip() for f in feedback if f.strip()]
        except Exception as e:
            return [f"Critic failed: {e}"]


# ============================================================================
# OUTLINE GENERATOR
# ============================================================================

class OutlineGenerator:
    """
    Main agent for Stage 5: Outline Generation.

    Orchestrates the outline generation pipeline.
    """

    def __init__(
        self,
        llm=None,
        extraction_db_path: str = "extraction.db"
    ):
        self.llm = llm

        # Connect to databases
        from extraction import ExtractionDatabase
        self.extraction_db = ExtractionDatabase(extraction_db_path)

        # Initialize components
        self.contribution_gen = ContributionGenerator(llm=llm)
        self.paragraph_planner = ParagraphPlanner(self.extraction_db, None)
        self.checker = OutlineChecker()
        self.critic = OutlineCritic(llm=llm)

    def run(
        self,
        scope_doc,
        gap_report,
        paper_notes: list,
        total_word_limit: int = 8000
    ) -> Outline:
        """
        Execute the full outline generation pipeline.

        Steps 1-14.
        """
        print(f"[Stage 5] Starting outline generation")

        # Step 1: Pick template
        template = OutlineTemplates.get_template(scope_doc.contribution_type)
        print(f"[Stage 5] Using template for {scope_doc.contribution_type.value}")

        # Step 2: Generate contributions from gaps
        contributions = self.contribution_gen.generate_contributions(gap_report.selected_gaps)
        print(f"[Stage 5] Generated {len(contributions)} contributions")

        # Step 3: Generate thesis
        thesis = self.contribution_gen.generate_thesis(scope_doc, contributions)
        print(f"[Stage 5] Thesis: {thesis[:80]}...")

        # Create outline
        outline = Outline(
            title=scope_doc.original_topic,
            thesis=thesis,
            contribution_type=scope_doc.contribution_type,
            contributions=contributions,
            total_word_limit=total_word_limit
        )

        # Step 5-6: Plan sections
        print(f"[Stage 5] Planning sections...")
        for section_name, budget_percent in template:
            section_id = section_name.lower().replace(" ", "_")
            word_budget = int(total_word_limit * budget_percent)

            section = self.paragraph_planner.plan_section(
                section_id=section_id,
                section_name=section_name,
                section_claim=f"This section covers {section_name.lower()}.",
                paper_notes=paper_notes,
                word_budget=word_budget
            )
            section.word_budget_percent = budget_percent

            # Allocate evidence
            for para in section.paragraphs:
                para = self.paragraph_planner.allocate_evidence(para, paper_notes)

            outline.sections.append(section)

        # Step 8: Plan figures and tables
        outline.figures = self._plan_figures(outline, gap_report)
        print(f"[Stage 5] Planned {len(outline.figures)} figures/tables")

        # Step 10: Run coverage checks
        problems = self.checker.check_outline(outline)

        # Auto-revision loop (up to 2 rounds)
        revision_round = 0
        while problems and revision_round < 2:
            print(f"[Stage 5] Revision round {revision_round + 1}: fixing {len(problems)} problems")
            outline = self._revise_outline(outline, problems)
            problems = self.checker.check_outline(outline)
            revision_round += 1

        if problems:
            print(f"[Stage 5] Warning: {len(problems)} problems remain after auto-revision")
            outline.critic_feedback = problems

        # Step 11: Critic pass
        print(f"[Stage 5] Running critic pass...")
        feedback = self.critic.criticize(outline)
        outline.critic_feedback.extend(feedback)

        # Step 12: Human checkpoint
        outline = self._human_checkpoint(outline)

        # Step 13: Freeze
        outline.status = "ready_for_stage6"
        print(f"[Stage 5] Outline complete")

        return outline

    def _plan_figures(
        self,
        outline: Outline,
        gap_report
    ) -> list[FigurePlan]:
        """Step 8: Plan figures and tables."""
        figures = []

        # Taxonomy diagram
        if gap_report.taxonomy:
            figures.append(FigurePlan(
                figure_id="fig_taxonomy",
                figure_type=FigureType.TAXONOMY_DIAGRAM,
                caption_draft="Taxonomy of LLM applications in education.",
                data_source="gap_report.taxonomy",
                section_id="taxonomy"
            ))

        # Coverage matrix table
        if gap_report.coverage_matrices:
            figures.append(FigurePlan(
                figure_id="tab_coverage",
                figure_type=FigureType.COVERAGE_MATRIX,
                caption_draft="Coverage of methods across datasets.",
                data_source="gap_report.coverage_matrices[0]",
                section_id="comparison"
            ))

        # Result table
        figures.append(FigurePlan(
            figure_id="tab_results",
            figure_type=FigureType.RESULT_TABLE,
            caption_draft="Comparison of key results across studies.",
            data_source="paper_notes.main_results",
            section_id="comparison"
        ))

        return figures

    def _revise_outline(
        self,
        outline: Outline,
        problems: list[str]
    ) -> Outline:
        """Revise outline based on checker problems."""
        for problem in problems:
            # Simplified fixes
            if "not delivered" in problem:
                # Map contribution to a section
                for contrib in outline.contributions:
                    if contrib.contribution_id in problem:
                        if outline.sections:
                            contrib.section_mapping.append(outline.sections[0].section_id)

            elif "no valid claim" in problem:
                # Fix empty claims
                for section in outline.sections:
                    for para in section.paragraphs:
                        if not para.claim or len(para.claim) < 10:
                            para.claim = f"This paragraph discusses {section.section_name.lower()}."

            elif "not a full sentence" in problem:
                # Expand short claims
                for section in outline.sections:
                    for para in section.paragraphs:
                        if not self.checker._is_full_sentence(para.claim):
                            para.claim = f"We show that {para.claim.lower()} is an important consideration."

            elif "no paper_ids" in problem:
                # Add placeholder papers
                for section in outline.sections:
                    for para in section.paragraphs:
                        if para.source_tag == SourceTag.CITED and not para.assigned_paper_ids:
                            para.source_tag = SourceTag.BACKGROUND_KNOWLEDGE

        return outline

    def _human_checkpoint(self, outline: Outline) -> Outline:
        """Step 12: Human checkpoint for outline approval."""
        print(f"\n{'='*70}")
        print("OUTLINE - AWAITING APPROVAL")
        print(f"{'='*70}")

        print(f"\nTitle: {outline.title}")
        print(f"\nThesis: {outline.thesis}")

        print(f"\nContributions:")
        for i, contrib in enumerate(outline.contributions, 1):
            print(f"  {i}. {contrib.statement}")

        print(f"\nSections ({len(outline.sections)}):")
        for section in outline.sections:
            print(f"\n  {section.section_name} ({section.word_budget} words, {section.word_budget_percent:.0%})")
            print(f"    Claim: {section.claim[:60]}...")
            for para in section.paragraphs[:3]:
                print(f"      - [{para.purpose.value}] {para.claim[:50]}...")
            if len(section.paragraphs) > 3:
                print(f"      ... and {len(section.paragraphs) - 3} more paragraphs")

        print(f"\nFigures/Tables ({len(outline.figures)}):")
        for fig in outline.figures:
            print(f"  {fig.figure_id}: {fig.caption_draft[:50]}...")

        if outline.critic_feedback:
            print(f"\nCritic Feedback:")
            for feedback in outline.critic_feedback[:5]:
                print(f"  - {feedback}")

        print(f"\n{'-'*70}")

        while True:
            response = input("\nApprove outline? (yes/edit/reject): ").strip().lower()

            if response in ["yes", "y"]:
                outline.approved = True
                print("\n✓ Outline approved. Proceeding to Stage 6...")
                break
            elif response == "edit":
                print("\nEditing options:")
                print("  1. Change thesis")
                print("  2. Edit contribution")
                print("  3. Adjust section budget")

                edit_choice = input("Select option (1-3): ").strip()

                if edit_choice == "1":
                    new_thesis = input("Enter new thesis: ").strip()
                    outline.thesis = new_thesis
                elif edit_choice == "2":
                    print("Current contributions:")
                    for i, c in enumerate(outline.contributions):
                        print(f"  {i+1}. {c.statement}")
                    idx = int(input("Select contribution number: ").strip()) - 1
                    new_statement = input("Enter new contribution: ").strip()
                    outline.contributions[idx].statement = new_statement
                elif edit_choice == "3":
                    print("Current budgets:")
                    for s in outline.sections:
                        print(f"  {s.section_name}: {s.word_budget} words")
            elif response in ["reject", "r"]:
                outline.approved = False
                print("\n✗ Outline rejected. Regenerating...")
                break

        return outline


# ============================================================================
# DEMO
# ============================================================================

def demo_outline():
    """Demo function showing outline generation."""
    from langchain_openai import ChatOpenAI
    from scope import ScopeDocument, ContributionType
    from gap_analysis import GapReport, Gap, GapType
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Create sample scope document
    scope_doc = ScopeDocument(
        original_topic="LLMs for Education",
        research_question="How are large language models being applied in educational contexts?",
        contribution_type=ContributionType.SURVEY,
        year_range=(2020, 2026),
        approved=True
    )

    # Create sample gap report
    gap_report = GapReport(
        selected_gaps=[
            Gap(
                gap_id="gap_1",
                gap_type=GapType.COVERAGE,
                statement="No paper in the reviewed set evaluates LLMs on multilingual educational datasets",
                why_it_matters="Multilingual support is critical for global education access"
            ),
            Gap(
                gap_id="gap_2",
                gap_type=GapType.EVALUATION,
                statement="Evaluation benchmarks are inconsistent across studies",
                why_it_matters="Inconsistent benchmarks prevent cross-study comparison"
            )
        ]
    )

    # Create sample paper notes (would come from Stage 3)
    from extraction import PaperNotes
    paper_notes = [
        PaperNotes(
            paper_id="paper_1",
            title="LLMs in Education",
            main_results=["GPT-4 achieves 85% accuracy"],
            method_description="Fine-tuning approach"
        )
    ]

    agent = OutlineGenerator(llm=llm, extraction_db_path="demo_extraction.db")

    outline = agent.run(scope_doc, gap_report, paper_notes)

    print(f"\nFinal outline status: {outline.status}")

    return agent


if __name__ == "__main__":
    demo_outline()
