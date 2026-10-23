"""
Stage 6: Drafting

Writes prose paragraph by paragraph from the Outline.
Each paragraph gets claim, sources, and word budget from the plan.
"""

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# SCHEMAS
# ============================================================================

class ParagraphDraft(BaseModel):
    """A drafted paragraph with citations."""
    paragraph_id: str
    claim: str
    text: str = Field(..., description="The drafted paragraph text")
    cite_keys: list[str] = Field(default_factory=list, description="Citation keys used")
    quote_verbatim: list[str] = Field(default_factory=list, description="Verbatim quotes used")
    numbers_reported: list[str] = Field(default_factory=list, description="Numbers mentioned")
    flags: list[str] = Field(default_factory=list, description="Issues or warnings")
    word_count: int = 0
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SectionDraft(BaseModel):
    """A drafted section with multiple paragraphs."""
    section_id: str
    section_name: str
    paragraphs: list[ParagraphDraft] = Field(default_factory=list)
    word_count: int = 0


class Draft(BaseModel):
    """The complete draft of the paper."""
    # Metadata
    title: str = ""
    abstract: str = ""

    # Content
    sections: list[SectionDraft] = Field(default_factory=list)

    # Tracking
    total_word_count: int = 0
    total_cite_keys: list[str] = Field(default_factory=list)
    ungrounded_claims: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list, description="Paragraph IDs needing user data")

    # Status
    status: str = "draft"
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# SINGLE PARAGRAPH GENERATOR
# ============================================================================

class ParagraphGenerator:
    """Generate a single paragraph from a plan."""

    def __init__(self, llm=None):
        self.llm = llm

    def generate(
        self,
        paragraph_plan,
        paper_notes: list,
        chunks: list = None
    ) -> ParagraphDraft:
        """
        Generate a paragraph from a plan with structured output.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        # Build evidence context
        evidence_context = self._build_evidence_context(
            paragraph_plan.assigned_paper_ids,
            paper_notes
        )

        # Build allowed cite keys
        allowed_cite_keys = paragraph_plan.assigned_paper_ids

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an academic writer drafting a paragraph from a plan.

PROMPT RULES:
1. Use ONLY the evidence provided. If it can't support the claim, add a flag instead of writing around the gap.
2. Every claim sentence must cite one or more of the allowed keys: {allowed_cite_keys}
3. Write in your own words. Quote only when exact wording matters, under 15 words.
4. Hedge to match the evidence: one study "suggests", several consistent studies "indicate".
5. Report numbers exactly as given, with dataset and metric.
6. Never write about your own experiments beyond what the outline marks as ready.
7. If this is a placeholder paragraph (needs_user_data), write "%% USER DATA REQUIRED %%" as the text.

CITATION FORMAT:
Use author-year format: (Smith et al., 2024) or Smith et al. (2024).

OUTPUT FORMAT:
{format_instructions}"""),
            ("human", """Paragraph Plan:
- Claim: {claim}
- Purpose: {purpose}
- Source Tag: {source_tag}
- Word Budget: {word_budget} words

Evidence:
{evidence_context}

Draft the paragraph.""")
        ]).partial(
            allowed_cite_keys=", ".join(allowed_cite_keys),
            format_instructions=PydanticOutputParser(pydantic_object=ParagraphDraft).get_format_instructions()
        )

        parser = PydanticOutputParser(pydantic_object=ParagraphDraft)

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "claim": paragraph_plan.claim,
                "purpose": paragraph_plan.purpose.value,
                "source_tag": paragraph_plan.source_tag.value,
                "word_budget": paragraph_plan.word_budget,
                "evidence_context": evidence_context
            })

            # Set metadata
            result.paragraph_id = paragraph_plan.paragraph_id
            result.claim = paragraph_plan.claim
            result.word_count = len(result.text.split())

            # Handle placeholders
            if paragraph_plan.source_tag.value == "needs_user_data":
                result.text = "%% USER DATA REQUIRED: {} %%".format(paragraph_plan.claim)
                result.flags.append("placeholder")

            return result

        except Exception as e:
            print(f"[Drafting] Paragraph generation failed: {e}")
            # Return empty draft with flag
            return ParagraphDraft(
                paragraph_id=paragraph_plan.paragraph_id,
                claim=paragraph_plan.claim,
                text="",
                flags=[f"Generation failed: {e}"]
            )

    def _build_evidence_context(
        self,
        paper_ids: list[str],
        paper_notes: list
    ) -> str:
        """Build evidence context string from paper notes."""
        if not paper_ids:
            return "No specific evidence assigned. Use background knowledge."

        context_parts = []

        for paper_id in paper_ids:
            # Find paper notes
            paper = next((p for p in paper_notes if p.paper_id == paper_id), None)
            if not paper:
                continue

            context_parts.append(f"\n[{paper_id}] {paper.title}")

            if paper.main_contributions:
                context_parts.append(f"Contributions: {', '.join(paper.main_contributions[:3])}")

            if paper.main_results:
                context_parts.append(f"Results: {', '.join(paper.main_results[:3])}")

            if paper.method_description:
                context_parts.append(f"Method: {paper.method_description[:200]}")

            # Add evidence quotes
            all_evidence = (
                paper.contributions_evidence +
                paper.results_evidence +
                paper.limitations_evidence
            )
            for evidence in all_evidence[:3]:
                context_parts.append(f"Quote: \"{evidence.quote}\"")

        return "\n".join(context_parts) if context_parts else "No specific evidence assigned."


# ============================================================================
# PARAGRAPH VALIDATOR
# ============================================================================

class ParagraphValidator:
    """Validate drafted paragraphs against evidence."""

    def __init__(self):
        self.max_ngram_overlap = 0.10  # 10% max overlap with source

    def validate(
        self,
        draft: ParagraphDraft,
        paragraph_plan,
        paper_notes: list,
        allowed_cite_keys: list[str]
    ) -> tuple[bool, list[str]]:
        """
        Validate a drafted paragraph.

        Returns: (is_valid, list of problems)
        """
        problems = []

        # Check 1: Placeholder handling
        if "%% USER DATA REQUIRED" in draft.text:
            if paragraph_plan.source_tag.value != "needs_user_data":
                problems.append("Placeholder text in non-placeholder paragraph")
            return True, problems  # Placeholders are valid as-is

        # Check 2: Uncited claims
        # Find sentences that look like claims but have no citation
        sentences = re.split(r'[.!?]+', draft.text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not any(cite in sentence for cite in draft.cite_keys):
                # Check if it's a claim sentence (has claim-like structure)
                claim_indicators = ["shows", "demonstrates", "indicates", "suggests",
                                    "reveals", "found", "reported", "achieved"]
                if any(indicator in sentence.lower() for indicator in claim_indicators):
                    problems.append(f"Uncited claim: '{sentence[:50]}...'")

        # Check 3: Wrong or invented citation keys
        for cite_key in draft.cite_keys:
            if cite_key not in allowed_cite_keys:
                problems.append(f"Invented citation key: {cite_key}")

        # Check 4: Numbers drift
        for number in draft.numbers_reported:
            # Check if number appears in evidence
            number_found = False
            for paper_id in paragraph_plan.assigned_paper_ids:
                paper = next((p for p in paper_notes if p.paper_id == paper_id), None)
                if paper:
                    all_text = " ".join(paper.main_results or [])
                    if number in all_text:
                        number_found = True
                        break

            if not number_found:
                problems.append(f"Number not found in evidence: {number}")

        # Check 5: Paraphrase too close to source
        for quote in draft.quote_verbatim:
            similarity = self._check_paraphrase_similarity(quote, draft.text)
            if similarity > self.max_ngram_overlap:
                problems.append(f"Paraphrase too close to source: '{quote[:30]}...'")

        # Check 6: Word count
        if draft.word_count > paragraph_plan.word_budget * 1.2:
            problems.append(f"Word count ({draft.word_count}) exceeds budget ({paragraph_plan.word_budget})")

        is_valid = len(problems) == 0
        return is_valid, problems

    def _check_paraphrase_similarity(self, quote: str, text: str) -> float:
        """Check n-gram overlap between quote and text."""
        # Simple check using SequenceMatcher
        quote_words = quote.lower().split()
        text_words = text.lower().split()

        if not quote_words or not text_words:
            return 0.0

        # Check for exact phrase matches
        quote_str = " ".join(quote_words)
        text_str = " ".join(text_words)

        matcher = SequenceMatcher(None, quote_str, text_str)
        return matcher.ratio()


# ============================================================================
# RETRY LOOP
# ============================================================================

class ParagraphRedrafter:
    """Redraft paragraphs with validation feedback."""

    def __init__(self, generator: ParagraphGenerator, validator: ParagraphValidator):
        self.generator = generator
        self.validator = validator
        self.max_retries = 2

    def draft_with_retry(
        self,
        paragraph_plan,
        paper_notes: list,
        allowed_cite_keys: list[str]
    ) -> ParagraphDraft:
        """
        Draft paragraph with retry loop on validation failure.
        """
        draft = self.generator.generate(paragraph_plan, paper_notes)

        is_valid, problems = self.validator.validate(
            draft, paragraph_plan, paper_notes, allowed_cite_keys
        )

        retry_count = 0
        while not is_valid and retry_count < self.max_retries:
            print(f"[Drafting] Retry {retry_count + 1} for {paragraph_plan.paragraph_id}: {problems[0]}")

            # Redraft with failure in prompt
            draft = self._redraft_with_feedback(
                paragraph_plan,
                paper_notes,
                problems
            )

            is_valid, problems = self.validator.validate(
                draft, paragraph_plan, paper_notes, allowed_cite_keys
            )

            retry_count += 1
            draft.version = retry_count + 1

        if not is_valid:
            draft.flags.extend(problems)
            print(f"[Drafting] Warning: {paragraph_plan.paragraph_id} still has problems after {retry_count} retries")

        return draft

    def _redraft_with_feedback(
        self,
        paragraph_plan,
        paper_notes: list,
        problems: list[str]
    ) -> ParagraphDraft:
        """Redraft with specific feedback in prompt."""
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        evidence_context = self.generator._build_evidence_context(
            paragraph_plan.assigned_paper_ids,
            paper_notes
        )

        allowed_cite_keys = paragraph_plan.assigned_paper_ids

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an academic writer redrafting a paragraph that had validation issues.

PREVIOUS ISSUES TO FIX:
{problems}

PROMPT RULES:
1. Use ONLY the evidence provided.
2. Every claim sentence must cite one or more of: {allowed_cite_keys}
3. Report numbers exactly as given in the evidence.
4. Do NOT invent citation keys - only use those provided.
5. If you cannot support the claim with the evidence, add a flag instead of guessing.

{format_instructions}"""),
            ("human", """Paragraph Plan:
- Claim: {claim}
- Word Budget: {word_budget} words

Evidence:
{evidence_context}

Redraft the paragraph fixing the issues.""")
        ]).partial(
            problems="\n".join(f"- {p}" for p in problems),
            allowed_cite_keys=", ".join(allowed_cite_keys),
            format_instructions=PydanticOutputParser(pydantic_object=ParagraphDraft).get_format_instructions()
        )

        parser = PydanticOutputParser(pydantic_object=ParagraphDraft)
        chain = prompt | self.generator.llm | parser

        try:
            result = chain.invoke({
                "claim": paragraph_plan.claim,
                "word_budget": paragraph_plan.word_budget,
                "evidence_context": evidence_context
            })

            result.paragraph_id = paragraph_plan.paragraph_id
            result.claim = paragraph_plan.claim
            result.word_count = len(result.text.split())

            return result

        except Exception as e:
            return ParagraphDraft(
                paragraph_id=paragraph_plan.paragraph_id,
                claim=paragraph_plan.claim,
                text="",
                flags=[f"Redraft failed: {e}"]
            )


# ============================================================================
# COPY CHECK AND GLOSSARY
# ============================================================================

class CopyChecker:
    """Check for plagiarism and term consistency."""

    def __init__(self):
        self.glossary: dict[str, str] = {}  # term -> canonical form

    def build_glossary(self, paper_notes: list):
        """Build glossary from paper notes."""
        # Common term variants
        term_variants = {
            "large language model": "large language model (LLM)",
            "llm": "large language model (LLM)",
            "gpt-4": "GPT-4",
            "gpt4": "GPT-4",
            "gpt 4": "GPT-4",
            "fine-tuning": "fine-tuning",
            "finetuning": "fine-tuning",
            "fine tuning": "fine-tuning"
        }

        self.glossary = term_variants

    def check_terms(self, draft: ParagraphDraft) -> list[str]:
        """Check for term inconsistencies."""
        issues = []

        text_lower = draft.text.lower()
        for variant, canonical in self.glossary.items():
            if variant in text_lower and variant != canonical.lower():
                issues.append(f"Term inconsistency: '{variant}' should be '{canonical}'")

        return issues

    def check_plagiarism(self, draft: ParagraphDraft, source_texts: list[str]) -> list[str]:
        """Check for excessive copying from sources."""
        issues = []

        for source in source_texts:
            similarity = self._compute_similarity(draft.text, source)
            if similarity > 0.3:  # 30% similarity threshold
                issues.append(f"High similarity ({similarity:.0%}) with source text")

        return issues

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


# ============================================================================
# SECTION ASSEMBLY
# ============================================================================

class SectionAssembler:
    """Assemble sections from paragraphs."""

    def assemble(
        self,
        section_plan,
        paragraph_drafts: list[ParagraphDraft]
    ) -> SectionDraft:
        """Assemble section from drafted paragraphs."""
        section = SectionDraft(
            section_id=section_plan.section_id,
            section_name=section_plan.section_name,
            paragraphs=paragraph_drafts
        )

        # Calculate word count
        section.word_count = sum(p.word_count for p in paragraph_drafts)

        return section

    def add_transitions(self, section: SectionDraft) -> SectionDraft:
        """Add transition sentences between paragraphs."""
        if len(section.paragraphs) < 2:
            return section

        # Add simple transitions
        transition_words = ["Furthermore,", "Additionally,", "Moreover,", "In contrast,", "However,"]

        for i, para in enumerate(section.paragraphs[1:], 1):
            if not any(tw in para.text for tw in transition_words):
                # Prepend transition
                prev_claim = section.paragraphs[i-1].claim
                if "however" in prev_claim.lower() or "contrast" in prev_claim.lower():
                    para.text = "However, " + para.text[0].lower() + para.text[1:]
                else:
                    para.text = "Furthermore, " + para.text[0].lower() + para.text[1:]

        return section


# ============================================================================
# DRAFTING AGENT
# ============================================================================

class DraftingAgent:
    """
    Main agent for Stage 6: Drafting.

    Orchestrates the drafting pipeline.
    """

    def __init__(
        self,
        llm,
        extraction_db_path: str = "extraction.db"
    ):
        self.llm = llm

        # Connect to databases
        from extraction import ExtractionDatabase
        self.extraction_db = ExtractionDatabase(extraction_db_path)

        # Initialize components
        self.generator = ParagraphGenerator(llm=llm)
        self.validator = ParagraphValidator()
        self.redrafter = ParagraphRedrafter(self.generator, self.validator)
        self.copy_checker = CopyChecker()
        self.assembler = SectionAssembler()

    def run(
        self,
        outline,
        paper_notes: list = None
    ) -> Draft:
        """
        Execute the full drafting pipeline.
        """
        print(f"[Stage 6] Starting drafting")

        # Load paper notes if not provided
        if paper_notes is None:
            paper_notes = self.extraction_db.get_all_notes()

        print(f"[Stage 6] Loaded {len(paper_notes)} paper notes")

        # Build glossary
        self.copy_checker.build_glossary(paper_notes)

        # Create draft
        draft = Draft(
            title=outline.title,
            status="drafting"
        )

        # Draft each section
        for section_plan in outline.sections:
            print(f"[Stage 6] Drafting section: {section_plan.section_name}")

            paragraph_drafts = []

            for para_plan in section_plan.paragraphs:
                # Get allowed cite keys
                allowed_cite_keys = para_plan.assigned_paper_ids

                # Draft with retry
                para_draft = self.redrafter.draft_with_retry(
                    para_plan,
                    paper_notes,
                    allowed_cite_keys
                )

                # Copy check
                term_issues = self.copy_checker.check_terms(para_draft)
                if term_issues:
                    para_draft.flags.extend(term_issues)

                paragraph_drafts.append(para_draft)

                # Track placeholders
                if "placeholder" in para_draft.flags:
                    draft.placeholders.append(para_draft.paragraph_id)

                # Track cite keys
                draft.total_cite_keys.extend(para_draft.cite_keys)

            # Assemble section
            section_draft = self.assembler.assemble(section_plan, paragraph_drafts)

            # Add transitions
            section_draft = self.assembler.add_transitions(section_draft)

            draft.sections.append(section_draft)

        # Remove duplicate cite keys
        draft.total_cite_keys = list(set(draft.total_cite_keys))

        # Calculate total word count
        draft.total_word_count = sum(s.word_count for s in draft.sections)

        # Generate title and abstract
        print(f"[Stage 6] Generating title and abstract...")
        draft.title = self._generate_title(outline)
        draft.abstract = self._generate_abstract(outline, draft)

        # Final status
        draft.status = "draft_complete"
        print(f"[Stage 6] Drafting complete: {draft.total_word_count} words")
        print(f"[Stage 6] Citations: {len(draft.total_cite_keys)} unique")
        print(f"[Stage 6] Placeholders: {len(draft.placeholders)}")

        return draft

    def _generate_title(self, outline) -> str:
        """Generate paper title."""
        if outline.title:
            return outline.title

        # Generate from thesis
        thesis = outline.thesis
        # Extract key topic
        if "addresses" in thesis:
            topic = thesis.split("addresses")[1].split("through")[0].strip()
            return f"A Survey of {topic.title()}"

        return "Research Paper"

    def _generate_abstract(self, outline, draft: Draft) -> str:
        """Generate abstract."""
        from langchain_core.prompts import ChatPromptTemplate

        # Build summary from contributions and key findings
        contributions = [c.statement for c in outline.contributions]
        word_count = draft.total_word_count

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are writing an academic abstract. Write 150-200 words summarizing the paper."),
            ("human", """Title: {title}

Thesis: {thesis}

Contributions:
{contributions}

Key findings from {word_count} word analysis.

Write the abstract.""")
        ])

        chain = prompt | self.llm

        try:
            result = chain.invoke({
                "title": draft.title,
                "thesis": outline.thesis,
                "contributions": "\n".join(f"- {c}" for c in contributions),
                "word_count": word_count
            })
            return result.content.strip()
        except Exception as e:
            print(f"[Stage 6] Abstract generation failed: {e}")
            return f"This paper {outline.thesis.lower()}"

    def get_placeholders(self, draft: Draft) -> list[tuple[str, str]]:
        """Get all placeholder paragraphs needing user data."""
        placeholders = []

        for section in draft.sections:
            for para in section.paragraphs:
                if "placeholder" in para.flags:
                    placeholders.append((para.paragraph_id, para.claim))

        return placeholders


# ============================================================================
# VERSIONING AND CONSISTENCY
# ============================================================================

class DraftVersionManager:
    """Manage draft versions."""

    def __init__(self, db_path: str = "draft_versions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create version tables."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS draft_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                draft_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraph_versions (
                paragraph_id TEXT,
                version INTEGER,
                text TEXT,
                flags TEXT,
                created_at TEXT,
                PRIMARY KEY (paragraph_id, version)
            )
        """)

        self.conn.commit()

    def save_version(self, draft: Draft):
        """Save a draft version."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO draft_versions (draft_json, created_at)
            VALUES (?, ?)
        """, (draft.model_dump_json(), draft.created_at))

        # Save paragraph versions
        for section in draft.sections:
            for para in section.paragraphs:
                cursor.execute("""
                    INSERT OR REPLACE INTO paragraph_versions
                    (paragraph_id, version, text, flags, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    para.paragraph_id,
                    para.version,
                    para.text,
                    json.dumps(para.flags),
                    para.created_at
                ))

        self.conn.commit()

    def get_latest_version(self) -> Optional[Draft]:
        """Get the latest draft version."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT draft_json FROM draft_versions
            ORDER BY version_id DESC LIMIT 1
        """)

        row = cursor.fetchone()
        if row:
            return Draft.model_validate_json(row[0])

        return None


# ============================================================================
# DEMO
# ============================================================================

def demo_drafting():
    """Demo function showing drafting."""
    from langchain_openai import ChatOpenAI
    from outline import Outline, SectionPlan, ParagraphPlan, ParagraphPurpose, SourceTag
    from scope import ContributionType
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Create sample outline
    outline = Outline(
        title="LLMs for Education: A Survey",
        thesis="This paper surveys LLM applications in education",
        contribution_type=ContributionType.SURVEY,
        sections=[
            SectionPlan(
                section_id="introduction",
                section_name="Introduction",
                claim="This section introduces the topic",
                paragraphs=[
                    ParagraphPlan(
                        paragraph_id="intro_p1",
                        claim="Large language models have transformed education",
                        purpose=ParagraphPurpose.CONTEXT,
                        source_tag=SourceTag.BACKGROUND_KNOWLEDGE,
                        word_budget=150
                    )
                ]
            )
        ]
    )

    agent = DraftingAgent(llm=llm, extraction_db_path="demo_extraction.db")

    draft = agent.run(outline)

    print(f"\nTitle: {draft.title}")
    print(f"Abstract: {draft.abstract[:200]}...")
    print(f"\nFirst paragraph:")
    print(draft.sections[0].paragraphs[0].text[:300])

    return agent


if __name__ == "__main__":
    demo_drafting()
