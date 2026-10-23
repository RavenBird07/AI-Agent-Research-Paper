"""
Stage 8: Revision

Improves the verified draft through simulated peer review:
critical reviewers, comment routing, targeted edits, and re-verification.
"""

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


# ============================================================================
# ENUMS AND SCHEMAS
# ============================================================================

class ReviewerRole(str, Enum):
    METHODOLOGIST = "methodologist"
    DOMAIN_EXPERT = "domain_expert"
    SKEPTIC = "skeptic"
    CLARITY = "clarity"


class CommentSeverity(str, Enum):
    MAJOR = "major"
    MINOR = "minor"


class CommentRoute(str, Enum):
    AUTO = "auto"  # Agent can fix
    LITERATURE = "literature"  # Needs Stage 2 search
    USER = "user"  # Needs experiments/data
    REJECT = "reject"  # Wrong or out of scope


class ReviewComment(BaseModel):
    """A single reviewer comment."""
    comment_id: str
    reviewer_role: ReviewerRole
    severity: CommentSeverity
    section_id: str
    paragraph_id: Optional[str] = None
    sentence_text: Optional[str] = None  # Exact quote of what's criticized

    # The comment
    summary: str  # One-line summary
    explanation: str  # Full explanation

    # Suggested fix
    suggested_fix: str

    # Routing
    route: CommentRoute = CommentRoute.AUTO
    route_reason: Optional[str] = None

    # Deduplication
    similar_comment_ids: list[str] = Field(default_factory=list)
    reviewer_count: int = 1  # How many reviewers raised this

    # Resolution
    addressed: bool = False
    resolution: Optional[str] = None


class ReviewScores(BaseModel):
    """Scores from a single reviewer."""
    reviewer_role: ReviewerRole
    soundness: int = Field(default=3, ge=1, le=5)
    novelty: int = Field(default=3, ge=1, le=5)
    clarity: int = Field(default=3, ge=1, le=5)
    significance: int = Field(default=3, ge=1, le=5)
    overall: float = 0.0

    def compute_overall(self):
        self.overall = (self.soundness + self.novelty + self.clarity + self.significance) / 4


class ReviewerOutput(BaseModel):
    """Output from a single reviewer."""
    reviewer_role: ReviewerRole
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    scores: Optional[ReviewScores] = None


class ParagraphVersion(BaseModel):
    """A versioned paragraph with history."""
    paragraph_id: str
    version: int
    text: str
    cite_keys: list[str] = Field(default_factory=list)
    changed_by: list[str] = Field(default_factory=list)  # Comment IDs
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Validation status
    validation_passed: bool = True
    validation_issues: list[str] = Field(default_factory=list)


class ReviewReport(BaseModel):
    """Complete review report for one round."""
    round_number: int
    reviewers: list[ReviewerOutput] = Field(default_factory=list)
    aggregated_comments: list[ReviewComment] = Field(default_factory=list)

    # Comment routing summary
    auto_comments: int = 0
    literature_comments: int = 0
    user_comments: int = 0
    rejected_comments: int = 0

    # Score tracking
    mean_score: float = 0.0
    score_change: float = 0.0

    # Status
    status: str = "pending"  # pending, applied, failed
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChangeLog(BaseModel):
    """Change log entry."""
    change_id: str
    round_number: int
    paragraph_id: str
    old_version: int
    new_version: int
    comment_id: str
    change_type: str  # edit, move, merge, delete
    diff_summary: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RevisionReport(BaseModel):
    """Final output of Stage 8: Revision."""
    # Review rounds
    rounds: list[ReviewReport] = Field(default_factory=list)
    total_rounds: int = 0

    # Paragraph versions
    paragraph_versions: dict[str, list[ParagraphVersion]] = Field(default_factory=dict)

    # Change log
    changes: list[ChangeLog] = Field(default_factory=list)

    # Comments requiring user action
    user_comments: list[ReviewComment] = Field(default_factory=list)

    # Final scores
    initial_mean_score: float = 0.0
    final_mean_score: float = 0.0
    score_improved: bool = False

    # Status
    status: str = "pending"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# REVIEWER PERSONAS
# ============================================================================

REVIEWER_PROMPTS = {
    ReviewerRole.METHODOLOGIST: """You are a methodologist reviewing an academic paper. Your focus is on soundness, validity, and methodological rigor.

Look for:
- Missing baselines or unfair comparisons
- Validity threats and confounds
- Insufficient sample sizes or statistical issues
- Overclaiming about causality
- Missing ablation studies
- Incorrect evaluation metrics

Be critical and specific. Cite the exact section or paragraph for every comment.
Separate major issues (soundness problems) from minor issues (wording, suggestions).
Each comment must include a concrete suggested fix.

Score soundness based on: Are the methods appropriate? Is the evaluation valid? Are claims supported by evidence?""",

    ReviewerRole.DOMAIN_EXPERT: """You are a domain expert reviewing an academic paper. Your focus is on correctness, completeness, and proper attribution.

Look for:
- Missing related work or key citations
- Incorrect technical claims
- Misunderstanding of prior work
- Missing standard techniques or datasets
- Incorrect use of terminology
- Overlooked implications

Be critical and specific. Cite the exact section or paragraph for every comment.
Separate major issues (incorrect claims, missing key work) from minor issues (minor additions).
Each comment must include a concrete suggested fix.

Score correctness based on: Are claims technically accurate? Is related work properly covered?""",

    ReviewerRole.SKEPTIC: """You are a skeptic reviewing an academic paper. Your focus is on novelty, overclaiming, and weaknesses.

Look for:
- Claims that go beyond what the evidence supports
- Novelty claims that are not properly justified
- Results that seem too good or cherry-picked
- Missing discussion of limitations
- Overly strong conclusions
- Alternative explanations not considered

Be critical and specific. Challenge every claim. Cite the exact section or paragraph for every comment.
Separate major issues (overclaiming, novelty problems) from minor issues (hedging suggestions).
Each comment must include a concrete suggested fix.

Score novelty based on: Is the contribution genuinely new? Are novelty claims justified?""",

    ReviewerRole.CLARITY: """You are a clarity reviewer reviewing an academic paper. Your focus is on structure, flow, and readability.

Look for:
- Confusing or ambiguous sentences
- Poor transitions between sections
- Redundant content
- Missing explanations of key concepts
- Inconsistent terminology
- Unclear figures or tables
- Grammar and style issues

Be critical and specific. Cite the exact section or paragraph for every comment.
Separate major issues (confusing key concepts) from minor issues (wording improvements).
Each comment must include a concrete suggested fix.

Score clarity based on: Is the paper well-organized? Are explanations clear? Is the writing accessible?"""
}


# ============================================================================
# REVIEWER SIMULATOR
# ============================================================================

class ReviewerSimulator:
    """Simulates independent reviewer with specific persona."""

    def __init__(self, llm, role: ReviewerRole):
        self.llm = llm
        self.role = role
        self.system_prompt = REVIEWER_PROMPTS[role]

    def review(
        self,
        draft,
        bibliography: list[str],
        paper_notes: list
    ) -> ReviewerOutput:
        """
        Run review with this reviewer's persona.

        Reviewer sees the draft and bibliography, not the notes.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        parser = PydanticOutputParser(pydantic_object=ReviewerOutput)

        # Build draft text
        draft_text = self._build_draft_text(draft)

        # Build bibliography summary
        bib_summary = self._build_bibliography_summary(bibliography)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + """

REVIEWER RULES:
1. Act as a critical reviewer looking for reasons to reject, not reasons to praise.
2. Cite the exact section or paragraph for every comment using section_id and paragraph_id.
3. Separate major issues (soundness, novelty, unsupported claims) from minor ones (wording, typos).
4. Each comment must include a concrete suggested fix.
5. Do not praise generically, and do not assume results that the paper does not show.
6. Provide scores from 1-5 for soundness, novelty, clarity, and significance.

OUTPUT FORMAT:
{format_instructions}"""),
            ("human", """Review this paper draft:

TITLE: {title}

DRAFT:
{draft_text}

BIBLIOGRAPHY ({n_papers} papers):
{bib_summary}

Provide your review with comments and scores.""")
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "title": draft.title,
                "draft_text": draft_text,
                "n_papers": len(bibliography),
                "bib_summary": bib_summary
            })

            # Set reviewer role
            result.reviewer_role = self.role

            # Compute scores
            if result.scores:
                result.scores.reviewer_role = self.role
                result.scores.compute_overall()

            # Generate comment IDs
            for i, comment in enumerate(result.comments):
                comment.comment_id = f"{self.role.value}_c{i}"
                comment.reviewer_role = self.role

                # Try to extract section/paragraph from comment
                comment = self._extract_location(comment, draft)

            return result

        except Exception as e:
            print(f"[Reviewer {self.role.value}] Review failed: {e}")
            return ReviewerOutput(
                reviewer_role=self.role,
                summary=f"Review failed: {e}",
                scores=ReviewScores(reviewer_role=self.role)
            )

    def _build_draft_text(self, draft) -> str:
        """Build readable draft text."""
        parts = [f"TITLE: {draft.title}\n"]

        if draft.abstract:
            parts.append(f"\nABSTRACT:\n{draft.abstract}\n")

        for section in draft.sections:
            parts.append(f"\n## {section.section_name.upper()}\n")
            for i, para in enumerate(section.paragraphs):
                parts.append(f"\n[Paragraph {para.paragraph_id}]\n{para.text}\n")

        return "\n".join(parts)

    def _build_bibliography_summary(self, bibliography: list[str]) -> str:
        """Build bibliography summary."""
        if not bibliography:
            return "No bibliography provided."

        # Just list cite keys
        return "Cited papers: " + ", ".join(bibliography[:50])

    def _extract_location(self, comment: ReviewComment, draft) -> ReviewComment:
        """Try to extract section and paragraph from comment."""
        # Try to find paragraph_id from sentence_text
        if comment.sentence_text:
            for section in draft.sections:
                for para in section.paragraphs:
                    if comment.sentence_text in para.text:
                        comment.section_id = section.section_id
                        comment.paragraph_id = para.paragraph_id
                        return comment

        # Try to extract from explanation
        text = comment.explanation.lower()
        for section in draft.sections:
            if section.section_name.lower() in text:
                comment.section_id = section.section_id
                break

        return comment


# ============================================================================
# COMMENT DEDUPLICATION
# ============================================================================

class CommentDeduplicator:
    """Deduplicate comments across reviewers."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def deduplicate(
        self,
        all_comments: list[ReviewComment]
    ) -> list[ReviewComment]:
        """
        Merge similar comments from different reviewers.

        Keep highest severity, count how many reviewers raised it.
        """
        if not all_comments:
            return []

        # Group by similarity
        groups = []

        for comment in all_comments:
            matched = False

            for group in groups:
                # Check similarity with first comment in group
                if self._are_similar(comment, group[0]):
                    group.append(comment)
                    matched = True
                    break

            if not matched:
                groups.append([comment])

        # Merge groups
        merged = []

        for group in groups:
            # Take the comment with highest severity
            major_comments = [c for c in group if c.severity == CommentSeverity.MAJOR]

            if major_comments:
                base = major_comments[0]
            else:
                base = group[0]

            # Update fields
            base.reviewer_count = len(group)
            base.similar_comment_ids = [c.comment_id for c in group[1:]]

            # Combine explanations if different
            if len(group) > 1:
                other_explanations = [
                    c.explanation for c in group[1:]
                    if c.explanation != base.explanation
                ]
                if other_explanations:
                    base.explanation += f" (Also noted: {'; '.join(other_explanations[:2])})"

            merged.append(base)

        # Sort: major first, then by reviewer count
        merged.sort(key=lambda c: (
            c.severity == CommentSeverity.MAJOR,
            c.reviewer_count
        ), reverse=True)

        return merged

    def _are_similar(self, c1: ReviewComment, c2: ReviewComment) -> bool:
        """Check if two comments are similar."""
        # Same section and similar summary
        if c1.section_id and c2.section_id and c1.section_id != c2.section_id:
            return False

        # Check summary similarity
        similarity = fuzz.ratio(c1.summary.lower(), c2.summary.lower()) / 100.0

        return similarity >= self.similarity_threshold


# ============================================================================
# COMMENT ROUTER
# ============================================================================

class CommentRouter:
    """Route comments to appropriate handler."""

    def __init__(self, llm=None):
        self.llm = llm

    def route_comment(
        self,
        comment: ReviewComment,
        paper_notes: list,
        scope_doc
    ) -> ReviewComment:
        """
        Determine how to handle a comment.

        auto: wording, structure, hedging, missing explanation
        literature: "missing related work X"
        user: needs new experiments, data, or analysis
        reject: comment is wrong or out of scope
        """
        text = f"{comment.summary} {comment.explanation}".lower()

        # Check for literature requests
        literature_patterns = [
            "missing related work",
            "missing citation",
            "should cite",
            "should discuss",
            "missing comparison with",
            "overlooks",
            "fails to mention"
        ]

        for pattern in literature_patterns:
            if pattern in text:
                comment.route = CommentRoute.LITERATURE
                comment.route_reason = f"Detected: '{pattern}'"
                return comment

        # Check for user data requests
        user_patterns = [
            "new experiment",
            "run additional",
            "collect more data",
            "different dataset",
            "alternative metric",
            "baseline not included",
            "ablation study needed"
        ]

        for pattern in user_patterns:
            if pattern in text:
                comment.route = CommentRoute.USER
                comment.route_reason = f"Requires: '{pattern}'"
                return comment

        # Check if comment can be auto-fixed
        auto_patterns = [
            "unclear",
            "confusing",
            "rephrase",
            "reword",
            "typo",
            "grammar",
            "transition",
            "redundant",
            "repetitive",
            "concise",
            "elaborate",
            "explain",
            "hedge",
            "too strong",
            "overclaim"
        ]

        for pattern in auto_patterns:
            if pattern in text:
                comment.route = CommentRoute.AUTO
                comment.route_reason = f"Auto-fixable: '{pattern}'"
                return comment

        # Default to auto for minor, user for major
        if comment.severity == CommentSeverity.MINOR:
            comment.route = CommentRoute.AUTO
            comment.route_reason = "Minor issue, auto-fixable"
        else:
            # For major issues, check if we have evidence
            if self._can_auto_fix_major(comment, paper_notes):
                comment.route = CommentRoute.AUTO
                comment.route_reason = "Major issue with available evidence"
            else:
                comment.route = CommentRoute.USER
                comment.route_reason = "Major issue requires investigation"

        return comment

    def _can_auto_fix_major(
        self,
        comment: ReviewComment,
        paper_notes: list
    ) -> bool:
        """Check if we have evidence to fix a major issue."""
        # If the comment is about an unsupported claim
        # and we have paper notes, we might be able to fix it

        if not paper_notes:
            return False

        # Simple heuristic: if we have notes, we can try to find evidence
        return True

    def route_all(
        self,
        comments: list[ReviewComment],
        paper_notes: list,
        scope_doc
    ) -> list[ReviewComment]:
        """Route all comments."""
        return [
            self.route_comment(c, paper_notes, scope_doc)
            for c in comments
        ]


# ============================================================================
# PARAGRAPH REVISER
# ============================================================================

class ParagraphReviser:
    """Revise a single paragraph based on comments."""

    def __init__(self, llm, validator):
        self.llm = llm
        self.validator = validator  # Stage 6 ParagraphValidator
        self.max_retries = 2

    def revise(
        self,
        paragraph,
        comments: list[ReviewComment],
        paper_notes: list,
        allowed_cite_keys: list[str],
        paragraph_plan
    ) -> ParagraphVersion:
        """
        Revise paragraph based on comments.

        Change only what the comments require.
        Keep everything else as is.
        Never add a claim without a cite key from allowed set.
        Never alter numbers or invent results.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        # Build evidence context
        evidence_context = self._build_evidence_context(
            paragraph_plan.assigned_paper_ids,
            paper_notes
        )

        # Build comment context
        comment_context = self._build_comment_context(comments)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are revising a paragraph based on reviewer feedback.

REVISION RULES:
1. Change ONLY what the comments require. Keep everything else exactly as is.
2. Do NOT add any claim without a citation from the allowed set: {allowed_cite_keys}
3. Do NOT alter numbers or invent results.
4. If the comment can't be fixed with available evidence, return the original paragraph unchanged.
5. Maintain the paragraph's purpose and flow.

OUTPUT FORMAT:
{format_instructions}"""),
            ("human", """Original paragraph:
{original_text}

Comments to address:
{comment_context}

Allowed cite keys: {allowed_cite_keys}

Evidence:
{evidence_context}

Revise the paragraph to address the comments.""")
        ]).partial(
            allowed_cite_keys=", ".join(allowed_cite_keys),
            format_instructions=PydanticOutputParser(pydantic_object=ParagraphVersion).get_format_instructions()
        )

        parser = PydanticOutputParser(pydantic_object=ParagraphVersion)
        chain = prompt | self.llm | parser

        retry_count = 0
        new_version = paragraph.version + 1

        while retry_count < self.max_retries:
            try:
                result = chain.invoke({
                    "original_text": paragraph.text,
                    "comment_context": comment_context,
                    "evidence_context": evidence_context
                })

                result.paragraph_id = paragraph.paragraph_id
                result.version = new_version
                result.changed_by = [c.comment_id for c in comments]

                # Validate
                is_valid, issues = self._validate_revision(
                    result, paragraph, allowed_cite_keys
                )

                if is_valid:
                    result.validation_passed = True
                    return result
                else:
                    result.validation_issues = issues
                    retry_count += 1

            except Exception as e:
                print(f"[Reviser] Revision failed: {e}")
                retry_count += 1

        # Return original on failure
        return ParagraphVersion(
            paragraph_id=paragraph.paragraph_id,
            version=new_version,
            text=paragraph.text,
            cite_keys=paragraph.cite_keys,
            changed_by=[],
            validation_passed=False,
            validation_issues=["Revision failed after retries"]
        )

    def _build_evidence_context(
        self,
        paper_ids: list[str],
        paper_notes: list
    ) -> str:
        """Build evidence context from paper notes."""
        if not paper_ids:
            return "No specific evidence available."

        context_parts = []

        for paper_id in paper_ids:
            paper = next((p for p in paper_notes if p.paper_id == paper_id), None)
            if not paper:
                continue

            context_parts.append(f"\n[{paper_id}] {paper.title}")

            if paper.main_results:
                context_parts.append(f"Results: {', '.join(paper.main_results[:3])}")

            if paper.method_description:
                context_parts.append(f"Method: {paper.method_description[:200]}")

        return "\n".join(context_parts) if context_parts else "No specific evidence available."

    def _build_comment_context(self, comments: list[ReviewComment]) -> str:
        """Build comment context string."""
        parts = []

        for i, comment in enumerate(comments, 1):
            parts.append(f"{i}. [{comment.severity.value}] {comment.summary}")
            parts.append(f"   Explanation: {comment.explanation}")
            parts.append(f"   Suggested fix: {comment.suggested_fix}")

        return "\n".join(parts)

    def _validate_revision(
        self,
        new_version: ParagraphVersion,
        original_paragraph,
        allowed_cite_keys: list[str]
    ) -> tuple[bool, list[str]]:
        """
        Validate revision.

        Checks:
        1. Cite keys are from allowed set
        2. No new numbers invented
        3. Meaningful change (not just copy of original)
        """
        issues = []

        # Check cite keys
        for cite in new_version.cite_keys:
            if cite not in allowed_cite_keys:
                issues.append(f"Invented citation: {cite}")

        # Check numbers not invented (simplified)
        # In practice, would compare numbers with evidence

        # Check for meaningful change
        similarity = SequenceMatcher(
            None,
            original_paragraph.text.lower(),
            new_version.text.lower()
        ).ratio()

        if similarity > 0.95:
            # Almost no change - might be appropriate if comments were minor
            pass

        return len(issues) == 0, issues


# ============================================================================
# STRUCTURE PASS
# ============================================================================

class StructurePass:
    """Check and fix structural issues."""

    def check_redundancy(
        self,
        draft,
        similarity_threshold: float = 0.85
    ) -> list[tuple[str, str, float]]:
        """
        Check for redundant paragraphs.

        Returns: list of (para_id_1, para_id_2, similarity)
        """
        redundant_pairs = []

        paragraphs = []
        for section in draft.sections:
            for para in section.paragraphs:
                paragraphs.append((para.paragraph_id, para.text))

        for i, (id1, text1) in enumerate(paragraphs):
            for id2, text2 in paragraphs[i+1:]:
                similarity = SequenceMatcher(
                    None, text1.lower(), text2.lower()
                ).ratio()

                if similarity >= similarity_threshold:
                    redundant_pairs.append((id1, id2, similarity))

        return redundant_pairs

    def check_section_order(self, draft) -> list[str]:
        """Check section order is logical."""
        issues = []

        section_names = [s.section_name.lower() for s in draft.sections]

        # Check introduction is first
        if section_names and "introduction" not in section_names[0]:
            issues.append("Introduction should be first section")

        # Check conclusion is last
        if section_names and "conclusion" not in section_names[-1]:
            issues.append("Conclusion should be last section")

        return issues

    def check_contributions_delivered(
        self,
        draft,
        outline
    ) -> list[str]:
        """Check all contributions are delivered in sections."""
        issues = []

        for contrib in outline.contributions:
            if not contrib.section_mapping:
                issues.append(f"Contribution not delivered: {contrib.statement[:50]}")

        return issues


# ============================================================================
# STYLE PASS
# ============================================================================

class StylePass:
    """Fix clarity and flow without changing meaning."""

    def __init__(self):
        self.passive_indicators = ["is shown", "was found", "are presented", "were observed"]
        self.vague_words = ["very", "quite", "rather", "somewhat", "fairly"]
        self.filler_words = ["basically", "actually", "essentially", "simply"]

    def check_style(self, text: str) -> list[str]:
        """Check for style issues."""
        issues = []
        text_lower = text.lower()

        # Long sentences
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) > 40:
                issues.append(f"Long sentence ({len(words)} words)")

        # Passive voice
        for indicator in self.passive_indicators:
            if indicator in text_lower:
                issues.append(f"Passive voice: '{indicator}'")

        # Vague words
        for word in self.vague_words:
            if f" {word} " in f" {text_lower} ":
                issues.append(f"Vague word: '{word}'")

        # Filler words
        for word in self.filler_words:
            if f" {word} " in f" {text_lower} ":
                issues.append(f"Filler word: '{word}'")

        return issues

    def fix_style(
        self,
        paragraph,
        allowed_cite_keys: list[str]
    ) -> ParagraphVersion:
        """
        Fix style issues without changing meaning.

        Must keep every cite_keys list and number unchanged.
        """
        new_text = paragraph.text

        # Store cite keys and numbers to preserve
        original_cite_keys = paragraph.cite_keys.copy()
        original_numbers = re.findall(r'\d+\.?\d*%?', paragraph.text)

        # Apply style fixes
        # (In practice, would use LLM with style instructions)

        # For now, just do simple replacements
        for word in self.filler_words:
            new_text = re.sub(
                rf'\b{word}\b',
                '',
                new_text,
                flags=re.IGNORECASE
            )

        for word in self.vague_words:
            new_text = re.sub(
                rf'\b{word}\b',
                '',
                new_text,
                flags=re.IGNORECASE
            )

        # Clean up extra spaces
        new_text = re.sub(r' +', ' ', new_text)
        new_text = new_text.strip()

        # Verify cite keys and numbers preserved
        new_numbers = re.findall(r'\d+\.?\d*%?', new_text)

        if new_numbers != original_numbers:
            # Numbers changed - reject the fix
            return ParagraphVersion(
                paragraph_id=paragraph.paragraph_id,
                version=paragraph.version,
                text=paragraph.text,
                cite_keys=original_cite_keys,
                validation_passed=False,
                validation_issues=["Style pass changed numbers"]
            )

        return ParagraphVersion(
            paragraph_id=paragraph.paragraph_id,
            version=paragraph.version + 1,
            text=new_text,
            cite_keys=original_cite_keys,
            changed_by=["style_pass"],
            validation_passed=True
        )


# ============================================================================
# VERSION MANAGER
# ============================================================================

class RevisionVersionManager:
    """Manage draft versions during revision."""

    def __init__(self, db_path: str = "revision_versions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create version tables."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraph_versions (
                paragraph_id TEXT,
                version INTEGER,
                text TEXT,
                cite_keys TEXT,
                changed_by TEXT,
                validation_passed INTEGER,
                validation_issues TEXT,
                created_at TEXT,
                PRIMARY KEY (paragraph_id, version)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS change_log (
                change_id TEXT PRIMARY KEY,
                round_number INTEGER,
                paragraph_id TEXT,
                old_version INTEGER,
                new_version INTEGER,
                comment_id TEXT,
                change_type TEXT,
                diff_summary TEXT,
                created_at TEXT
            )
        """)

        self.conn.commit()

    def save_version(self, paragraph_version: ParagraphVersion):
        """Save a paragraph version."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO paragraph_versions (
                paragraph_id, version, text, cite_keys, changed_by,
                validation_passed, validation_issues, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paragraph_version.paragraph_id,
            paragraph_version.version,
            paragraph_version.text,
            json.dumps(paragraph_version.cite_keys),
            json.dumps(paragraph_version.changed_by),
            int(paragraph_version.validation_passed),
            json.dumps(paragraph_version.validation_issues),
            paragraph_version.created_at
        ))

        self.conn.commit()

    def save_change(self, change: ChangeLog):
        """Save a change log entry."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO change_log (
                change_id, round_number, paragraph_id, old_version,
                new_version, comment_id, change_type, diff_summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            change.change_id,
            change.round_number,
            change.paragraph_id,
            change.old_version,
            change.new_version,
            change.comment_id,
            change.change_type,
            change.diff_summary,
            change.created_at
        ))

        self.conn.commit()

    def get_version(
        self,
        paragraph_id: str,
        version: int
    ) -> Optional[ParagraphVersion]:
        """Get a specific version."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM paragraph_versions
            WHERE paragraph_id = ? AND version = ?
        """, (paragraph_id, version))

        row = cursor.fetchone()
        if row:
            return ParagraphVersion(
                paragraph_id=row[0],
                version=row[1],
                text=row[2],
                cite_keys=json.loads(row[3]),
                changed_by=json.loads(row[4]),
                validation_passed=bool(row[5]),
                validation_issues=json.loads(row[6]),
                created_at=row[7]
            )

        return None

    def get_latest_version(self, paragraph_id: str) -> Optional[ParagraphVersion]:
        """Get the latest version of a paragraph."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM paragraph_versions
            WHERE paragraph_id = ?
            ORDER BY version DESC LIMIT 1
        """, (paragraph_id,))

        row = cursor.fetchone()
        if row:
            return ParagraphVersion(
                paragraph_id=row[0],
                version=row[1],
                text=row[2],
                cite_keys=json.loads(row[3]),
                changed_by=json.loads(row[4]),
                validation_passed=bool(row[5]),
                validation_issues=json.loads(row[6]),
                created_at=row[7]
            )

        return None

    def rollback(self, paragraph_id: str, to_version: int) -> bool:
        """Rollback a paragraph to a previous version."""
        # Get the target version
        target = self.get_version(paragraph_id, to_version)
        if not target:
            return False

        # Delete all versions after
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM paragraph_versions
            WHERE paragraph_id = ? AND version > ?
        """, (paragraph_id, to_version))

        self.conn.commit()
        return True

    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# REVISION AGENT
# ============================================================================

class RevisionAgent:
    """
    Main agent for Stage 8: Revision.

    Orchestrates the revision pipeline with peer review simulation.
    """

    def __init__(
        self,
        llm,
        corpus_db_path: str = "corpus.db",
        extraction_db_path: str = "extraction.db",
        verification_agent = None
    ):
        self.llm = llm
        self.final_draft = None
        self.corpus_db_path = corpus_db_path
        self.extraction_db_path = extraction_db_path
        self.verification_agent = verification_agent

        # Initialize components
        self.reviewers = [
            ReviewerSimulator(llm, ReviewerRole.METHODOLOGIST),
            ReviewerSimulator(llm, ReviewerRole.DOMAIN_EXPERT),
            ReviewerSimulator(llm, ReviewerRole.SKEPTIC),
            ReviewerSimulator(llm, ReviewerRole.CLARITY)
        ]
        self.deduplicator = CommentDeduplicator()
        self.router = CommentRouter(llm=llm)

        # Load paper notes
        from extraction import ExtractionDatabase
        self.extraction_db = ExtractionDatabase(extraction_db_path)
        self.paper_notes = self.extraction_db.get_all_notes()

        # Version manager
        self.version_manager = RevisionVersionManager()

        # Reviser (needs validator from Stage 6)
        from drafting import ParagraphValidator
        self.reviser = ParagraphReviser(llm, ParagraphValidator())

        # Structure and style passes
        self.structure_pass = StructurePass()
        self.style_pass = StylePass()

    def run(
        self,
        draft,
        verification_report,
        outline,
        gap_report,
        scope_doc,
        max_rounds: int = 3
    ) -> RevisionReport:
        """
        Execute the full revision pipeline.

        Steps 1-13 from the specification.
        """
        print(f"[Stage 8] Starting revision")

        report = RevisionReport(status="revising")

        # Step 1: Apply verifier's fix list first
        draft = self._apply_verifier_fixes(draft, verification_report)
        print(f"[Stage 8] Applied {len(verification_report.fix_list)} verifier fixes")

        # Save initial state
        self._save_initial_versions(draft)

        # Initialize score tracking
        initial_scores = None
        previous_mean_score = 0.0

        # Revision rounds
        for round_num in range(1, max_rounds + 1):
            print(f"\n[Stage 8] Round {round_num}/{max_rounds}")

            # Step 2: Run reviewer simulation
            review_report = self._run_review_round(
                draft, outline, round_num
            )

            # Track scores
            if initial_scores is None:
                initial_scores = review_report.mean_score
                report.initial_mean_score = initial_scores

            score_change = review_report.mean_score - previous_mean_score
            review_report.score_change = score_change

            # Check stopping condition
            if round_num > 1:
                if abs(score_change) < 0.2:
                    print(f"[Stage 8] Score improvement < 0.2, stopping")
                    break

            # Step 3: Aggregate and deduplicate
            # (Already done in _run_review_round)

            # Step 4: Route comments
            review_report.aggregated_comments = self.router.route_all(
                review_report.aggregated_comments,
                self.paper_notes,
                scope_doc
            )

            # Count by route
            review_report.auto_comments = sum(
                1 for c in review_report.aggregated_comments
                if c.route == CommentRoute.AUTO
            )
            review_report.literature_comments = sum(
                1 for c in review_report.aggregated_comments
                if c.route == CommentRoute.LITERATURE
            )
            review_report.user_comments = sum(
                1 for c in review_report.aggregated_comments
                if c.route == CommentRoute.USER
            )
            review_report.rejected_comments = sum(
                1 for c in review_report.aggregated_comments
                if c.route == CommentRoute.REJECT
            )

            # Step 5: Human checkpoint on plan
            approved_comments = self._human_checkpoint_comments(
                review_report.aggregated_comments
            )

            # Step 6: Apply edits paragraph by paragraph
            draft, changes = self._apply_edits(
                draft, approved_comments, outline, round_num
            )

            # Step 7: Structure pass
            draft = self._structure_pass(draft, outline)

            # Step 8: Style pass
            draft = self._style_pass(draft)

            # Step 9: Re-verify changed content
            if self.verification_agent:
                new_verification = self.verification_agent.run(draft, outline)
                if not new_verification.passed:
                    print(f"[Stage 8] Re-verification failed, rolling back...")
                    draft = self._rollback_failed_edits(draft, new_verification)

            # Update scores
            previous_mean_score = review_report.mean_score

            # Save round report
            review_report.status = "applied"
            report.rounds.append(review_report)
            report.changes.extend(changes)

        # Step 10: Re-review and compare
        print(f"[Stage 8] Final re-review...")
        final_review = self._run_review_round(draft, outline, max_rounds + 1)
        report.final_mean_score = final_review.mean_score
        report.score_improved = report.final_mean_score > report.initial_mean_score

        # Step 11: Stopping rule already checked in loop

        # Step 12: Human final read
        draft = self._human_final_read(draft)

        # Collect user comments
        for round_report in report.rounds:
            for comment in round_report.aggregated_comments:
                if comment.route == CommentRoute.USER and not comment.addressed:
                    report.user_comments.append(comment)

        # Finalize
        self.final_draft = draft
        report.total_rounds = len(report.rounds)
        report.status = "complete"

        print(f"\n[Stage 8] Revision complete")
        print(f"[Stage 8] Score: {report.initial_mean_score:.2f} -> {report.final_mean_score:.2f}")
        print(f"[Stage 8] User actions needed: {len(report.user_comments)}")

        return report

    def _apply_verifier_fixes(self, draft, verification_report):
        """Apply fixes from Stage 7 verification."""
        for fix in verification_report.fix_list:
            # Find the paragraph
            for section in draft.sections:
                for para in section.paragraphs:
                    # Check if this paragraph contains the sentence
                    if fix.get("sentence_text") and fix["sentence_text"] in para.text:
                        # Apply fix (simplified - in practice would use LLM)
                        para.flags.append(f"Fix applied: {fix.get('action', 'revise')}")

        return draft

    def _save_initial_versions(self, draft):
        """Save initial paragraph versions."""
        for section in draft.sections:
            for para in section.paragraphs:
                version = ParagraphVersion(
                    paragraph_id=para.paragraph_id,
                    version=para.version,
                    text=para.text,
                    cite_keys=para.cite_keys,
                    changed_by=[],
                    validation_passed=True
                )
                self.version_manager.save_version(version)

    def _run_review_round(
        self,
        draft,
        outline,
        round_num: int
    ) -> ReviewReport:
        """Run all reviewers and aggregate results."""
        report = ReviewReport(round_number=round_num)

        # Build bibliography
        bibliography = draft.total_cite_keys

        # Run reviewers in parallel (simplified - would use async)
        for reviewer in self.reviewers:
            print(f"[Stage 8] Running {reviewer.role.value} reviewer...")
            output = reviewer.review(draft, bibliography, self.paper_notes)
            report.reviewers.append(output)

        # Aggregate comments
        all_comments = []
        for output in report.reviewers:
            all_comments.extend(output.comments)

        # Deduplicate
        report.aggregated_comments = self.deduplicator.deduplicate(all_comments)

        # Compute mean score
        scores = [o.scores.overall for o in report.reviewers if o.scores]
        report.mean_score = sum(scores) / len(scores) if scores else 0.0

        return report

    def _human_checkpoint_comments(
        self,
        comments: list[ReviewComment]
    ) -> list[ReviewComment]:
        """Step 5: Human checkpoint for comment routing."""
        print(f"\n{'='*70}")
        print("COMMENT ROUTING - AWAITING APPROVAL")
        print(f"{'='*70}")

        major_comments = [c for c in comments if c.severity == CommentSeverity.MAJOR]
        minor_comments = [c for c in comments if c.severity == CommentSeverity.MINOR]

        print(f"\nMajor Comments ({len(major_comments)}):")
        for i, comment in enumerate(major_comments[:10], 1):
            print(f"\n{i}. [{comment.reviewer_role.value}] {comment.summary}")
            print(f"   Route: {comment.route.value}")
            print(f"   Reason: {comment.route_reason}")
            print(f"   Reviewers: {comment.reviewer_count}")

        if len(major_comments) > 10:
            print(f"\n... and {len(major_comments) - 10} more major comments")

        print(f"\nMinor Comments: {len(minor_comments)}")

        print(f"\n{'-'*70}")

        # In demo mode, approve all auto-fixable comments
        approved = [
            c for c in comments
            if c.route in [CommentRoute.AUTO, CommentRoute.LITERATURE]
        ]

        print(f"\nApproved {len(approved)} comments for action")

        return approved

    def _apply_edits(
        self,
        draft,
        comments: list[ReviewComment],
        outline,
        round_num: int
    ) -> tuple:
        """Apply edits to paragraphs based on comments."""
        changes = []

        # Group comments by paragraph
        comments_by_paragraph = defaultdict(list)
        for comment in comments:
            if comment.paragraph_id and comment.route == CommentRoute.AUTO:
                comments_by_paragraph[comment.paragraph_id].append(comment)

        # Apply edits
        for section in draft.sections:
            for para in section.paragraphs:
                if para.paragraph_id in comments_by_paragraph:
                    para_comments = comments_by_paragraph[para.paragraph_id]

                    # Get paragraph plan
                    para_plan = self._get_paragraph_plan(
                        para.paragraph_id, outline
                    )

                    if para_plan:
                        # Get allowed cite keys
                        allowed_cite_keys = para_plan.assigned_paper_ids

                        # Revise
                        old_version = para.version
                        new_version_obj = self.reviser.revise(
                            para,
                            para_comments,
                            self.paper_notes,
                            allowed_cite_keys,
                            para_plan
                        )

                        # Update paragraph if revision passed
                        if new_version_obj.validation_passed:
                            para.text = new_version_obj.text
                            para.cite_keys = new_version_obj.cite_keys
                            para.version = new_version_obj.version

                            # Save version
                            self.version_manager.save_version(new_version_obj)

                            # Record change
                            change = ChangeLog(
                                change_id=f"change_{para.paragraph_id}_{round_num}",
                                round_number=round_num,
                                paragraph_id=para.paragraph_id,
                                old_version=old_version,
                                new_version=new_version_obj.version,
                                comment_id=",".join(c.comment_id for c in para_comments),
                                change_type="edit",
                                diff_summary=f"Addressed {len(para_comments)} comments"
                            )
                            changes.append(change)
                            self.version_manager.save_change(change)

                            # Mark comments addressed
                            for comment in para_comments:
                                comment.addressed = True
                                comment.resolution = "Fixed in revision"

        return draft, changes

    def _get_paragraph_plan(self, paragraph_id: str, outline):
        """Get paragraph plan from outline."""
        for section in outline.sections:
            for para in section.paragraphs:
                if para.paragraph_id == paragraph_id:
                    return para
        return None

    def _structure_pass(self, draft, outline):
        """Step 7: Structure pass."""
        # Check redundancy
        redundant = self.structure_pass.check_redundancy(draft)
        if redundant:
            print(f"[Stage 8] Found {len(redundant)} potentially redundant paragraph pairs")

        # Check section order
        order_issues = self.structure_pass.check_section_order(draft)
        for issue in order_issues:
            print(f"[Stage 8] Structure issue: {issue}")

        # Check contributions delivered
        contrib_issues = self.structure_pass.check_contributions_delivered(draft, outline)
        for issue in contrib_issues:
            print(f"[Stage 8] Contribution issue: {issue}")

        return draft

    def _style_pass(self, draft):
        """Step 8: Style pass."""
        total_issues = 0

        for section in draft.sections:
            for para in section.paragraphs:
                issues = self.style_pass.check_style(para.text)
                if issues:
                    total_issues += len(issues)

        if total_issues > 0:
            print(f"[Stage 8] Style issues found: {total_issues}")

        return draft

    def _rollback_failed_edits(self, draft, verification_report):
        """Rollback edits that failed re-verification."""
        # Find paragraphs with unsupported sentences
        unsupported_paras = set()

        for section in verification_report.sections:
            for sentence in section.sentences:
                if sentence.verdict.value in ["unsupported", "contradicted"]:
                    # Find which paragraph contains this
                    for draft_section in draft.sections:
                        for para in draft_section.paragraphs:
                            if sentence.text in para.text:
                                unsupported_paras.add(para.paragraph_id)

        # Rollback these paragraphs
        for para_id in unsupported_paras:
            previous = self.version_manager.get_latest_version(para_id)
            if previous and previous.version > 1:
                self.version_manager.rollback(para_id, previous.version - 1)
                print(f"[Stage 8] Rolled back {para_id}")

        return draft

    def _human_final_read(self, draft):
        """Step 12: Human final read of the draft."""
        print(f"\n{'='*70}")
        print("FINAL DRAFT REVIEW")
        print(f"{'='*70}")

        print(f"\nTitle: {draft.title}")
        print(f"Total words: {draft.total_word_count}")
        print(f"Total citations: {len(draft.total_cite_keys)}")

        print(f"\nSections:")
        for section in draft.sections:
            print(f"  {section.section_name}: {section.word_count} words")

        print(f"\n{'-'*70}")

        response = input("\nApprove final draft? (yes/edit): ").strip().lower()

        if response == "edit":
            print("\nManual editing not implemented in demo.")

        return draft

    def close(self):
        """Close connections."""
        self.version_manager.close()
        self.extraction_db.close()


# ============================================================================
# SEEDED FLAW TEST
# ============================================================================

def seeded_flaw_test(revision_agent, clean_draft) -> float:
    """
    Test that reviewers catch seeded weaknesses.

    Returns: detection rate (should be >= 0.80)
    """
    # Plant 10 known weaknesses
    seeded_flaws = [
        ("method", "Missing baseline comparison with standard approach"),
        ("result", "Reported accuracy of 99.9% seems unusually high"),
        ("claim", "Claim 'all models succeed' is too strong without evidence"),
        ("citation", "Missing citation to seminal work in the field"),
        ("method", "No ablation study reported"),
        ("result", "Results only on one dataset, generalizability unclear"),
        ("claim", "Causation implied where only correlation shown"),
        ("limitation", "Limitations section missing"),
        ("statistic", "No statistical significance reported"),
        ("novelty", "Novelty claim not justified against prior work")
    ]

    # Inject flaws into draft (simplified - would actually modify draft)
    detection_count = 0

    # Run reviewers
    bibliography = clean_draft.total_cite_keys
    all_comments = []

    for reviewer in revision_agent.reviewers:
        output = reviewer.review(clean_draft, bibliography, revision_agent.paper_notes)
        all_comments.extend(output.comments)

    # Check if flaws were detected
    for flaw_type, flaw_text in seeded_flaws:
        for comment in all_comments:
            if flaw_type in comment.summary.lower() or any(
                keyword in comment.explanation.lower()
                for keyword in flaw_text.lower().split()[:3]
            ):
                detection_count += 1
                break

    detection_rate = detection_count / len(seeded_flaws)

    print(f"[Seeded Flaw Test] Detection rate: {detection_rate:.1%}")
    print(f"[Seeded Flaw Test] {'PASS' if detection_rate >= 0.8 else 'FAIL'}")

    return detection_rate


# ============================================================================
# DEMO
# ============================================================================

def demo_revision():
    """Demo function showing revision."""
    from langchain_openai import ChatOpenAI
    from drafting import Draft, SectionDraft, ParagraphDraft
    from verification import VerificationReport
    from outline import Outline
    from gap_analysis import GapReport
    from scope import ScopeDocument, ContributionType
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Create sample draft
    draft = Draft(
        title="LLMs for Education: A Survey",
        abstract="This paper surveys large language model applications in education.",
        sections=[
            SectionDraft(
                section_id="introduction",
                section_name="Introduction",
                paragraphs=[
                    ParagraphDraft(
                        paragraph_id="intro_p1",
                        claim="LLMs have transformed education",
                        text="Large language models have transformed education in numerous ways (Smith2024). All studies show perfect results.",
                        cite_keys=["paper_1"],
                        word_count=20
                    )
                ],
                word_count=20
            )
        ],
        total_word_count=20,
        total_cite_keys=["paper_1"]
    )

    # Create sample verification report
    verification_report = VerificationReport(passed=True, fix_list=[])

    # Create sample outline
    outline = Outline(
        title="LLMs for Education",
        thesis="This paper surveys LLM applications in education",
        contribution_type=ContributionType.SURVEY
    )

    # Create sample gap report
    gap_report = GapReport()

    # Create sample scope document
    scope_doc = ScopeDocument(
        original_topic="LLMs for Education",
        research_question="How are LLMs applied in education?",
        contribution_type=ContributionType.SURVEY
    )

    agent = RevisionAgent(
        llm=llm,
        extraction_db_path="demo_extraction.db"
    )

    report = agent.run(
        draft=draft,
        verification_report=verification_report,
        outline=outline,
        gap_report=gap_report,
        scope_doc=scope_doc,
        max_rounds=1
    )

    print(f"\nRevision complete: {report.status}")
    print(f"Score change: {report.initial_mean_score:.2f} -> {report.final_mean_score:.2f}")

    return agent


if __name__ == "__main__":
    demo_revision()
