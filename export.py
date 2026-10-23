"""
Stage 9: Export

Turns the verified, revised draft into a submission-ready package:
compiled PDF, LaTeX source, BibTeX, and reproducibility bundle.
Export changes formatting only - never changes verified content.
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import requests
from pydantic import BaseModel, Field


# ============================================================================
# ENUMS AND SCHEMAS
# ============================================================================

class VenueTemplate(str, Enum):
    IEEE = "ieee"
    ACM = "acm"
    SPRINGER_LNCS = "springer_lncs"
    ELSEVIER = "elsevier"
    ARXIV = "arxiv"
    GENERIC = "generic"


class ExportStatus(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    COMPILED = "compiled"
    FAILED = "failed"


class BibEntryStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    MISSING = "missing"


class BibEntry(BaseModel):
    """A BibTeX entry."""
    cite_key: str
    entry_type: str  # article, inproceedings, etc.
    fields: dict[str, str] = Field(default_factory=dict)
    paper_id: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    status: BibEntryStatus = BibEntryStatus.UNVERIFIED
    source: str = "unknown"  # crossref, stored, fallback


class ExportConfig(BaseModel):
    """Configuration for export."""
    # Venue
    venue_template: VenueTemplate = VenueTemplate.GENERIC
    citation_style: str = "numeric"  # numeric, authoryear
    page_limit: Optional[int] = None

    # Front matter (user-provided)
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    corresponding_email: Optional[str] = None

    # Statements (user-provided)
    ai_disclosure: Optional[str] = None
    data_availability: Optional[str] = None
    code_availability: Optional[str] = None
    ethics_statement: Optional[str] = None
    conflicts_of_interest: Optional[str] = None
    acknowledgements: Optional[str] = None
    funding: Optional[str] = None

    # Options
    generate_docx: bool = False
    include_supplementary: bool = True


class FigureExport(BaseModel):
    """Export info for a figure."""
    figure_id: str
    file_path: str
    caption: str
    label: str
    file_exists: bool = False
    resolution_ok: bool = True


class TableExport(BaseModel):
    """Export info for a table."""
    table_id: str
    latex_code: str
    caption: str
    label: str


class CompileResult(BaseModel):
    """Result of LaTeX compilation."""
    success: bool
    pdf_path: Optional[str] = None
    log_path: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    undefined_cites: list[str] = Field(default_factory=list)
    undefined_refs: list[str] = Field(default_factory=list)
    overfull_boxes: list[str] = Field(default_factory=list)
    page_count: int = 0


class ExportReport(BaseModel):
    """Final output of Stage 9: Export."""
    # Preflight
    preflight_passed: bool = False
    preflight_issues: list[str] = Field(default_factory=list)

    # Files
    tex_path: Optional[str] = None
    bib_path: Optional[str] = None
    pdf_path: Optional[str] = None
    docx_path: Optional[str] = None

    # Bibliography
    bib_entries: list[BibEntry] = Field(default_factory=list)
    unverified_refs: list[str] = Field(default_factory=list)

    # Figures and tables
    figures: list[FigureExport] = Field(default_factory=list)
    tables: list[TableExport] = Field(default_factory=list)

    # Compile
    compile_result: Optional[CompileResult] = None

    # Post-compile checks
    page_count: int = 0
    page_limit_ok: bool = True
    pdf_has_placeholders: bool = False
    missing_refs: list[str] = Field(default_factory=list)

    # Survey extras
    prisma_flow: Optional[dict] = None
    search_queries: list[str] = Field(default_factory=list)
    reproducibility_log_path: Optional[str] = None

    # Reproducibility
    reproducibility_zip: Optional[str] = None
    git_tag: Optional[str] = None

    # Status
    status: ExportStatus = ExportStatus.BLOCKED
    ready_to_submit: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# PREFLIGHT GATE
# ============================================================================

class PreflightChecker:
    """Check all prerequisites before export."""

    def __init__(self, verified_hash: Optional[str] = None):
        self.verified_hash = verified_hash

    def check(
        self,
        draft,
        verification_report,
        revision_report,
        draft_hash: Optional[str] = None
    ) -> tuple[bool, list[str]]:
        """
        Run preflight checks.

        Returns: (passed, list of issues)
        """
        issues = []

        # Check 1: Stage 7 passed
        if not verification_report.passed:
            issues.append("Stage 7 verification not passed")

        # Check 2: Draft hash matches verified version
        if self.verified_hash and draft_hash:
            if draft_hash != self.verified_hash:
                issues.append(
                    f"Draft changed after verification. "
                    f"Expected: {self.verified_hash[:12]}, Got: {draft_hash[:12]}"
                )
        elif self.verified_hash:
            issues.append("Draft hash not provided for comparison")

        # Check 3: No placeholders remain
        placeholders = self._find_placeholders(draft)
        if placeholders:
            issues.append(f"Placeholders remain: {placeholders}")

        # Check 4: No unresolved user comments
        if revision_report:
            unresolved = [
                c for c in revision_report.user_comments
                if not c.addressed and c.route.value == "user"
            ]
            if unresolved:
                issues.append(
                    f"Unresolved user comments: {len(unresolved)}. "
                    "Waive them in config to proceed."
                )

        passed = len(issues) == 0
        return passed, issues

    def _find_placeholders(self, draft) -> list[str]:
        """Find any remaining placeholders in the draft."""
        placeholders = []

        for section in draft.sections:
            for para in section.paragraphs:
                if "%% USER DATA REQUIRED" in para.text:
                    placeholders.append(para.paragraph_id)
                if "[[" in para.text and "]]" in para.text:
                    placeholders.append(f"{para.paragraph_id}: bracketed placeholder")

        return placeholders


# ============================================================================
# BIBTEX BUILDER
# ============================================================================

class BibTeXBuilder:
    """Build BibTeX from corpus records."""

    def __init__(self, corpus_db_path: str):
        self.corpus_db_path = corpus_db_path
        self.key_map: dict[str, str] = {}  # paper_id -> cite_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/x-bibtex"})

    def build_bibliography(
        self,
        cited_paper_ids: list[str]
    ) -> tuple[list[BibEntry], list[str]]:
        """
        Build refs.bib from corpus.

        Returns: (entries, unverified_keys)
        """
        entries = []
        unverified = []

        for paper_id in cited_paper_ids:
            entry = self._build_entry(paper_id)

            if entry:
                entries.append(entry)
                self.key_map[paper_id] = entry.cite_key

                if entry.status == BibEntryStatus.UNVERIFIED:
                    unverified.append(entry.cite_key)
            else:
                # Missing entry
                missing_entry = BibEntry(
                    cite_key=self._generate_key_from_id(paper_id),
                    entry_type="misc",
                    paper_id=paper_id,
                    status=BibEntryStatus.MISSING,
                    fields={"note": "Entry not found in corpus"}
                )
                entries.append(missing_entry)
                self.key_map[paper_id] = missing_entry.cite_key
                unverified.append(missing_entry.cite_key)

        return entries, unverified

    def _build_entry(self, paper_id: str) -> Optional[BibEntry]:
        """Build a single BibTeX entry."""
        # Load paper record from corpus
        record = self._load_paper_record(paper_id)
        if not record:
            return None

        # Try DOI first
        if record.get("doi"):
            entry = self._fetch_from_crossref(record["doi"], paper_id)
            if entry:
                return entry

        # Try arXiv
        if record.get("arxiv_id"):
            return self._build_from_arxiv(record, paper_id)

        # Records without a DOI or arXiv ID are not citable.
        return None

    def _load_paper_record(self, paper_id: str) -> Optional[dict]:
        """Load paper record from corpus database."""
        import sqlite3

        try:
            conn = sqlite3.connect(self.corpus_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM papers WHERE paper_id = ?
            """, (paper_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return dict(row)
            return None

        except Exception as e:
            print(f"[BibTeX] Error loading record {paper_id}: {e}")
            return None

    def _fetch_from_crossref(
        self,
        doi: str,
        paper_id: str
    ) -> Optional[BibEntry]:
        """Fetch BibTeX from Crossref."""
        try:
            response = self.session.get(
                f"https://doi.org/{doi}",
                timeout=10
            )

            if response.status_code == 200:
                bibtex = response.text

                # Parse BibTeX
                cite_key, entry_type, fields = self._parse_bibtex(bibtex)

                # Ensure key is stable
                if not cite_key:
                    cite_key = self._generate_key_from_doi(doi)

                return BibEntry(
                    cite_key=cite_key,
                    entry_type=entry_type,
                    fields=fields,
                    paper_id=paper_id,
                    doi=doi,
                    status=BibEntryStatus.VERIFIED,
                    source="crossref"
                )

        except Exception as e:
            print(f"[BibTeX] Crossref error for {doi}: {e}")

        return None

    def _build_from_arxiv(self, record: dict, paper_id: str) -> BibEntry:
        """Build entry from arXiv record."""
        arxiv_id = record.get("arxiv_id", "")

        cite_key = self._generate_key(record, paper_id)

        fields = {
            "title": record.get("title", "Unknown"),
            "author": record.get("authors", "Unknown"),
            "year": str(record.get("year", "")),
            "eprint": arxiv_id,
            "archivePrefix": "arXiv"
        }

        if arxiv_id:
            fields["url"] = f"https://arxiv.org/abs/{arxiv_id}"

        return BibEntry(
            cite_key=cite_key,
            entry_type="article",
            fields=fields,
            paper_id=paper_id,
            arxiv_id=arxiv_id,
            status=BibEntryStatus.VERIFIED,
            source="arxiv"
        )

    def _build_from_record(self, record: dict, paper_id: str) -> BibEntry:
        """Build entry from stored record (fallback)."""
        cite_key = self._generate_key(record, paper_id)

        fields = {
            "title": record.get("title", "Unknown"),
            "author": record.get("authors", "Unknown"),
            "year": str(record.get("year", "")),
            "note": "Generated from stored record"
        }

        if record.get("venue"):
            fields["journal"] = record.get("venue")
        if record.get("doi"):
            fields["doi"] = record.get("doi")

        return BibEntry(
            cite_key=cite_key,
            entry_type="misc",
            fields=fields,
            paper_id=paper_id,
            doi=record.get("doi"),
            arxiv_id=record.get("arxiv_id"),
            status=BibEntryStatus.UNVERIFIED,
            source="stored"
        )

    def _generate_key(self, record: dict, paper_id: str) -> str:
        """Generate stable cite key."""
        # Try: firstauthor_year_firstword
        authors = record.get("authors", "")
        if authors:
            first_author = authors.split(",")[0].split(" and ")[0].strip()
            # Normalize
            first_author = re.sub(r'[^a-zA-Z]', '', first_author)
        else:
            first_author = "unknown"

        year = record.get("year", "")
        title = record.get("title", "")

        # First significant word from title
        title_words = re.findall(r'[a-zA-Z]{3,}', title)
        first_word = title_words[0].lower() if title_words else ""

        key = f"{first_author.lower()}{year}{first_word}"

        # Ensure uniqueness
        if key in self.key_map.values():
            key = f"{key}_{paper_id[-4:]}"

        return key

    def _generate_key_from_doi(self, doi: str) -> str:
        """Generate cite key from DOI."""
        # Extract meaningful part
        parts = doi.split("/")
        if len(parts) >= 2:
            return f"ref_{parts[-1].replace('.', '_')[:20]}"
        return f"ref_{hashlib.md5(doi.encode()).hexdigest()[:8]}"

    def _generate_key_from_id(self, paper_id: str) -> str:
        """Generate cite key from paper ID."""
        return f"ref_{paper_id}"

    def _parse_bibtex(self, bibtex: str) -> tuple[str, str, dict]:
        """Parse BibTeX string into components."""
        # Extract entry type and key
        match = re.match(r'@(\w+)\s*\{\s*([^,]+),', bibtex)
        if not match:
            return "", "misc", {}

        entry_type = match.group(1).lower()
        cite_key = match.group(2).strip()

        # Extract fields
        fields = {}
        field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'

        for match in re.finditer(field_pattern, bibtex):
            fields[match.group(1).lower()] = match.group(2).strip()

        return cite_key, entry_type, fields

    def write_bib_file(
        self,
        entries: list[BibEntry],
        output_path: str
    ) -> bool:
        """Write entries to .bib file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("% BibTeX file generated by Stage 9: Export\n")
                f.write(f"% Generated: {datetime.now(timezone.utc).isoformat()}\n")
                f.write(f"% {len(entries)} entries\n\n")

                for entry in entries:
                    f.write(self._format_entry(entry))
                    f.write("\n")

            return True

        except Exception as e:
            print(f"[BibTeX] Error writing .bib file: {e}")
            return False

    def _format_entry(self, entry: BibEntry) -> str:
        """Format a single entry."""
        lines = [f"@{entry.entry_type}{{{entry.cite_key},"]

        for key, value in entry.fields.items():
            lines.append(f"  {key} = {{{value}}},")

        lines.append("}")
        return "\n".join(lines)

    def get_key_map(self) -> dict[str, str]:
        """Get paper_id to cite_key mapping."""
        return self.key_map


# ============================================================================
# LATEX RENDERER
# ============================================================================

class LaTeXRenderer:
    """Convert draft to LaTeX."""

    # Characters that need escaping in LaTeX
    SPECIAL_CHARS = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }

    def __init__(self, key_map: dict[str, str]):
        self.key_map = key_map

    def render(
        self,
        draft,
        outline,
        config: ExportConfig
    ) -> str:
        """Render draft to LaTeX."""
        parts = []

        # Document class
        parts.append(self._render_document_class(config))

        # Preamble
        parts.append(self._render_preamble(config))

        # Begin document
        parts.append("\n\\begin{document}")

        # Title and authors
        parts.append(self._render_front_matter(draft, config))

        # Abstract
        if draft.abstract:
            parts.append(self._render_abstract(draft.abstract))

        # Sections
        for section in draft.sections:
            parts.append(self._render_section(section, outline))

        # Statements
        parts.append(self._render_statements(config))

        # Bibliography
        parts.append(self._render_bibliography(config))

        # End document
        parts.append("\n\\end{document}")

        return "\n".join(parts)

    def _render_document_class(self, config: ExportConfig) -> str:
        """Render document class based on venue."""
        templates = {
            VenueTemplate.IEEE: "\\documentclass[conference]{IEEEtran}",
            VenueTemplate.ACM: "\\documentclass[sigconf]{acmart}",
            VenueTemplate.SPRINGER_LNCS: "\\documentclass[runningheads]{llncs}",
            VenueTemplate.ELSEVIER: "\\documentclass[preprint]{elsarticle}",
            VenueTemplate.ARXIV: "\\documentclass{article}",
            VenueTemplate.GENERIC: "\\documentclass{article}",
        }

        return templates.get(config.venue_template, templates[VenueTemplate.GENERIC])

    def _render_preamble(self, config: ExportConfig) -> str:
        """Render LaTeX preamble."""
        lines = [
            "% Packages",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{amsmath,amssymb}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{hyperref}",
        ]

        # Citation style
        if config.citation_style == "numeric":
            lines.append("\\usepackage[numbers]{natbib}")
        else:
            lines.append("\\usepackage[authoryear]{natbib}")

        # Keywords
        if config.keywords:
            lines.append(f"\\keywords{{{', '.join(config.keywords)}}}")

        return "\n".join(lines)

    def _render_front_matter(
        self,
        draft,
        config: ExportConfig
    ) -> str:
        """Render title, authors, affiliations."""
        parts = []

        # Title
        title = config.title or draft.title
        parts.append(f"\n\\title{{{self._escape(title)}}}")

        # Authors
        if config.authors:
            author_strs = []
            for i, author in enumerate(config.authors):
                aff = i + 1 if config.affiliations and i < len(config.affiliations) else 1
                author_strs.append(f"\\author[{aff}]{{{author}}}")

            parts.append("\n".join(author_strs))

            # Affiliations
            for i, aff in enumerate(config.affiliations, 1):
                parts.append(f"\\affil[{i}]{{{aff}}}")

        parts.append("\\maketitle")

        return "\n".join(parts)

    def _render_abstract(self, abstract: str) -> str:
        """Render abstract."""
        return f"\n\\begin{{abstract}}\n{self._escape(abstract)}\n\\end{{abstract}}"

    def _render_section(self, section, outline) -> str:
        """Render a section."""
        parts = []

        # Section heading
        parts.append(f"\n\\section{{{section.section_name}}}")
        parts.append(f"\\label{{sec:{section.section_id}}}")

        # Paragraphs
        for para in section.paragraphs:
            parts.append(self._render_paragraph(para))

        return "\n".join(parts)

    def _render_paragraph(self, paragraph) -> str:
        """Render a paragraph."""
        text = paragraph.text

        # Replace cite_keys with \cite{}
        text = self._replace_citations(text, paragraph.cite_keys)

        # Escape special characters (but not in math or citations)
        text = self._escape(text)

        # Add paragraph break
        return f"\n{text}\n"

    def _replace_citations(self, text: str, cite_keys: list[str]) -> str:
        """Replace paper_ids with LaTeX citations."""
        for paper_id in cite_keys:
            cite_key = self.key_map.get(paper_id, paper_id)
            # Replace various citation formats
            text = re.sub(
                rf'\({paper_id}\)',
                rf'\\cite{{{cite_key}}}',
                text
            )
            text = re.sub(
                rf'\[{paper_id}\]',
                rf'\\cite{{{cite_key}}}',
                text
            )
            # Also handle author-year style in text
            text = re.sub(
                rf'\b{paper_id}\b',
                rf'\\cite{{{cite_key}}}',
                text
            )

        return text

    def _escape(self, text: str) -> str:
        """Escape special LaTeX characters."""
        # Don't escape math mode
        # Split on $ to preserve math
        parts = text.split('$')

        for i in range(0, len(parts), 2):
            # Even indices are outside math mode
            for char, escaped in self.SPECIAL_CHARS.items():
                parts[i] = parts[i].replace(char, escaped)

        return '$'.join(parts)

    def _render_statements(self, config: ExportConfig) -> str:
        """Render required statements."""
        parts = ["\n% Required Statements"]

        if config.ai_disclosure:
            parts.append(f"\n\\section*{{AI Use Disclosure}}\n{config.ai_disclosure}")

        if config.data_availability:
            parts.append(f"\n\\section*{{Data Availability}}\n{config.data_availability}")

        if config.code_availability:
            parts.append(f"\n\\section*{{Code Availability}}\n{config.code_availability}")

        if config.ethics_statement:
            parts.append(f"\n\\section*{{Ethics Statement}}\n{config.ethics_statement}")

        if config.conflicts_of_interest:
            parts.append(f"\n\\section*{{Conflicts of Interest}}\n{config.conflicts_of_interest}")

        if config.acknowledgements:
            parts.append(f"\n\\section*{{Acknowledgements}}\n{config.acknowledgements}")

        return "\n".join(parts)

    def _render_bibliography(self, config: ExportConfig) -> str:
        """Render bibliography command."""
        return "\n\\bibliographystyle{plain}\n\\bibliography{refs}"


# ============================================================================
# FIGURE AND TABLE INSERTER
# ============================================================================

class FigureTableInserter:
    """Insert figures and tables into LaTeX."""

    def insert_figures(
        self,
        latex_code: str,
        figures: list[FigureExport]
    ) -> str:
        """Insert figure environments."""
        for fig in figures:
            if not fig.file_exists:
                print(f"[Export] Warning: Figure file missing: {fig.file_path}")
                continue

            figure_env = f"""
\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.8\\columnwidth]{{{fig.file_path}}}
\\caption{{{fig.caption}}}
\\label{{{fig.label}}}
\\end{{figure}}
"""
            # Insert after section that references it
            latex_code = latex_code.replace(
                f"% FIGURE:{fig.figure_id}",
                figure_env
            )

        return latex_code

    def insert_tables(
        self,
        latex_code: str,
        tables: list[TableExport]
    ) -> str:
        """Insert table environments."""
        for table in tables:
            table_env = f"""
\\begin{{table}}[t]
\\centering
\\caption{{{table.caption}}}
\\label{{{table.label}}}
{table.latex_code}
\\end{{table}}
"""
            latex_code = latex_code.replace(
                f"% TABLE:{table.table_id}",
                table_env
            )

        return latex_code

    def check_references(
        self,
        latex_code: str,
        figures: list[FigureExport],
        tables: list[TableExport]
    ) -> list[str]:
        """Check that all refs have matching labels."""
        missing = []

        # Find all \ref{...}
        refs = re.findall(r'\\ref\{([^}]+)\}', latex_code)

        # Get all labels
        labels = set(re.findall(r'\\label\{([^}]+)\}', latex_code))

        for ref in refs:
            if ref not in labels:
                missing.append(ref)

        return missing


# ============================================================================
# LATEX COMPILER
# ============================================================================

class LaTeXCompiler:
    """Compile LaTeX to PDF."""

    def __init__(self, max_loops: int = 3):
        self.max_loops = max_loops

    def compile(
        self,
        tex_path: str,
        output_dir: str
    ) -> CompileResult:
        """Run latexmk to compile."""
        result = CompileResult(success=False)

        try:
            # Run latexmk
            proc = subprocess.run(
                [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-output-directory=" + output_dir,
                    tex_path
                ],
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse log
            log_path = tex_path.replace(".tex", ".log")
            if os.path.exists(log_path):
                with open(log_path, "r", errors="ignore") as f:
                    log_content = f.read()

                result.errors, result.warnings = self._parse_log(log_content)
                result.undefined_cites = self._find_undefined_cites(log_content)
                result.undefined_refs = self._find_undefined_refs(log_content)
                result.overfull_boxes = self._find_overfull(log_content)

            # Check for PDF
            pdf_path = tex_path.replace(".tex", ".pdf")
            if os.path.exists(pdf_path):
                result.success = True
                result.pdf_path = pdf_path
                result.page_count = self._count_pages(pdf_path)

            result.log_path = log_path

        except subprocess.TimeoutExpired:
            result.errors.append("Compilation timeout")
        except FileNotFoundError:
            result.errors.append("latexmk not found")
        except Exception as e:
            result.errors.append(f"Compilation error: {e}")

        return result

    def _parse_log(self, log: str) -> tuple[list[str], list[str]]:
        """Parse LaTeX log for errors and warnings."""
        errors = []
        warnings = []

        for line in log.split("\n"):
            if line.startswith("!"):
                errors.append(line)
            elif "Warning:" in line:
                warnings.append(line.strip())

        return errors, warnings

    def _find_undefined_cites(self, log: str) -> list[str]:
        """Find undefined citations."""
        # Pattern for natbib
        pattern = r"Warning--Citation `([^']+)' on page"
        return re.findall(pattern, log)

    def _find_undefined_refs(self, log: str) -> list[str]:
        """Find undefined references."""
        pattern = r"Warning--Reference `([^']+)' on page"
        return re.findall(pattern, log)

    def _find_overfull(self, log: str) -> list[str]:
        """Find overfull boxes."""
        pattern = r"Overfull \\[hv]box .+ in paragraph at lines? (\d+)"
        return re.findall(pattern, log)

    def _count_pages(self, pdf_path: str) -> int:
        """Count pages in PDF."""
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except:
            return 0

    def fix_mechanical_issues(
        self,
        tex_path: str,
        errors: list[str]
    ) -> bool:
        """Fix mechanical issues (escaping, packages)."""
        with open(tex_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for common issues
        fixes_applied = False

        # Add missing packages
        if "\\usepackage{booktabs}" not in content and "\\toprule" in content:
            content = content.replace(
                "\\begin{document}",
                "\\usepackage{booktabs}\n\\begin{document}"
            )
            fixes_applied = True

        # Write back
        if fixes_applied:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(content)

        return fixes_applied


# ============================================================================
# POST-COMPILE CHECKER
# ============================================================================

class PostCompileChecker:
    """Check the compiled PDF."""

    def check(
        self,
        pdf_path: str,
        page_limit: Optional[int],
        compile_result: CompileResult
    ) -> dict:
        """
        Run post-compile checks.

        Returns: dict of check results
        """
        results = {
            "page_count": compile_result.page_count,
            "page_limit_ok": True,
            "pdf_has_placeholders": False,
            "undefined_cites": compile_result.undefined_cites,
            "undefined_refs": compile_result.undefined_refs,
            "overfull_boxes": compile_result.overfull_boxes
        }

        # Page limit
        if page_limit:
            results["page_limit_ok"] = compile_result.page_count <= page_limit

        # Check for placeholders in PDF
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if "[[" in text or "]]" in text or "USER DATA REQUIRED" in text:
                    results["pdf_has_placeholders"] = True
                    break
            doc.close()
        except Exception as e:
            print(f"[Export] Error checking PDF: {e}")

        return results


# ============================================================================
# DOCX CONVERTER
# ============================================================================

class DocxConverter:
    """Convert to DOCX using pandoc."""

    def convert(
        self,
        tex_path: str,
        output_path: str
    ) -> bool:
        """Convert LaTeX to DOCX."""
        try:
            proc = subprocess.run(
                [
                    "pandoc",
                    "--citeproc",
                    tex_path,
                    "-o",
                    output_path
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            return proc.returncode == 0

        except Exception as e:
            print(f"[Export] DOCX conversion failed: {e}")
            return False


# ============================================================================
# REPRODUCIBILITY BUILDER
# ============================================================================

class ReproducibilityBuilder:
    """Build reproducibility package."""

    def build_zip(
        self,
        tex_path: str,
        bib_path: str,
        figures_dir: str,
        search_log: dict,
        corpus_metadata: dict,
        screening_decisions: list,
        config: ExportConfig,
        draft_hash: str,
        output_path: str
    ) -> bool:
        """Create reproducibility zip file."""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # LaTeX source
                zf.write(tex_path, "main.tex")
                zf.write(bib_path, "refs.bib")

                # Figures
                if os.path.exists(figures_dir):
                    for root, dirs, files in os.walk(figures_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, figures_dir)
                            zf.write(file_path, f"figures/{arc_name}")

                # Metadata
                zf.writestr("search_log.json", json.dumps(search_log, indent=2))
                zf.writestr("corpus_metadata.json", json.dumps(corpus_metadata, indent=2))
                zf.writestr("screening_decisions.json", json.dumps(screening_decisions, indent=2))
                zf.writestr("config.json", config.model_dump_json(indent=2))
                zf.writestr("draft_hash.txt", draft_hash)

                # README
                readme = self._generate_readme(draft_hash)
                zf.writestr("README.md", readme)

            return True

        except Exception as e:
            print(f"[Export] Error building zip: {e}")
            return False

    def _generate_readme(self, draft_hash: str) -> str:
        """Generate README for reproducibility package."""
        return f"""# Reproducibility Package

Generated: {datetime.now(timezone.utc).isoformat()}

## Contents

- `main.tex`: LaTeX source
- `refs.bib`: Bibliography
- `figures/`: Figure files
- `search_log.json`: Search queries and dates
- `corpus_metadata.json`: Paper metadata
- `screening_decisions.json`: Screening decisions with reasons
- `config.json`: Export configuration
- `draft_hash.txt`: Hash of the verified draft

## Draft Hash

{draft_hash}

## Compilation

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Note: Source PDFs are not included due to copyright.
"""


# ============================================================================
# SURVEY EXTRAS
# ============================================================================

class SurveyExtras:
    """Generate survey-specific extras."""

    def generate_prisma_flow(
        self,
        search_log: dict
    ) -> dict:
        """Generate PRISMA flow numbers."""
        return {
            "identified": search_log.get("total_identified", 0),
            "duplicates_removed": search_log.get("duplicates", 0),
            "screened": search_log.get("screened", 0),
            "excluded": search_log.get("excluded", 0),
            "included": search_log.get("included", 0),
            "reasons_for_exclusion": search_log.get("exclusion_reasons", {})
        }

    def generate_appendix(
        self,
        search_queries: list[str],
        inclusion_criteria: list[str],
        exclusion_criteria: list[str]
    ) -> str:
        """Generate supplementary appendix."""
        parts = ["\\appendix\n", "\\section{Search Methodology}\n"]

        parts.append("\\subsection{Search Queries}\n")
        parts.append("\\begin{itemize}\n")
        for query in search_queries:
            parts.append(f"\\item {query}\n")
        parts.append("\\end{itemize}\n")

        parts.append("\\subsection{Inclusion Criteria}\n")
        parts.append("\\begin{itemize}\n")
        for criterion in inclusion_criteria:
            parts.append(f"\\item {criterion}\n")
        parts.append("\\end{itemize}\n")

        parts.append("\\subsection{Exclusion Criteria}\n")
        parts.append("\\begin{itemize}\n")
        for criterion in exclusion_criteria:
            parts.append(f"\\item {criterion}\n")
        parts.append("\\end{itemize}\n")

        return "".join(parts)


# ============================================================================
# EXPORT AGENT
# ============================================================================

class ExportAgent:
    """
    Main agent for Stage 9: Export.

    Orchestrates the export pipeline.
    """

    def __init__(
        self,
        corpus_db_path: str = "corpus.db",
        verified_hash: Optional[str] = None
    ):
        self.corpus_db_path = corpus_db_path
        self.verified_hash = verified_hash

        # Initialize components
        self.preflight = PreflightChecker(verified_hash)
        self.bibtex_builder = BibTeXBuilder(corpus_db_path)
        self.latex_compiler = LaTeXCompiler()
        self.post_checker = PostCompileChecker()
        self.docx_converter = DocxConverter()
        self.repro_builder = ReproducibilityBuilder()
        self.survey_extras = SurveyExtras()

    def run(
        self,
        draft,
        verification_report,
        revision_report,
        outline,
        gap_report,
        scope_doc,
        config: ExportConfig,
        output_dir: str = "export"
    ) -> ExportReport:
        """
        Execute the full export pipeline.

        Steps 1-15 from the specification.
        """
        print(f"[Stage 9] Starting export")

        report = ExportReport()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Preflight gate
        draft_hash = self._compute_draft_hash(draft)
        passed, issues = self.preflight.check(
            draft, verification_report, revision_report, draft_hash
        )

        report.preflight_passed = passed
        report.preflight_issues = issues

        if not passed:
            print(f"[Stage 9] Preflight failed: {issues}")
            report.status = ExportStatus.BLOCKED
            return report

        print(f"[Stage 9] Preflight passed")

        # Step 2: Load venue config (already in config parameter)

        # Step 3: Build refs.bib
        print(f"[Stage 9] Building bibliography...")
        entries, unverified = self.bibtex_builder.build_bibliography(
            draft.total_cite_keys
        )
        report.bib_entries = entries
        report.unverified_refs = unverified
        if unverified:
            report.preflight_passed = False
            report.preflight_issues.append(
                "Unverified bibliography entries: " + ", ".join(unverified)
            )
            report.status = ExportStatus.BLOCKED
            return report

        bib_path = os.path.join(output_dir, "refs.bib")
        self.bibtex_builder.write_bib_file(entries, bib_path)
        report.bib_path = bib_path

        report.search_queries = self._write_reproducibility_log(
            output_dir, scope_doc, cited_paper_ids=draft.total_cite_keys
        )

        # Step 4: Convert to LaTeX
        print(f"[Stage 9] Rendering LaTeX...")
        key_map = self.bibtex_builder.get_key_map()
        renderer = LaTeXRenderer(key_map)
        latex_code = renderer.render(draft, outline, config)

        tex_path = os.path.join(output_dir, "main.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
        report.tex_path = tex_path

        # Step 5: Insert figures and tables
        print(f"[Stage 9] Processing figures and tables...")
        # (Would extract from outline.figures)

        # Step 6: Front matter already rendered

        # Step 7: Statements already rendered

        # Step 8: Compile
        print(f"[Stage 9] Compiling PDF...")
        compile_result = self.latex_compiler.compile(tex_path, output_dir)
        report.compile_result = compile_result

        # Retry with fixes if needed
        loops = 0
        while not compile_result.success and loops < 3:
            print(f"[Stage 9] Compilation failed, attempting fixes (loop {loops + 1})...")
            if self.latex_compiler.fix_mechanical_issues(tex_path, compile_result.errors):
                compile_result = self.latex_compiler.compile(tex_path, output_dir)
            loops += 1

        if compile_result.success:
            report.status = ExportStatus.COMPILED
            report.pdf_path = compile_result.pdf_path
            print(f"[Stage 9] Compilation successful: {compile_result.page_count} pages")
        else:
            report.status = ExportStatus.FAILED
            print(f"[Stage 9] Compilation failed: {compile_result.errors}")
            return report

        # Step 9: Post-compile checks
        print(f"[Stage 9] Running post-compile checks...")
        check_results = self.post_checker.check(
            compile_result.pdf_path,
            config.page_limit,
            compile_result
        )

        report.page_count = check_results["page_count"]
        report.page_limit_ok = check_results["page_limit_ok"]
        report.pdf_has_placeholders = check_results["pdf_has_placeholders"]
        report.missing_refs = check_results["undefined_refs"]

        # Step 10: Re-validate references
        # (Would run Stage 7 DOI check)

        # Step 11: Optional DOCX
        if config.generate_docx:
            print(f"[Stage 9] Generating DOCX...")
            docx_path = os.path.join(output_dir, "main.docx")
            if self.docx_converter.convert(tex_path, docx_path):
                report.docx_path = docx_path

        # Step 12: Survey extras
        if scope_doc.contribution_type.value == "survey":
            print(f"[Stage 9] Generating survey extras...")
            # (Would extract from search log)
            report.prisma_flow = self.survey_extras.generate_prisma_flow({})

        # Step 13: Reproducibility package
        print(f"[Stage 9] Building reproducibility package...")
        zip_path = os.path.join(output_dir, "reproducibility.zip")
        if self.repro_builder.build_zip(
            tex_path, bib_path, "figures", {}, {}, [], config, draft_hash, zip_path
        ):
            report.reproducibility_zip = zip_path

        # Step 14: Human final check
        self._human_final_check(report)

        # Step 15: Freeze
        report.ready_to_submit = all([
            report.preflight_passed,
            report.status == ExportStatus.COMPILED,
            report.page_limit_ok,
            not report.pdf_has_placeholders,
            not report.missing_refs
        ])

        print(f"\n[Stage 9] Export complete")
        print(f"[Stage 9] Ready to submit: {report.ready_to_submit}")

        return report

    def _write_reproducibility_log(
        self,
        output_dir: str,
        scope_doc,
        cited_paper_ids: list[str],
    ) -> list[str]:
        """Persist search provenance and screening criteria beside the export."""
        conn = sqlite3.connect(self.corpus_db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT query_id, query_text, source, timestamp, params, "
                "results_count, status, error FROM search_log ORDER BY timestamp"
            ).fetchall()
        finally:
            conn.close()

        queries = [dict(row) for row in rows]
        path = os.path.join(output_dir, "reproducibility.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "search_queries": queries,
                    "year_range": list(scope_doc.year_range),
                    "inclusion_criteria": scope_doc.inclusion_criteria,
                    "exclusion_criteria": scope_doc.exclusion_criteria,
                    "cited_paper_ids": cited_paper_ids,
                },
                handle,
                indent=2,
            )
        return [row["query_text"] for row in queries]

    def _compute_draft_hash(self, draft) -> str:
        """Compute hash of draft content."""
        content = draft.model_dump_json()
        return hashlib.sha256(content.encode()).hexdigest()

    def _human_final_check(self, report: ExportReport):
        """Step 14: Human final check."""
        print(f"\n{'='*70}")
        print("EXPORT SUMMARY - HUMAN FINAL CHECK")
        print(f"{'='*70}")

        print(f"\nFiles generated:")
        print(f"  LaTeX: {report.tex_path}")
        print(f"  BibTeX: {report.bib_path}")
        print(f"  PDF: {report.pdf_path}")

        if report.docx_path:
            print(f"  DOCX: {report.docx_path}")

        print(f"\nChecks:")
        print(f"  Pages: {report.page_count}")
        print(f"  Page limit: {'OK' if report.page_limit_ok else 'EXCEEDED'}")
        print(f"  Placeholders: {'FOUND' if report.pdf_has_placeholders else 'None'}")
        print(f"  Unverified refs: {len(report.unverified_refs)}")

        if report.missing_refs:
            print(f"\nMissing references: {report.missing_refs}")

        print(f"\n{'-'*70}")
        print("\nPlease check:")
        print("  1. Read the compiled PDF against the template")
        print("  2. Run plagiarism check (iThenticate)")
        print("  3. Verify the venue's submission rules")
        print("  4. Check the AI-use disclosure statement")

        print(f"\nReady to submit: {report.ready_to_submit}")

        input("\nPress Enter to acknowledge...")


# ============================================================================
# DEMO
# ============================================================================

def demo_export():
    """Demo function showing export."""
    from drafting import Draft, SectionDraft, ParagraphDraft
    from verification import VerificationReport
    from revision import RevisionReport
    from outline import Outline
    from gap_analysis import GapReport
    from scope import ScopeDocument, ContributionType

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
                        text="Large language models have transformed education in numerous ways (paper_1).",
                        cite_keys=["paper_1"],
                        word_count=15
                    )
                ],
                word_count=15
            )
        ],
        total_word_count=15,
        total_cite_keys=["paper_1"]
    )

    # Create sample verification report
    verification_report = VerificationReport(passed=True)

    # Create sample revision report
    revision_report = RevisionReport()

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

    # Create export config
    config = ExportConfig(
        venue_template=VenueTemplate.GENERIC,
        authors=["John Doe"],
        affiliations=["University of Example"],
        ai_disclosure="AI assistance was used for literature search and drafting. The authors reviewed and edited all content."
    )

    agent = ExportAgent(corpus_db_path="demo_corpus.db")

    report = agent.run(
        draft=draft,
        verification_report=verification_report,
        revision_report=revision_report,
        outline=outline,
        gap_report=gap_report,
        scope_doc=scope_doc,
        config=config,
        output_dir="demo_export"
    )

    print(f"\nExport status: {report.status.value}")
    print(f"Ready to submit: {report.ready_to_submit}")

    return agent


if __name__ == "__main__":
    demo_export()
