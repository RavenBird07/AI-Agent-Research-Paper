"""
Stage 3: Note Extraction

Converts each included paper into structured, evidence-backed notes.
Stages 4-8 work only from these notes and stored text chunks.
"""

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import requests
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


# ============================================================================
# SCHEMAS
# ============================================================================

class Evidence(BaseModel):
    """A piece of evidence with a verbatim quote."""
    quote: str = Field(..., description="Verbatim quote from the paper (≤30 words)")
    section: str = Field(..., description="Section where the quote appears")
    page: Optional[int] = Field(default=None, description="Page number")


class PaperNotes(BaseModel):
    """Structured notes extracted from a paper."""
    # Identifiers
    paper_id: str
    title: str
    extraction_source: str = Field(default="full_text", description="full_text, abstract, or ocr")
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Core content
    main_contributions: list[str] = Field(default_factory=list, description="Key contributions claimed")
    contributions_evidence: list[Evidence] = Field(default_factory=list)

    # Methodology
    method_category: Optional[str] = Field(default=None, description="Type of method (e.g., 'fine-tuning', 'RAG')")
    method_description: str = Field(default="", description="Summary of the method")
    method_evidence: Optional[Evidence] = None

    # Datasets
    datasets: list[str] = Field(default_factory=list, description="Datasets used")
    datasets_evidence: list[Evidence] = Field(default_factory=list)

    # Results
    main_results: list[str] = Field(default_factory=list, description="Key numerical results with units")
    results_evidence: list[Evidence] = Field(default_factory=list)
    baselines_compared: list[str] = Field(default_factory=list, description="Baseline methods compared against")

    # Limitations and Future Work
    limitations: list[str] = Field(default_factory=list, description="Limitations stated by authors")
    limitations_evidence: list[Evidence] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list, description="Future work suggested")

    # Metadata
    confidence: str = Field(default="medium", description="high, medium, low")
    needs_review: bool = Field(default=False, description="Flagged for human review")
    review_reason: Optional[str] = Field(default=None)
    quote_verification_failures: int = Field(default=0)


class TextChunk(BaseModel):
    """A chunk of text from a paper for vector indexing."""
    chunk_id: str
    paper_id: str
    section: str
    page: Optional[int] = None
    text: str
    token_count: int
    embedding: Optional[list[float]] = None


# ============================================================================
# PDF PARSING
# ============================================================================

class PDFParser:
    """Parse PDFs to structured text with section detection."""

    def __init__(self, use_grobid: bool = False, grobid_url: str = "http://localhost:8070"):
        self.use_grobid = use_grobid
        self.grobid_url = grobid_url

    def parse(self, pdf_path: str) -> dict:
        """
        Parse a PDF and return structured text.

        Returns:
            dict with 'sections', 'full_text', 'references', 'tables'
        """
        if self.use_grobid:
            result = self._parse_with_grobid(pdf_path)
            if result:
                return result

        # Fallback to PyMuPDF
        return self._parse_with_pymupdf(pdf_path)

    def _parse_with_grobid(self, pdf_path: str) -> Optional[dict]:
        """Parse PDF using GROBID service (returns TEI XML)."""
        try:
            with open(pdf_path, "rb") as f:
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files={"input": f},
                    timeout=120
                )
                response.raise_for_status()
                tei_xml = response.text

            # Parse TEI XML
            return self._parse_tei_xml(tei_xml)

        except Exception as e:
            print(f"[GROBID] Parsing failed: {e}")
            return None

    def _parse_tei_xml(self, tei_xml: str) -> dict:
        """Parse TEI XML from GROBID."""
        import xml.etree.ElementTree as ET

        sections = {}
        full_text = []
        references = []

        try:
            root = ET.fromstring(tei_xml)

            # TEI namespace
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}

            # Extract title
            title_elem = root.find(".//tei:titleStmt/tei:title", ns)
            title = title_elem.text if title_elem is not None else ""

            # Extract abstract
            abstract_elem = root.find(".//tei:abstract", ns)
            if abstract_elem is not None:
                abstract_text = "".join(abstract_elem.itertext()).strip()
                sections["abstract"] = abstract_text
                full_text.append(f"ABSTRACT\n{abstract_text}")

            # Extract body sections
            body = root.find(".//tei:body", ns)
            if body is not None:
                for div in body.findall(".//tei:div", ns):
                    head = div.find("tei:head", ns)
                    section_name = head.text if head is not None else "unknown"
                    section_name = self._normalize_section_name(section_name)

                    text = "".join(div.itertext()).strip()
                    sections[section_name] = text
                    full_text.append(f"{section_name.upper()}\n{text}")

            # Extract references
            list_bibl = root.find(".//tei:listBibl", ns)
            if list_bibl is not None:
                for bibl in list_bibl.findall("tei:biblStruct", ns):
                    ref_text = "".join(bibl.itertext()).strip()
                    if ref_text:
                        references.append(ref_text)

        except Exception as e:
            print(f"[GROBID] XML parsing error: {e}")

        return {
            "title": title,
            "sections": sections,
            "full_text": "\n\n".join(full_text),
            "references": references,
            "tables": []
        }

    def _parse_with_pymupdf(self, pdf_path: str) -> dict:
        """Parse PDF using PyMuPDF (fitz)."""
        sections = {}
        full_text = []
        references = []
        tables = []

        try:
            doc = fitz.open(pdf_path)

            current_section = "unknown"
            section_text = []

            for page_num, page in enumerate(doc):
                text = page.get_text()

                # Check if this is a scanned PDF
                if len(text.strip()) < 100:
                    print(f"[PDF] Page {page_num + 1} has very little text, may need OCR")
                    continue

                # Clean text
                text = self._clean_text(text)

                # Detect section headings
                lines = text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if this is a heading
                    if self._is_heading(line, doc, page_num):
                        # Save previous section
                        if section_text:
                            sections[current_section] = "\n".join(section_text)

                        # Start new section
                        current_section = self._normalize_section_name(line)
                        section_text = []
                    else:
                        section_text.append(line)

                full_text.append(text)

            # Save last section
            if section_text:
                sections[current_section] = "\n".join(section_text)

            # Extract references section
            ref_text = sections.get("references", "")
            if ref_text:
                references = self._parse_references(ref_text)

            doc.close()

        except Exception as e:
            print(f"[PyMuPDF] Parsing error: {e}")

        return {
            "sections": sections,
            "full_text": "\n".join(full_text),
            "references": references,
            "tables": tables
        }

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove page numbers (standalone numbers)
        text = re.sub(r"\n\d+\s*\n", "\n", text)

        # Remove headers/footers (repeated patterns)
        lines = text.split("\n")
        if len(lines) > 10:
            # Check for repeated patterns at start/end
            first_lines = lines[:3]
            last_lines = lines[-3:]

        # Fix hyphenated line breaks
        text = re.sub(r"-\s*\n\s*", "-", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def _is_heading(self, line: str, doc, page_num: int) -> bool:
        """Detect if a line is a section heading."""
        # Common section names
        heading_patterns = [
            r"^(abstract|introduction|related\s*work|background|method|methodology|"
            r"approach|system|architecture|experiments?|evaluation|results?|"
            r"discussion|limitations|conclusion|future\s*work|references|appendix)",
            r"^\d+\.?\s+[A-Z]"  # Numbered sections like "1. Introduction"
        ]

        line_lower = line.lower().strip()

        for pattern in heading_patterns:
            if re.match(pattern, line_lower):
                return True

        # Check if line is short and mostly uppercase (common for headings)
        if len(line) < 60 and line.isupper():
            return True

        return False

    def _normalize_section_name(self, heading: str) -> str:
        """Map heading to canonical section label."""
        heading_lower = heading.lower().strip()

        # Remove numbering
        heading_lower = re.sub(r"^\d+\.?\s*", "", heading_lower)

        # Map to canonical names
        mapping = {
            "abstract": "abstract",
            "introduction": "introduction",
            "related work": "related_work",
            "background": "related_work",
            "literature review": "related_work",
            "method": "method",
            "methodology": "method",
            "approach": "method",
            "system": "method",
            "architecture": "method",
            "experiments": "experiments",
            "evaluation": "experiments",
            "results": "results",
            "discussion": "discussion",
            "limitations": "limitations",
            "conclusion": "conclusion",
            "conclusions": "conclusion",
            "future work": "conclusion",
            "references": "references",
            "bibliography": "references"
        }

        for key, canonical in mapping.items():
            if key in heading_lower:
                return canonical

        return "unknown"

    def _parse_references(self, ref_text: str) -> list[str]:
        """Parse references section into individual references."""
        # Simple splitting by common patterns
        refs = re.split(r"\[\d+\]|\n\d+\.", ref_text)
        return [r.strip() for r in refs if r.strip()]


# ============================================================================
# TEXT CHUNKING
# ============================================================================

class TextChunker:
    """Split text into chunks for vector indexing."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_sections(
        self,
        paper_id: str,
        sections: dict,
        full_text: str
    ) -> list[TextChunk]:
        """
        Chunk paper sections into token-sized pieces.

        Args:
            paper_id: Paper identifier
            sections: Dict of section_name -> section_text
            full_text: Full paper text (fallback)

        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_num = 0

        for section_name, section_text in sections.items():
            if section_name == "references":
                continue  # Skip references for chunking

            # Skip empty sections
            if not section_text or len(section_text.strip()) < 50:
                continue

            # Split into chunks
            section_chunks = self._split_text(section_text)

            for chunk_text in section_chunks:
                chunk_id = f"{paper_id}_chunk_{chunk_num}"
                token_count = self._estimate_tokens(chunk_text)

                chunk = TextChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    section=section_name,
                    text=chunk_text,
                    token_count=token_count
                )
                chunks.append(chunk)
                chunk_num += 1

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()

        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            # Move start with overlap
            start = end - self.overlap

            if start <= 0:
                start = end

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4


# ============================================================================
# QUOTE VERIFICATION
# ============================================================================

class QuoteVerifier:
    """Verify that quotes actually appear in the paper text."""

    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = similarity_threshold

    def verify_quote(
        self,
        quote: str,
        full_text: str,
        section_text: Optional[str] = None
    ) -> tuple[bool, float]:
        """
        Verify a quote against paper text.

        Returns:
            Tuple of (is_valid, similarity_score)
        """
        # Normalize quote
        quote_normalized = self._normalize_text(quote)

        # Check exact match first
        if quote_normalized in self._normalize_text(full_text):
            return True, 1.0

        # Check section text if provided
        if section_text:
            if quote_normalized in self._normalize_text(section_text):
                return True, 1.0

        # Fuzzy match
        best_similarity = 0.0

        # Split text into windows similar to quote length
        words = full_text.split()
        quote_words = quote_normalized.split()
        window_size = len(quote_words) + 10

        for i in range(0, len(words) - window_size, window_size // 2):
            window = " ".join(words[i:i + window_size])
            window_normalized = self._normalize_text(window)

            similarity = fuzz.ratio(quote_normalized, window_normalized) / 100.0
            best_similarity = max(best_similarity, similarity)

            if similarity >= self.similarity_threshold:
                return True, similarity

        return False, best_similarity

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text


# ============================================================================
# EXTRACTION DATABASE
# ============================================================================

class ExtractionDatabase:
    """SQLite database for storing paper notes and chunks."""

    def __init__(self, db_path: str = "extraction.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Paper notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_notes (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                extraction_source TEXT DEFAULT 'full_text',
                extraction_timestamp TEXT,
                main_contributions TEXT,
                contributions_evidence TEXT,
                method_category TEXT,
                method_description TEXT,
                method_evidence TEXT,
                datasets TEXT,
                datasets_evidence TEXT,
                main_results TEXT,
                results_evidence TEXT,
                baselines_compared TEXT,
                limitations TEXT,
                limitations_evidence TEXT,
                future_work TEXT,
                confidence TEXT DEFAULT 'medium',
                needs_review INTEGER DEFAULT 0,
                review_reason TEXT,
                quote_verification_failures INTEGER DEFAULT 0
            )
        """)

        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                section TEXT,
                page INTEGER,
                text TEXT NOT NULL,
                token_count INTEGER,
                embedding BLOB,
                FOREIGN KEY (paper_id) REFERENCES paper_notes(paper_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section)")

        self.conn.commit()

    def insert_notes(self, notes: PaperNotes) -> bool:
        """Insert paper notes."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO paper_notes (
                    paper_id, title, extraction_source, extraction_timestamp,
                    main_contributions, contributions_evidence,
                    method_category, method_description, method_evidence,
                    datasets, datasets_evidence,
                    main_results, results_evidence, baselines_compared,
                    limitations, limitations_evidence, future_work,
                    confidence, needs_review, review_reason, quote_verification_failures
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                notes.paper_id,
                notes.title,
                notes.extraction_source,
                notes.extraction_timestamp,
                json.dumps(notes.main_contributions),
                json.dumps([e.model_dump() for e in notes.contributions_evidence]),
                notes.method_category,
                notes.method_description,
                json.dumps(notes.method_evidence.model_dump()) if notes.method_evidence else None,
                json.dumps(notes.datasets),
                json.dumps([e.model_dump() for e in notes.datasets_evidence]),
                json.dumps(notes.main_results),
                json.dumps([e.model_dump() for e in notes.results_evidence]),
                json.dumps(notes.baselines_compared),
                json.dumps(notes.limitations),
                json.dumps([e.model_dump() for e in notes.limitations_evidence]),
                json.dumps(notes.future_work),
                notes.confidence,
                int(notes.needs_review),
                notes.review_reason,
                notes.quote_verification_failures
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[Database] Insert notes error: {e}")
            return False

    def insert_chunk(self, chunk: TextChunk) -> bool:
        """Insert a text chunk."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, paper_id, section, page, text, token_count, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.paper_id,
                chunk.section,
                chunk.page,
                chunk.text,
                chunk.token_count,
                json.dumps(chunk.embedding) if chunk.embedding else None
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[Database] Insert chunk error: {e}")
            return False

    def get_notes(self, paper_id: str) -> Optional[PaperNotes]:
        """Get notes for a paper."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM paper_notes WHERE paper_id = ?", (paper_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_notes(row)
        return None

    def get_all_notes(self) -> list[PaperNotes]:
        """Get all paper notes."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM paper_notes")
        return [self._row_to_notes(row) for row in cursor.fetchall()]

    def get_chunks(self, paper_id: str) -> list[TextChunk]:
        """Get all chunks for a paper."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE paper_id = ?", (paper_id,))
        return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def get_papers_for_review(self) -> list[PaperNotes]:
        """Get all papers flagged for review."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM paper_notes WHERE needs_review = 1")
        return [self._row_to_notes(row) for row in cursor.fetchall()]

    def _row_to_notes(self, row: sqlite3.Row) -> PaperNotes:
        """Convert database row to PaperNotes."""
        return PaperNotes(
            paper_id=row["paper_id"],
            title=row["title"],
            extraction_source=row["extraction_source"],
            extraction_timestamp=row["extraction_timestamp"],
            main_contributions=json.loads(row["main_contributions"]) if row["main_contributions"] else [],
            contributions_evidence=[Evidence(**e) for e in json.loads(row["contributions_evidence"])] if row["contributions_evidence"] else [],
            method_category=row["method_category"],
            method_description=row["method_description"] or "",
            method_evidence=Evidence(**json.loads(row["method_evidence"])) if row["method_evidence"] else None,
            datasets=json.loads(row["datasets"]) if row["datasets"] else [],
            datasets_evidence=[Evidence(**e) for e in json.loads(row["datasets_evidence"])] if row["datasets_evidence"] else [],
            main_results=json.loads(row["main_results"]) if row["main_results"] else [],
            results_evidence=[Evidence(**e) for e in json.loads(row["results_evidence"])] if row["results_evidence"] else [],
            baselines_compared=json.loads(row["baselines_compared"]) if row["baselines_compared"] else [],
            limitations=json.loads(row["limitations"]) if row["limitations"] else [],
            limitations_evidence=[Evidence(**e) for e in json.loads(row["limitations_evidence"])] if row["limitations_evidence"] else [],
            future_work=json.loads(row["future_work"]) if row["future_work"] else [],
            confidence=row["confidence"] or "medium",
            needs_review=bool(row["needs_review"]),
            review_reason=row["review_reason"],
            quote_verification_failures=row["quote_verification_failures"] or 0
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> TextChunk:
        """Convert database row to TextChunk."""
        return TextChunk(
            chunk_id=row["chunk_id"],
            paper_id=row["paper_id"],
            section=row["section"],
            page=row["page"],
            text=row["text"],
            token_count=row["token_count"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None
        )

    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# VECTOR INDEX
# ============================================================================

class VectorIndex:
    """Simple vector index for chunk retrieval."""

    def __init__(self, db: ExtractionDatabase):
        self.db = db
        self.embeddings = {}  # chunk_id -> embedding

    def add_chunks(self, chunks: list[TextChunk], embedding_model=None):
        """Add chunks to the index with embeddings."""
        for chunk in chunks:
            if embedding_model and not chunk.embedding:
                # Generate embedding (placeholder - use actual model in production)
                # chunk.embedding = embedding_model.embed(chunk.text)
                chunk.embedding = self._dummy_embedding(chunk.text)

            self.db.insert_chunk(chunk)
            if chunk.embedding:
                self.embeddings[chunk.chunk_id] = chunk.embedding

    def search(self, query: str, top_k: int = 10, embedding_model=None) -> list[TextChunk]:
        """Search for similar chunks."""
        # Get all chunks from database
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT * FROM chunks")
        all_chunks = [self.db._row_to_chunk(row) for row in cursor.fetchall()]

        if not all_chunks:
            return []

        # Generate query embedding
        if embedding_model:
            query_embedding = embedding_model.embed(query)
        else:
            query_embedding = self._dummy_embedding(query)

        # Compute similarities
        similarities = []
        for chunk in all_chunks:
            if chunk.embedding:
                sim = self._cosine_similarity(query_embedding, chunk.embedding)
                similarities.append((chunk, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in similarities[:top_k]]

    def _dummy_embedding(self, text: str) -> list[float]:
        """Generate a dummy embedding based on text hash."""
        # Placeholder - use actual embedding model in production
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        return [float(int(h[i:i+2], 16)) / 255.0 for i in range(0, 64, 4)]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


# ============================================================================
# NOTE EXTRACTION AGENT
# ============================================================================

class NoteExtractionAgent:
    """
    Main agent for Stage 3: Note Extraction.

    Converts papers to structured notes with evidence.
    """

    def __init__(
        self,
        llm,  # LangChain LLM instance
        db_path: str = "extraction.db",
        use_grobid: bool = False,
        grobid_url: str = "http://localhost:8070"
    ):
        self.llm = llm
        self.db = ExtractionDatabase(db_path)
        self.parser = PDFParser(use_grobid=use_grobid, grobid_url=grobid_url)
        self.chunker = TextChunker()
        self.verifier = QuoteVerifier()
        self.vector_index = VectorIndex(self.db)

    def extract_from_paper(
        self,
        paper_record,  # PaperRecord from Stage 2
        temperature: float = 0.0
    ) -> PaperNotes:
        """
        Extract notes from a single paper.

        Steps 1-11 for a single paper.
        """
        paper_id = paper_record.paper_id
        print(f"[Stage 3] Extracting notes for: {paper_id}")

        # Step 1: Load paper
        if paper_record.fulltext_status == "pdf" and paper_record.pdf_path:
            parsed = self.parser.parse(paper_record.pdf_path)
            full_text = parsed["full_text"]
            sections = parsed["sections"]
            extraction_source = "full_text"
        else:
            # Abstract only
            full_text = paper_record.abstract or ""
            sections = {"abstract": full_text}
            extraction_source = "abstract"

        # Check for garbled text
        if extraction_source == "full_text" and self._is_garbled(full_text):
            print(f"[Stage 3] Warning: Garbled text detected, confidence will be low")
            # Could trigger OCR here

        # Step 4-5: Chunk and index
        chunks = self.chunker.chunk_sections(paper_id, sections, full_text)
        self.vector_index.add_chunks(chunks)

        # Step 6-7: Extract notes
        notes = self._extract_with_llm(
            paper_id=paper_id,
            title=paper_record.title,
            sections=sections,
            full_text=full_text,
            extraction_source=extraction_source,
            temperature=temperature
        )

        # Step 8: Verify quotes
        notes = self._verify_all_quotes(notes, full_text, sections)

        # Step 10: Score confidence
        notes.confidence = self._score_confidence(notes, extraction_source)

        return notes

    def _is_garbled(self, text: str) -> bool:
        """Check if text appears garbled (low dictionary word ratio)."""
        words = text.split()
        if len(words) < 50:
            return True

        # Check ratio of alphabetic words
        alpha_words = [w for w in words if w.isalpha()]
        ratio = len(alpha_words) / len(words) if words else 0

        return ratio < 0.7

    def _extract_with_llm(
        self,
        paper_id: str,
        title: str,
        sections: dict,
        full_text: str,
        extraction_source: str,
        temperature: float = 0.0
    ) -> PaperNotes:
        """
        Extract structured notes using LLM.

        Step 6-7: One LLM call per paper with structured output.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        parser = PydanticOutputParser(pydantic_object=PaperNotes)

        # Build context from key sections
        context_parts = []
        priority_sections = ["abstract", "introduction", "method", "results", "limitations", "conclusion"]

        for section in priority_sections:
            if section in sections and sections[section]:
                context_parts.append(f"## {section.upper()}\n{sections[section][:2000]}")

        # Add other sections
        for section, text in sections.items():
            if section not in priority_sections and text:
                context_parts.append(f"## {section.upper()}\n{text[:1000]}")

        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant extracting structured notes from academic papers.

RULES:
1. Extract ONLY what the paper states. Do not infer or fill gaps.
2. Every item MUST have a verbatim quote (≤30 words) from the paper.
3. Copy numbers exactly with units and dataset names.
4. Use null or empty list when the paper doesn't mention something.
5. Record limitations ONLY if the authors explicitly state them.

{format_instructions}"""),
            ("human", """Extract notes from this paper:

Title: {title}

{context}

Extract structured notes with evidence.""")
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm | parser

        try:
            notes = chain.invoke({
                "title": title,
                "context": context
            })

            # Set paper_id and metadata
            notes.paper_id = paper_id
            notes.title = title
            notes.extraction_source = extraction_source

            return notes

        except Exception as e:
            print(f"[Stage 3] LLM extraction error: {e}")
            if getattr(e, "status_code", None) == 429 or "Rate limit exceeded" in str(e):
                raise RuntimeError(
                    "OpenRouter rate limit reached during Stage 3 extraction. "
                    "Wait for the quota reset, add credits, or set RESEARCH_MODEL "
                    "to a model available on your account."
                ) from e
            # Return empty notes
            return PaperNotes(
                paper_id=paper_id,
                title=title,
                extraction_source=extraction_source,
                needs_review=True,
                review_reason=f"LLM extraction failed: {e}"
            )

    def _verify_all_quotes(
        self,
        notes: PaperNotes,
        full_text: str,
        sections: dict
    ) -> PaperNotes:
        """
        Step 8: Verify all quotes in notes.
        """
        failures = 0

        # Verify contributions evidence
        verified_contributions = []
        for evidence in notes.contributions_evidence:
            is_valid, _ = self.verifier.verify_quote(
                evidence.quote,
                full_text,
                sections.get(evidence.section)
            )
            if is_valid:
                verified_contributions.append(evidence)
            else:
                failures += 1

        notes.contributions_evidence = verified_contributions

        # Verify results evidence
        verified_results = []
        for evidence in notes.results_evidence:
            is_valid, _ = self.verifier.verify_quote(
                evidence.quote,
                full_text,
                sections.get(evidence.section)
            )
            if is_valid:
                verified_results.append(evidence)
            else:
                failures += 1

        notes.results_evidence = verified_results

        # Verify limitations evidence
        verified_limitations = []
        for evidence in notes.limitations_evidence:
            is_valid, _ = self.verifier.verify_quote(
                evidence.quote,
                full_text,
                sections.get(evidence.section)
            )
            if is_valid:
                verified_limitations.append(evidence)
            else:
                failures += 1

        notes.limitations_evidence = verified_limitations

        # Track failures
        notes.quote_verification_failures = failures

        if failures > 3:
            notes.needs_review = True
            notes.review_reason = f"Too many quote verification failures: {failures}"

        return notes

    def _score_confidence(
        self,
        notes: PaperNotes,
        extraction_source: str
    ) -> str:
        """
        Step 10: Score confidence level.
        """
        if extraction_source == "abstract":
            return "low"

        # Check for missing key fields
        missing_fields = 0
        if not notes.main_contributions:
            missing_fields += 1
        if not notes.method_description:
            missing_fields += 1
        if not notes.main_results:
            missing_fields += 1

        if notes.quote_verification_failures == 0 and missing_fields == 0:
            return "high"
        elif missing_fields <= 1:
            return "medium"
        else:
            return "low"

    def self_consistency_check(
        self,
        paper_record,
        temperature_1: float = 0.0,
        temperature_2: float = 0.3
    ) -> tuple[PaperNotes, bool]:
        """
        Step 9: Run extraction twice and compare key fields.
        """
        notes_1 = self.extract_from_paper(paper_record, temperature=temperature_1)
        notes_2 = self.extract_from_paper(paper_record, temperature=temperature_2)

        # Compare key fields
        disagrees = False

        if notes_1.method_category != notes_2.method_category:
            disagrees = True
            print(f"[Stage 3] Self-consistency: method_category disagrees")

        # Compare main results (check if same numbers appear)
        results_1 = set(re.findall(r"\d+\.?\d*", " ".join(notes_1.main_results)))
        results_2 = set(re.findall(r"\d+\.?\d*", " ".join(notes_2.main_results)))

        if results_1 != results_2:
            disagrees = True
            print(f"[Stage 3] Self-consistency: results disagree")

        if disagrees:
            notes_1.needs_review = True
            notes_1.review_reason = "Self-consistency check failed"

        return notes_1, disagrees

    def spot_check(self, num_papers: int = 5):
        """
        Step 11: Human spot-check of random papers.
        """
        import random

        all_notes = self.db.get_all_notes()
        papers_for_review = self.db.get_papers_for_review()

        # Random papers
        random_papers = random.sample(
            all_notes,
            min(num_papers, len(all_notes))
        )

        # Combine with flagged papers while preserving PaperNotes objects.
        # Pydantic models are not hashable, so deduplicate by stable paper ID.
        to_review = []
        seen_paper_ids = set()
        for notes in random_papers + papers_for_review:
            if notes.paper_id not in seen_paper_ids:
                seen_paper_ids.add(notes.paper_id)
                to_review.append(notes)

        print(f"\n[Stage 3] Spot-checking {len(to_review)} papers...")

        for notes in to_review:
            print(f"\n{'='*70}")
            print(f"Paper: {notes.title}")
            print(f"Confidence: {notes.confidence}")
            print(f"Needs Review: {notes.needs_review}")
            if notes.review_reason:
                print(f"Reason: {notes.review_reason}")

            print(f"\nMain Contributions:")
            for i, contrib in enumerate(notes.main_contributions[:3], 1):
                print(f"  {i}. {contrib}")
                if i <= len(notes.contributions_evidence):
                    ev = notes.contributions_evidence[i-1]
                    print(f"     Quote: \"{ev.quote[:80]}...\"")

            print(f"\nMain Results:")
            for result in notes.main_results[:3]:
                print(f"  • {result}")

            response = input("\nApprove these notes? (yes/edit/reject): ").strip().lower()

            if response == "edit":
                print("Editing not implemented in demo. Notes remain flagged.")
            elif response == "reject":
                notes.needs_review = True
                notes.review_reason = input("Enter rejection reason: ").strip()
                self.db.insert_notes(notes)

    def run(
        self,
        papers: list,  # List of PaperRecord from Stage 2
        enable_self_consistency: bool = False,
        enable_spot_check: bool = True
    ) -> str:
        """
        Execute the full extraction pipeline.

        Returns status string.
        """
        print(f"[Stage 3] Starting extraction for {len(papers)} papers")

        for i, paper in enumerate(papers):
            print(f"\n[Stage 3] Processing paper {i+1}/{len(papers)}: {paper.title[:50]}...")

            if enable_self_consistency:
                notes, disagreed = self.self_consistency_check(paper)
            else:
                notes = self.extract_from_paper(paper)

            # Save to database
            self.db.insert_notes(notes)

        # Step 11: Spot check
        if enable_spot_check:
            self.spot_check()

        # Step 12: Freeze
        print(f"\n[Stage 3] Extraction complete")

        # Summary
        all_notes = self.db.get_all_notes()
        high = sum(1 for n in all_notes if n.confidence == "high")
        medium = sum(1 for n in all_notes if n.confidence == "medium")
        low = sum(1 for n in all_notes if n.confidence == "low")
        review = sum(1 for n in all_notes if n.needs_review)

        print(f"[Stage 3] Summary: {high} high, {medium} medium, {low} low confidence")
        print(f"[Stage 3] {review} papers flagged for review")

        return "ready_for_stage4"


# ============================================================================
# DEMO
# ============================================================================

def demo_extraction():
    """Demo function showing the extraction pipeline."""
    from langchain_openai import ChatOpenAI
    from retrieval import PaperRecord
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    agent = NoteExtractionAgent(llm=llm, db_path="demo_extraction.db")

    # Create a sample paper record
    paper = PaperRecord(
        paper_id="demo_paper_1",
        title="Large Language Models in Education: A Survey",
        abstract="This paper surveys the application of large language models (LLMs) in educational contexts. We review applications in intelligent tutoring, automated assessment, and content generation. Our analysis shows that GPT-4 achieves 85% accuracy on math problems, while fine-tuned models reach 92% on domain-specific tasks.",
        fulltext_status="abstract_only",
        year=2024
    )

    notes = agent.extract_from_paper(paper)
    print(f"\nExtracted notes:")
    print(f"Confidence: {notes.confidence}")
    print(f"Contributions: {notes.main_contributions}")
    print(f"Results: {notes.main_results}")

    return agent


if __name__ == "__main__":
    demo_extraction()
