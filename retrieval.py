"""
Stage 2: Literature Retrieval

Turns an approved ScopeDocument into a clean, deduplicated, screened corpus
of real papers with metadata and full text where possible.
"""

import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


# ============================================================================
# SCHEMAS
# ============================================================================

class PaperRecord(BaseModel):
    """Normalized paper record from any source."""
    # Identifiers
    paper_id: str = Field(..., description="Unique ID (DOI, arXiv ID, or generated)")
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None

    # Metadata
    title: str
    abstract: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    venue_type: Optional[str] = None  # journal, conference, workshop, preprint
    citation_count: int = Field(default=0)
    keywords: list[str] = Field(default_factory=list)

    # Full text
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_sha256: Optional[str] = None
    fulltext_status: str = Field(default="pending", description="pending, pdf, abstract_only, unavailable")

    # Screening
    screening_status: str = Field(default="pending", description="pending, include, exclude, maybe")
    screening_reason: Optional[str] = None
    screening_timestamp: Optional[str] = None

    # Ranking
    relevance_score: float = Field(default=0.0)
    semantic_similarity: Optional[float] = None
    citations_per_year: Optional[float] = None

    # Provenance
    sources: list[str] = Field(default_factory=list, description="Sources this record came from")
    query_ids: list[str] = Field(default_factory=list, description="Queries that found this paper")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SearchLog(BaseModel):
    """Log entry for each search query."""
    query_id: str
    query_text: str
    source: str
    timestamp: str
    params: dict = Field(default_factory=dict)
    results_count: int = 0
    status: str = "pending"
    error: Optional[str] = None
    response_cached: bool = False


def _normalize_abstract(value) -> Optional[str]:
    """Convert provider-specific abstract payloads into plain text."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        # OpenAlex returns an inverted index: {word: [token positions]}.
        tokens = []
        for word, positions in value.items():
            if isinstance(positions, list):
                tokens.extend(
                    (position, str(word))
                    for position in positions
                    if isinstance(position, int)
                )
        if tokens:
            return " ".join(word for _, word in sorted(tokens))
    return None


# ============================================================================
# API WRAPPERS (Source-Agnostic Interface)
# ============================================================================

class BaseSourceWrapper:
    """Base class for source wrappers."""

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        self.email = email
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"Research-Agent/1.0 ({email or 'anonymous'})"})
        self.rate_limit_delay = 1.0
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _retry_with_backoff(self, request_func, max_retries: int = 3):
        """Execute request with exponential backoff on failures."""
        for attempt in range(max_retries):
            try:
                return request_func()
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * self.rate_limit_delay
                time.sleep(wait_time)
        return None

    def search(self, query: str, year_range: tuple[int, int], limit: int = 50) -> list[dict]:
        """Search and return normalized paper dicts. Override in subclasses."""
        raise NotImplementedError

    def normalize(self, raw_paper: dict) -> dict:
        """Convert source-specific format to PaperRecord-compatible dict."""
        raise NotImplementedError


class SemanticScholarWrapper(BaseSourceWrapper):
    """Semantic Scholar API wrapper."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        super().__init__(email=email, api_key=api_key)
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self.rate_limit_delay = 1.0  # 100 requests per 5 min with key, else 100 per 5 min per IP

    def search(self, query: str, year_range: tuple[int, int], limit: int = 50) -> list[dict]:
        """Search Semantic Scholar for papers."""
        self._rate_limit()

        params = {
            "query": query,
            "limit": limit,
            "year": f"{year_range[0]}-{year_range[1]}",
            "fields": "paperId,title,abstract,year,venue,authors,openAccessPdf,citationCount,externalIds,publicationTypes"
        }

        def make_request():
            response = self.session.get(f"{self.BASE_URL}/paper/search", params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        try:
            data = self._retry_with_backoff(make_request)
            papers = data.get("data", [])
            return [self.normalize(p) for p in papers]
        except Exception as e:
            print(f"[Semantic Scholar] Search error: {e}")
            return []

    def normalize(self, raw_paper: dict) -> dict:
        """Normalize Semantic Scholar paper to common format."""
        # Extract identifiers
        external_ids = raw_paper.get("externalIds", {}) or {}

        # Extract authors
        authors = []
        for author in (raw_paper.get("authors") or []):
            if isinstance(author, dict) and author.get("name"):
                authors.append(author["name"])
            elif isinstance(author, str):
                authors.append(author)

        # Extract PDF URL
        pdf_info = raw_paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url") if isinstance(pdf_info, dict) else None

        # Determine venue type
        pub_types = raw_paper.get("publicationTypes") or []
        venue_type = None
        if "JournalArticle" in pub_types:
            venue_type = "journal"
        elif "Conference" in pub_types:
            venue_type = "conference"
        elif "Review" in pub_types:
            venue_type = "review"

        return {
            "paper_id": raw_paper.get("paperId"),
            "doi": external_ids.get("DOI"),
            "arxiv_id": external_ids.get("ArXiv"),
            "semantic_scholar_id": raw_paper.get("paperId"),
            "title": raw_paper.get("title", ""),
            "abstract": raw_paper.get("abstract"),
            "authors": authors,
            "year": raw_paper.get("year"),
            "venue": raw_paper.get("venue"),
            "venue_type": venue_type,
            "citation_count": raw_paper.get("citationCount", 0) or 0,
            "pdf_url": pdf_url,
            "sources": ["semantic_scholar"]
        }

    def get_references(self, paper_id: str) -> list[dict]:
        """Get papers cited by this paper."""
        self._rate_limit()

        params = {
            "fields": "paperId,title,abstract,year,venue,authors,citationCount,externalIds"
        }

        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            papers = []
            for ref in data.get("data", []):
                if ref.get("citedPaper"):
                    papers.append(self.normalize(ref["citedPaper"]))
            return papers
        except Exception as e:
            print(f"[Semantic Scholar] References error: {e}")
            return []

    def get_citations(self, paper_id: str) -> list[dict]:
        """Get papers that cite this paper."""
        self._rate_limit()

        params = {
            "fields": "paperId,title,abstract,year,venue,authors,citationCount,externalIds"
        }

        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            papers = []
            for cite in data.get("data", []):
                if cite.get("citingPaper"):
                    papers.append(self.normalize(cite["citingPaper"]))
            return papers
        except Exception as e:
            print(f"[Semantic Scholar] Citations error: {e}")
            return []


class ArxivWrapper(BaseSourceWrapper):
    """arXiv API wrapper."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, email: Optional[str] = None):
        super().__init__(email=email)
        self.rate_limit_delay = 3.0  # arXiv requires 1 request per 3 seconds

    def search(self, query: str, year_range: tuple[int, int], limit: int = 50) -> list[dict]:
        """Search arXiv for papers."""
        self._rate_limit()

        # Build query with year filter (arXiv doesn't support year filter in API, filter after)
        search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            papers = self._parse_arxiv_response(response.text, year_range)
            return [self.normalize(p) for p in papers]
        except Exception as e:
            print(f"[arXiv] Search error: {e}")
            return []

    def _parse_arxiv_response(self, xml_text: str, year_range: tuple[int, int]) -> list[dict]:
        """Parse arXiv XML response."""
        import xml.etree.ElementTree as ET

        papers = []
        try:
            root = ET.fromstring(xml_text)

            # arXiv uses Atom namespace
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }

            for entry in root.findall("atom:entry", ns):
                # Extract year from published date
                published = entry.find("atom:published", ns)
                year = None
                if published is not None and published.text:
                    year = int(published.text[:4])

                # Filter by year range
                if year and not (year_range[0] <= year <= year_range[1]):
                    continue

                # Extract arXiv ID from URL
                arxiv_id = None
                pdf_url = None
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    if "arxiv.org/abs/" in href:
                        arxiv_id = href.split("/abs/")[-1]
                    elif "arxiv.org/pdf/" in href:
                        pdf_url = href

                # Extract authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None and name.text:
                        authors.append(name.text)

                # Extract title and abstract
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)

                paper = {
                    "arxiv_id": arxiv_id,
                    "title": title_elem.text.strip() if title_elem is not None and title_elem.text else "",
                    "abstract": summary_elem.text.strip() if summary_elem is not None and summary_elem.text else None,
                    "authors": authors,
                    "year": year,
                    "venue": "arXiv",
                    "venue_type": "preprint",
                    "citation_count": 0,
                    "pdf_url": pdf_url,
                }
                papers.append(paper)

        except Exception as e:
            print(f"[arXiv] Parse error: {e}")

        return papers

    def normalize(self, raw_paper: dict) -> dict:
        """Normalize arXiv paper to common format."""
        return {
            "paper_id": raw_paper.get("arxiv_id") or f"arxiv:{raw_paper.get('arxiv_id')}",
            "doi": None,
            "arxiv_id": raw_paper.get("arxiv_id"),
            "title": raw_paper.get("title", ""),
            "abstract": raw_paper.get("abstract"),
            "authors": raw_paper.get("authors", []),
            "year": raw_paper.get("year"),
            "venue": raw_paper.get("venue", "arXiv"),
            "venue_type": raw_paper.get("venue_type", "preprint"),
            "citation_count": raw_paper.get("citation_count", 0),
            "pdf_url": raw_paper.get("pdf_url"),
            "sources": ["arxiv"]
        }


class OpenAlexWrapper(BaseSourceWrapper):
    """OpenAlex API wrapper."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None):
        super().__init__(email=email)
        self.rate_limit_delay = 0.5  # Polite pool (with email) = faster

    def search(self, query: str, year_range: tuple[int, int], limit: int = 50) -> list[dict]:
        """Search OpenAlex for papers."""
        self._rate_limit()

        # Build filter string
        filters = [
            f"from_publication_date:{year_range[0]}-01-01",
            f"to_publication_date:{year_range[1]}-12-31",
            "type:article"  # Focus on articles
        ]

        params = {
            "search": query,
            "filter": ",".join(filters),
            "per_page": limit,
            "mailto": self.email  # For polite pool
        }

        try:
            response = self.session.get(
                f"{self.BASE_URL}/works",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            papers = data.get("results", [])
            return [self.normalize(p) for p in papers]
        except Exception as e:
            print(f"[OpenAlex] Search error: {e}")
            return []

    def normalize(self, raw_paper: dict) -> dict:
        """Normalize OpenAlex paper to common format."""
        # Extract DOI
        doi = raw_paper.get("doi", "")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

        # Extract authors
        authors = []
        for authorship in (raw_paper.get("authorships") or []):
            author = authorship.get("author", {})
            if author and author.get("display_name"):
                authors.append(author["display_name"])

        # Extract year
        year = None
        pub_date = raw_paper.get("publication_date")
        if pub_date:
            year = int(pub_date[:4])

        # Extract venue type
        venue_type = None
        type_val = raw_paper.get("type")
        if type_val == "article":
            venue_type = "journal"
        elif type_val == "conference-paper":
            venue_type = "conference"

        # Extract PDF URL from open access
        oa = raw_paper.get("open_access", {}) or {}
        pdf_url = oa.get("oa_url")

        return {
            "paper_id": raw_paper.get("id", "").replace("https://openalex.org/", ""),
            "doi": doi,
            "openalex_id": raw_paper.get("id", "").replace("https://openalex.org/", ""),
            "title": raw_paper.get("title", ""),
            "abstract": _normalize_abstract(raw_paper.get("abstract_inverted_index")),
            "authors": authors,
            "year": year,
            "venue": raw_paper.get("primary_location", {}).get("source", {}).get("display_name"),
            "venue_type": venue_type,
            "citation_count": raw_paper.get("cited_by_count", 0) or 0,
            "pdf_url": pdf_url,
            "sources": ["openalex"]
        }


class CrossrefWrapper(BaseSourceWrapper):
    """Crossref API wrapper for DOI validation and gap filling."""

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: Optional[str] = None):
        super().__init__(email=email)
        self.rate_limit_delay = 1.0

    def search(self, query: str, year_range: tuple[int, int], limit: int = 50) -> list[dict]:
        """Search Crossref for papers."""
        self._rate_limit()

        params = {
            "query": query,
            "rows": limit,
            "filter": f"from-pub-year:{year_range[0]},until-pub-year:{year_range[1]}",
            "mailto": self.email
        }

        try:
            response = self.session.get(
                f"{self.BASE_URL}/works",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("message", {}).get("items", [])
            return [self.normalize(item) for item in items]
        except Exception as e:
            print(f"[Crossref] Search error: {e}")
            return []

    def lookup_by_doi(self, doi: str) -> Optional[dict]:
        """Look up a paper by DOI."""
        self._rate_limit()

        try:
            response = self.session.get(
                f"{self.BASE_URL}/works/{doi}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return self.normalize(data.get("message", {}))
        except Exception as e:
            print(f"[Crossref] DOI lookup error for {doi}: {e}")
            return None

    def normalize(self, raw_paper: dict) -> dict:
        """Normalize Crossref paper to common format."""
        # Extract authors
        authors = []
        for author in (raw_paper.get("author") or []):
            name_parts = []
            if author.get("given"):
                name_parts.append(author["given"])
            if author.get("family"):
                name_parts.append(author["family"])
            if name_parts:
                authors.append(" ".join(name_parts))

        # Extract year
        year = None
        published = raw_paper.get("published-print") or raw_paper.get("published-online") or {}
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        # Determine venue type
        venue_type = None
        type_val = raw_paper.get("type", "").lower()
        if "journal" in type_val:
            venue_type = "journal"
        elif "conference" in type_val or "proceedings" in type_val:
            venue_type = "conference"

        # Extract PDF URL
        pdf_url = None
        for link in (raw_paper.get("link") or []):
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        return {
            "paper_id": raw_paper.get("DOI", ""),
            "doi": raw_paper.get("DOI"),
            "title": raw_paper.get("title", [""])[0] if raw_paper.get("title") else "",
            "abstract": raw_paper.get("abstract"),
            "authors": authors,
            "year": year,
            "venue": raw_paper.get("container-title", [""])[0] if raw_paper.get("container-title") else None,
            "venue_type": venue_type,
            "citation_count": raw_paper.get("is-referenced-by-count", 0) or 0,
            "pdf_url": pdf_url,
            "sources": ["crossref"]
        }


class UnpaywallWrapper(BaseSourceWrapper):
    """Unpaywall API wrapper for finding open access PDFs."""

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, email: str):
        super().__init__(email=email)
        if not email:
            raise ValueError("Unpaywall requires an email address")
        self.rate_limit_delay = 1.0

    def get_pdf_url(self, doi: str) -> Optional[str]:
        """Get open access PDF URL for a DOI."""
        self._rate_limit()

        try:
            response = self.session.get(
                f"{self.BASE_URL}/{doi}",
                params={"email": self.email},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Check for OA locations
            for location in (data.get("oa_locations") or []):
                if location.get("url_for_pdf"):
                    return location["url_for_pdf"]

            # Fallback to best OA location
            best_oa = data.get("best_oa_location") or {}
            return best_oa.get("url_for_pdf") or best_oa.get("url")

        except Exception as e:
            print(f"[Unpaywall] Error for DOI {doi}: {e}")
            return None


# ============================================================================
# SQLITE CORPUS STORAGE
# ============================================================================

class CorpusDatabase:
    """SQLite database for storing paper corpus and search logs."""

    def __init__(self, db_path: str = "corpus.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                doi TEXT,
                arxiv_id TEXT,
                semantic_scholar_id TEXT,
                openalex_id TEXT,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                venue_type TEXT,
                citation_count INTEGER DEFAULT 0,
                keywords TEXT,
                pdf_url TEXT,
                pdf_path TEXT,
                pdf_sha256 TEXT,
                fulltext_status TEXT DEFAULT 'pending',
                screening_status TEXT DEFAULT 'pending',
                screening_reason TEXT,
                screening_timestamp TEXT,
                relevance_score REAL DEFAULT 0.0,
                semantic_similarity REAL,
                citations_per_year REAL,
                sources TEXT,
                query_ids TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Search log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_log (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                params TEXT,
                results_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error TEXT,
                response_cached INTEGER DEFAULT 0
            )
        """)

        # Create indexes for deduplication lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers(arxiv_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screening ON papers(screening_status)")

        self.conn.commit()

    def insert_paper(self, paper: PaperRecord) -> bool:
        """Insert or update a paper record."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO papers (
                    paper_id, doi, arxiv_id, semantic_scholar_id, openalex_id,
                    title, abstract, authors, year, venue, venue_type,
                    citation_count, keywords, pdf_url, pdf_path, pdf_sha256,
                    fulltext_status, screening_status, screening_reason, screening_timestamp,
                    relevance_score, semantic_similarity, citations_per_year,
                    sources, query_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.paper_id,
                paper.doi,
                paper.arxiv_id,
                paper.semantic_scholar_id,
                paper.openalex_id,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                paper.year,
                paper.venue,
                paper.venue_type,
                paper.citation_count,
                json.dumps(paper.keywords),
                paper.pdf_url,
                paper.pdf_path,
                paper.pdf_sha256,
                paper.fulltext_status,
                paper.screening_status,
                paper.screening_reason,
                paper.screening_timestamp,
                paper.relevance_score,
                paper.semantic_similarity,
                paper.citations_per_year,
                json.dumps(paper.sources),
                json.dumps(paper.query_ids),
                paper.created_at,
                paper.updated_at
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[Database] Insert error: {e}")
            return False

    def get_paper(self, paper_id: str) -> Optional[PaperRecord]:
        """Get a paper by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_paper(row)
        return None

    def get_paper_by_doi(self, doi: str) -> Optional[PaperRecord]:
        """Get a paper by DOI."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE doi = ?", (doi,))
        row = cursor.fetchone()
        if row:
            return self._row_to_paper(row)
        return None

    def get_paper_by_arxiv(self, arxiv_id: str) -> Optional[PaperRecord]:
        """Get a paper by arXiv ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_paper(row)
        return None

    def find_by_title_fuzzy(self, title: str, year: int, threshold: int = 95) -> Optional[PaperRecord]:
        """Find a paper by fuzzy title match and year."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE year = ?", (year,))
        for row in cursor.fetchall():
            paper = self._row_to_paper(row)
            similarity = fuzz.ratio(title.lower(), paper.title.lower())
            if similarity >= threshold:
                return paper
        return None

    def get_all_papers(self, screening_status: Optional[str] = None) -> list[PaperRecord]:
        """Get all papers, optionally filtered by screening status."""
        cursor = self.conn.cursor()
        if screening_status:
            cursor.execute(
                "SELECT * FROM papers WHERE screening_status = ? ORDER BY relevance_score DESC",
                (screening_status,)
            )
        else:
            cursor.execute("SELECT * FROM papers ORDER BY relevance_score DESC")
        return [self._row_to_paper(row) for row in cursor.fetchall()]

    def get_papers_count(self) -> dict:
        """Get counts by screening status."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT screening_status, COUNT(*) as count
            FROM papers
            GROUP BY screening_status
        """)
        return {row["screening_status"]: row["count"] for row in cursor.fetchall()}

    def insert_search_log(self, log: SearchLog) -> bool:
        """Insert a search log entry."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO search_log (
                    query_id, query_text, source, timestamp, params,
                    results_count, status, error, response_cached
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.query_id,
                log.query_text,
                log.source,
                log.timestamp,
                json.dumps(log.params),
                log.results_count,
                log.status,
                log.error,
                int(log.response_cached)
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[Database] Log insert error: {e}")
            return False

    def _row_to_paper(self, row: sqlite3.Row) -> PaperRecord:
        """Convert database row to PaperRecord."""
        return PaperRecord(
            paper_id=row["paper_id"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            semantic_scholar_id=row["semantic_scholar_id"],
            openalex_id=row["openalex_id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=json.loads(row["authors"]) if row["authors"] else [],
            year=row["year"],
            venue=row["venue"],
            venue_type=row["venue_type"],
            citation_count=row["citation_count"] or 0,
            keywords=json.loads(row["keywords"]) if row["keywords"] else [],
            pdf_url=row["pdf_url"],
            pdf_path=row["pdf_path"],
            pdf_sha256=row["pdf_sha256"],
            fulltext_status=row["fulltext_status"] or "pending",
            screening_status=row["screening_status"] or "pending",
            screening_reason=row["screening_reason"],
            screening_timestamp=row["screening_timestamp"],
            relevance_score=row["relevance_score"] or 0.0,
            semantic_similarity=row["semantic_similarity"],
            citations_per_year=row["citations_per_year"],
            sources=json.loads(row["sources"]) if row["sources"] else [],
            query_ids=json.loads(row["query_ids"]) if row["query_ids"] else [],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# RESPONSE CACHING
# ============================================================================

class ResponseCache:
    """Cache API responses to disk to avoid re-fetching."""

    def __init__(self, cache_dir: str = ".cache/api_responses"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, source: str, query: str) -> Optional[dict]:
        """Get cached response."""
        cache_key = hashlib.sha256(f"{source}:{query}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, source: str, query: str, response: dict):
        """Cache a response."""
        cache_key = hashlib.sha256(f"{source}:{query}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(response, f)
        except Exception as e:
            print(f"[Cache] Write error: {e}")


# ============================================================================
# LITERATURE RETRIEVAL AGENT
# ============================================================================

class LiteratureRetrievalAgent:
    """
    Main agent for Stage 2: Literature Retrieval.

    Orchestrates the retrieval pipeline from ScopeDocument to corpus database.
    """

    def __init__(
        self,
        llm,  # LangChain LLM instance
        email: str,
        scholar_api_key: Optional[str] = None,
        corpus_db_path: str = "corpus.db",
        pdf_dir: str = "corpus/pdfs",
        cache_dir: str = ".cache/api_responses"
    ):
        self.llm = llm
        self.email = email

        # Initialize source wrappers
        self.scholar = SemanticScholarWrapper(api_key=scholar_api_key, email=email)
        self.arxiv = ArxivWrapper(email=email)
        self.openalex = OpenAlexWrapper(email=email)
        self.crossref = CrossrefWrapper(email=email)
        self.unpaywall = UnpaywallWrapper(email=email)

        # Initialize database and cache
        self.db = CorpusDatabase(corpus_db_path)
        self.cache = ResponseCache(cache_dir)

        # PDF storage
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Corpus status
        self.status = "initialized"

    def build_queries(self, scope_doc) -> list[tuple[str, str]]:
        """
        Step 1: Build queries from scope document.

        Returns list of (query_id, query_text) tuples.
        """
        queries = []
        query_id = 0

        # Research question itself
        queries.append((f"q{query_id}", scope_doc.research_question))
        query_id += 1

        # Keyword combinations
        if scope_doc.keywords:
            for keyword in scope_doc.keywords[:5]:
                queries.append((f"q{query_id}", keyword))
                query_id += 1

        # Subtopic queries
        for subtopic in scope_doc.subtopics:
            queries.append((f"q{query_id}", f"{scope_doc.original_topic} {subtopic}"))
            query_id += 1

        # Limit to 20 queries
        return queries[:20]

    def search_all_sources(
        self,
        query_id: str,
        query_text: str,
        year_range: tuple[int, int],
        limit_per_source: int = 50
    ) -> list[dict]:
        """
        Step 2: Search multiple sources in parallel.
        """
        results = []

        # Search each source
        sources = [
            ("semantic_scholar", self.scholar),
            ("arxiv", self.arxiv),
            ("openalex", self.openalex),
            ("crossref", self.crossref)
        ]

        for source_name, source in sources:
            # Check cache first
            cached = self.cache.get(source_name, query_text)
            if cached:
                papers = cached.get("papers", [])
                for paper in papers:
                    paper["query_ids"] = paper.get("query_ids", []) + [query_id]
                results.extend(papers)

                # Log cached response
                log = SearchLog(
                    query_id=f"{query_id}_{source_name}",
                    query_text=query_text,
                    source=source_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    params={"year_range": year_range, "limit": limit_per_source},
                    results_count=len(papers),
                    status="success",
                    response_cached=True
                )
                self.db.insert_search_log(log)
                continue

            # Live search
            try:
                papers = source.search(query_text, year_range, limit_per_source)

                # Add query_id to each paper
                for paper in papers:
                    paper["query_ids"] = [query_id]

                results.extend(papers)

                # Cache response
                self.cache.set(source_name, query_text, {"papers": papers})

                # Log search
                log = SearchLog(
                    query_id=f"{query_id}_{source_name}",
                    query_text=query_text,
                    source=source_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    params={"year_range": year_range, "limit": limit_per_source},
                    results_count=len(papers),
                    status="success",
                    response_cached=False
                )
                self.db.insert_search_log(log)

            except Exception as e:
                # Log error
                log = SearchLog(
                    query_id=f"{query_id}_{source_name}",
                    query_text=query_text,
                    source=source_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    params={"year_range": year_range, "limit": limit_per_source},
                    results_count=0,
                    status="error",
                    error=str(e),
                    response_cached=False
                )
                self.db.insert_search_log(log)

        return results

    def deduplicate(self, papers: list[dict]) -> list[PaperRecord]:
        """
        Step 4: Deduplicate papers by DOI, arXiv ID, or fuzzy title match.
        """
        unique_papers = {}

        for paper_dict in papers:
            paper_dict = dict(paper_dict)
            paper_dict["abstract"] = _normalize_abstract(
                paper_dict.get("abstract")
            )
            paper = PaperRecord(**paper_dict)

            # Check for existing duplicate
            existing = None

            # Match by normalized DOI
            if paper.doi:
                existing = self.db.get_paper_by_doi(paper.doi.lower())

            # Match by arXiv ID
            if not existing and paper.arxiv_id:
                existing = self.db.get_paper_by_arxiv(paper.arxiv_id)

            # Match by fuzzy title + year
            if not existing and paper.title and paper.year:
                existing = self.db.find_by_title_fuzzy(paper.title, paper.year, threshold=95)

            if existing:
                # Merge: keep richest fields
                merged = self._merge_papers(existing, paper)
                unique_papers[merged.paper_id] = merged
            else:
                # Generate paper_id if missing
                if not paper.paper_id:
                    paper.paper_id = self._generate_paper_id(paper)

                unique_papers[paper.paper_id] = paper

        return list(unique_papers.values())

    def _merge_papers(self, existing: PaperRecord, new: PaperRecord) -> PaperRecord:
        """Merge two paper records, keeping richest fields."""
        merged = existing.model_copy()

        # Keep richer abstract
        if not merged.abstract and new.abstract:
            merged.abstract = new.abstract

        # Keep higher citation count
        if new.citation_count > merged.citation_count:
            merged.citation_count = new.citation_count

        # Merge sources
        for source in new.sources:
            if source not in merged.sources:
                merged.sources.append(source)

        # Merge query IDs
        for qid in new.query_ids:
            if qid not in merged.query_ids:
                merged.query_ids.append(qid)

        # Keep PDF URL if new one exists
        if not merged.pdf_url and new.pdf_url:
            merged.pdf_url = new.pdf_url

        # Fill missing identifiers
        if not merged.doi and new.doi:
            merged.doi = new.doi
        if not merged.arxiv_id and new.arxiv_id:
            merged.arxiv_id = new.arxiv_id
        if not merged.semantic_scholar_id and new.semantic_scholar_id:
            merged.semantic_scholar_id = new.semantic_scholar_id
        if not merged.openalex_id and new.openalex_id:
            merged.openalex_id = new.openalex_id

        merged.updated_at = datetime.now(timezone.utc).isoformat()

        return merged

    def _generate_paper_id(self, paper: PaperRecord) -> str:
        """Generate a unique paper ID."""
        if paper.doi:
            return f"doi:{paper.doi.lower()}"
        if paper.arxiv_id:
            return f"arxiv:{paper.arxiv_id}"
        if paper.semantic_scholar_id:
            return f"ss:{paper.semantic_scholar_id}"
        if paper.openalex_id:
            return f"oa:{paper.openalex_id}"

        # Fallback: hash of title
        title_hash = hashlib.sha256(paper.title.encode()).hexdigest()[:12]
        return f"gen:{title_hash}"

    def apply_hard_filters(
        self,
        papers: list[PaperRecord],
        scope_doc,
        year_range: tuple[int, int]
    ) -> list[PaperRecord]:
        """
        Step 5: Apply hard filters based on scope document.
        """
        filtered = []

        for paper in papers:
            # Filter: year range
            if paper.year:
                if not (year_range[0] <= paper.year <= year_range[1]):
                    continue

            # Filter: must have abstract or DOI
            if not paper.abstract and not paper.doi:
                continue

            # Filter: exclusion criteria keywords
            if scope_doc.exclusion_criteria:
                text_to_check = f"{paper.title} {paper.abstract or ''}".lower()
                excluded = False
                for criterion in scope_doc.exclusion_criteria:
                    # Simple keyword check (could be enhanced)
                    keywords = criterion.lower().split()
                    if all(kw in text_to_check for kw in keywords):
                        excluded = True
                        break
                if excluded:
                    continue

            # Filter: language (English only for now)
            # In practice, you'd use a language detection library
            # For now, we assume papers from these sources are predominantly English

            filtered.append(paper)

        return filtered

    def rank_papers(
        self,
        papers: list[PaperRecord],
        research_question: str,
        top_n: int = 100
    ) -> list[PaperRecord]:
        """
        Step 6: Rank papers by relevance.
        """
        # For now, use simple keyword overlap for semantic similarity
        # In production, use actual embeddings

        question_words = set(research_question.lower().split())

        for paper in papers:
            # Calculate semantic similarity (simplified)
            if paper.abstract:
                abstract_words = set(paper.abstract.lower().split())
                overlap = len(question_words & abstract_words)
                similarity = overlap / len(question_words) if question_words else 0
                paper.semantic_similarity = min(similarity, 1.0)
            else:
                paper.semantic_similarity = 0.0

            # Calculate citations per year
            if paper.year and paper.year > 0:
                current_year = datetime.now(timezone.utc).year
                years_old = max(current_year - paper.year, 1)
                paper.citations_per_year = paper.citation_count / years_old
            else:
                paper.citations_per_year = 0.0

            # Normalize citations per year (log scale)
            import math
            normalized_citations = math.log1p(paper.citations_per_year) / 10.0  # Normalize to ~0-1

            # Venue bonus (simplified)
            venue_bonus = 0.0
            if paper.venue_type == "journal":
                venue_bonus = 0.8
            elif paper.venue_type == "conference":
                venue_bonus = 0.6
            elif paper.venue_type == "review":
                venue_bonus = 0.7
            else:
                venue_bonus = 0.4

            # Calculate final score
            paper.relevance_score = (
                0.6 * paper.semantic_similarity +
                0.25 * normalized_citations +
                0.15 * venue_bonus
            )

        # Sort by relevance score and return top N
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers[:top_n]

    def snowballing(
        self,
        seed_papers: list[PaperRecord],
        year_range: tuple[int, int]
    ) -> list[dict]:
        """
        Step 7: Snowballing - fetch references and citations.
        """
        new_papers = []

        for paper in seed_papers[:15]:
            if paper.semantic_scholar_id:
                # Get references
                refs = self.scholar.get_references(paper.semantic_scholar_id)
                new_papers.extend(refs)

                # Get citations
                cites = self.scholar.get_citations(paper.semantic_scholar_id)
                new_papers.extend(cites)

        return new_papers

    def llm_screening(
        self,
        papers: list[PaperRecord],
        scope_doc
    ) -> list[PaperRecord]:
        """
        Step 8: LLM-based screening with inclusion/exclusion criteria.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        class ScreeningResult(BaseModel):
            decision: str = Field(description="include, exclude, or maybe")
            reason: str = Field(description="One-line reason for the decision")

        parser = PydanticOutputParser(pydantic_object=ScreeningResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant screening papers for inclusion in a literature review.

Research Question: {research_question}

Inclusion Criteria:
{inclusion_criteria}

Exclusion Criteria:
{exclusion_criteria}

For each paper, decide whether to include, exclude, or mark as maybe.
Provide a one-line reason for your decision.

{format_instructions}"""),
            ("human", """Paper Title: {title}

Abstract: {abstract}

Year: {year}
Venue: {venue}

Make a screening decision.""")
        ]).partial(
            research_question=scope_doc.research_question,
            inclusion_criteria="\n".join(f"- {c}" for c in scope_doc.inclusion_criteria),
            exclusion_criteria="\n".join(f"- {c}" for c in scope_doc.exclusion_criteria),
            format_instructions=parser.get_format_instructions()
        )

        chain = prompt | self.llm | parser

        screened_papers = []
        maybe_papers = []

        for paper in papers:
            try:
                result = chain.invoke({
                    "title": paper.title,
                    "abstract": paper.abstract or "No abstract available",
                    "year": paper.year or "Unknown",
                    "venue": paper.venue or "Unknown"
                })

                paper.screening_status = result.decision
                paper.screening_reason = result.reason
                paper.screening_timestamp = datetime.now(timezone.utc).isoformat()

                if result.decision == "maybe":
                    maybe_papers.append(paper)

                screened_papers.append(paper)

            except Exception as e:
                print(f"[Screening] Error for paper {paper.paper_id}: {e}")
                paper.screening_status = "maybe"
                paper.screening_reason = f"Screening error: {e}"
                screened_papers.append(paper)

        # CLI for "maybe" papers
        if maybe_papers:
            print(f"\n[Stage 2] {len(maybe_papers)} papers marked as 'maybe'. Manual review required.")
            for paper in maybe_papers:
                print(f"\nTitle: {paper.title}")
                print(f"Reason: {paper.screening_reason}")
                decision = input("Include this paper? (yes/no): ").strip().lower()
                if decision in ["yes", "y"]:
                    paper.screening_status = "include"
                else:
                    paper.screening_status = "exclude"

        return screened_papers

    def download_full_text(
        self,
        paper: PaperRecord
    ) -> PaperRecord:
        """
        Step 9: Download full-text PDF for a paper.
        """
        pdf_url = paper.pdf_url

        # Try sources in order
        if not pdf_url:
            # 1. Check Semantic Scholar openAccessPdf (already in pdf_url from wrapper)

            # 2. Try arXiv
            if paper.arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

            # 3. Try Unpaywall
            if not pdf_url and paper.doi:
                pdf_url = self.unpaywall.get_pdf_url(paper.doi)

        if pdf_url:
            try:
                # Download PDF
                response = requests.get(pdf_url, timeout=60, stream=True)
                response.raise_for_status()

                # Generate filename
                safe_title = re.sub(r'[^\w\s-]', '', paper.title[:50]).strip().replace(' ', '_')
                filename = f"{paper.paper_id}_{safe_title}.pdf"
                pdf_path = self.pdf_dir / filename

                # Save PDF
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Compute SHA-256
                sha256 = hashlib.sha256()
                with open(pdf_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)

                paper.pdf_path = str(pdf_path)
                paper.pdf_sha256 = sha256.hexdigest()
                paper.fulltext_status = "pdf"
                paper.pdf_url = pdf_url

                print(f"[Stage 2] Downloaded PDF: {paper.title[:50]}...")

            except Exception as e:
                print(f"[Stage 2] PDF download failed for {paper.paper_id}: {e}")
                paper.fulltext_status = "abstract_only"
        else:
            paper.fulltext_status = "abstract_only"

        return paper

    def check_saturation(
        self,
        previous_count: int,
        new_count: int
    ) -> bool:
        """
        Step 10: Check if corpus has reached saturation.
        """
        if previous_count == 0:
            return False

        growth_rate = (new_count - previous_count) / previous_count

        # Stop if growth rate is below 10%
        if growth_rate < 0.1:
            print(f"[Stage 2] Saturation reached. Growth rate: {growth_rate:.1%}")
            return True

        return False

    def run(
        self,
        scope_doc,
        top_n: int = 100
    ) -> str:
        """
        Execute the full retrieval pipeline.

        Returns status string: "ready_for_stage3" or "needs_expansion".
        """
        self.status = "running"

        # Step 1: Build queries
        queries = self.build_queries(scope_doc)
        print(f"[Stage 2] Built {len(queries)} queries")

        year_range = scope_doc.year_range

        # Step 2-3: Search and normalize
        all_papers = []
        for query_id, query_text in queries:
            papers = self.search_all_sources(
                query_id,
                query_text,
                year_range,
                limit_per_source=30
            )
            all_papers.extend(papers)

        print(f"[Stage 2] Found {len(all_papers)} raw results")

        # Step 4: Deduplicate
        unique_papers = self.deduplicate(all_papers)
        print(f"[Stage 2] After deduplication: {len(unique_papers)} unique papers")

        # Step 5: Hard filters
        filtered_papers = self.apply_hard_filters(unique_papers, scope_doc, year_range)
        print(f"[Stage 2] After filtering: {len(filtered_papers)} papers")

        # Check for minimum corpus size
        if len(filtered_papers) < 15:
            print("[Stage 2] Warning: Corpus too small (< 15 papers). Consider expanding keywords.")
            self.status = "needs_expansion"
            return self.status

        # Step 6: Rank
        ranked_papers = self.rank_papers(
            filtered_papers,
            scope_doc.research_question,
            top_n=top_n
        )
        print(f"[Stage 2] Ranked top {len(ranked_papers)} papers")

        # Step 7: Snowballing (1 round)
        print("[Stage 2] Running snowballing...")
        snowball_papers = self.snowballing(ranked_papers[:10], year_range)

        # Deduplicate and filter snowball results
        snowball_unique = self.deduplicate(snowball_papers)
        snowball_filtered = self.apply_hard_filters(snowball_unique, scope_doc, year_range)

        # Combine and re-rank
        combined = ranked_papers + snowball_filtered
        combined_unique = self.deduplicate([p.model_dump() for p in combined])
        final_ranked = self.rank_papers(
            combined_unique,
            scope_doc.research_question,
            top_n=top_n
        )

        print(f"[Stage 2] After snowballing: {len(final_ranked)} papers")

        # Step 8: LLM screening
        print("[Stage 2] Running LLM screening...")
        screened_papers = self.llm_screening(final_ranked, scope_doc)

        # Count by status
        included = [p for p in screened_papers if p.screening_status == "include"]
        # A paper without a DOI or arXiv identifier cannot be used as a
        # citable source in the pipeline.
        for paper in included:
            if not paper.doi and not paper.arxiv_id:
                paper.screening_status = "exclude"
                paper.screening_reason = (
                    "No DOI or arXiv ID retrieved; not eligible as a citable source"
                )
                self.db.insert_paper(paper)
        included = [p for p in included if p.doi or p.arxiv_id]
        print(f"[Stage 2] Screening complete: {len(included)} papers included")

        if not included:
            self.status = "needs_expansion"
            print("[Stage 2] No included papers have a DOI or arXiv ID.")
            return self.status

        # Step 9: Download full text for included papers
        print("[Stage 2] Downloading full texts...")
        for paper in included:
            paper = self.download_full_text(paper)

        # Save all papers to database
        for paper in screened_papers:
            self.db.insert_paper(paper)

        # Step 10: Check saturation
        # (Would compare against previous corpus if running multiple iterations)

        # Step 11: Freeze
        self.status = "ready_for_stage3"
        print(f"\n[Stage 2] Complete. Corpus status: {self.status}")

        # Print summary
        counts = self.db.get_papers_count()
        print(f"[Stage 2] Corpus summary: {counts}")

        return self.status


# ============================================================================
# DEMO
# ============================================================================

def demo_retrieval():
    """Demo function showing the retrieval pipeline."""
    from langchain_openai import ChatOpenAI
    from scope import ScopeDocument, ContributionType
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Create a sample scope document (would come from Stage 1)
    scope_doc = ScopeDocument(
        original_topic="LLMs for education",
        research_question="How are large language models being applied in educational contexts?",
        contribution_type=ContributionType.SURVEY,
        question_rationale="This question addresses a rapidly growing field with practical implications.",
        year_range=(2020, 2026),
        subtopics=[
            "Intelligent Tutoring Systems",
            "Automated Assessment",
            "Content Generation",
            "Learning Analytics"
        ],
        keywords=["LLM education", "GPT tutoring", "language model assessment"],
        inclusion_criteria=[
            "Peer-reviewed publications",
            "Studies involving LLM applications in education",
            "English language publications"
        ],
        exclusion_criteria=[
            "Papers without clear methodology",
            "Non-educational applications"
        ],
        approved=True
    )

    agent = LiteratureRetrievalAgent(
        llm=llm,
        email="research@example.com",  # Replace with your email
        corpus_db_path="demo_corpus.db",
        pdf_dir="demo_corpus/pdfs"
    )

    status = agent.run(scope_doc, top_n=50)
    print(f"\nFinal status: {status}")

    return agent


if __name__ == "__main__":
    demo_retrieval()
