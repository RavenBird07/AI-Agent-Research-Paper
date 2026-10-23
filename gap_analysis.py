"""
Stage 4: Gap Analysis

Compares extracted notes across all papers to find what the literature is missing,
with every gap backed by evidence. Selected gaps become the paper's motivation.
"""

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


# ============================================================================
# SCHEMAS
# ============================================================================

class GapType(str, Enum):
    METHOD = "method"
    EVALUATION = "evaluation"
    DOMAIN = "domain"
    CONTRADICTION = "contradiction"
    REPRODUCIBILITY = "reproducibility"
    SCALABILITY = "scalability"
    COVERAGE = "coverage"


class GapEvidence(BaseModel):
    """Evidence supporting a gap."""
    paper_id: str
    quote: str
    section: str


class Gap(BaseModel):
    """A validated gap in the literature."""
    gap_id: str
    gap_type: GapType
    statement: str
    why_it_matters: str
    evidence: list[GapEvidence] = Field(default_factory=list)
    coverage_cell: Optional[str] = None  # For coverage gaps: "method X × dataset Y"
    addressed: bool = False
    partially_addressed: bool = False
    addressing_papers: list[str] = Field(default_factory=list)
    importance_score: int = Field(default=0, ge=1, le=5)
    feasibility_score: int = Field(default=0, ge=1, le=5)
    alignment_score: int = Field(default=0, ge=1, le=5)
    total_score: float = 0.0
    selected: bool = False


class TaxonomyCluster(BaseModel):
    """A cluster of methods/approaches for Related Work structure."""
    cluster_id: str
    cluster_name: str
    description: str
    paper_ids: list[str] = Field(default_factory=list)
    representative_quotes: list[str] = Field(default_factory=list)


class LimitationCluster(BaseModel):
    """A cluster of limitations reported by multiple papers."""
    cluster_id: str
    cluster_name: str
    limitation_summary: str
    paper_ids: list[str] = Field(default_factory=list)
    count: int = 0


class Contradiction(BaseModel):
    """A contradiction between two papers."""
    paper_id_1: str
    paper_id_2: str
    topic: str
    claim_1: str
    claim_2: str
    evidence_1: str
    evidence_2: str
    dataset: Optional[str] = None
    metric: Optional[str] = None


class CoverageMatrix(BaseModel):
    """Coverage matrix showing method × dataset or method × metric."""
    matrix_type: str  # "method_dataset", "method_metric", "year_method"
    row_labels: list[str]
    column_labels: list[str]
    cells: dict[str, list[str]]  # "row:col" -> list of paper_ids
    empty_cells: list[str] = Field(default_factory=list)


class GapReport(BaseModel):
    """Final output of Stage 4: Gap Analysis."""
    # Taxonomy
    taxonomy: list[TaxonomyCluster] = Field(default_factory=list)

    # Coverage
    coverage_matrices: list[CoverageMatrix] = Field(default_factory=list)

    # Limitations
    limitation_clusters: list[LimitationCluster] = Field(default_factory=list)

    # Contradictions
    contradictions: list[Contradiction] = Field(default_factory=list)

    # Gaps
    candidate_gaps: list[Gap] = Field(default_factory=list)
    selected_gaps: list[Gap] = Field(default_factory=list)

    # Motivation
    motivation_paragraph: str = ""

    # Metadata
    n_papers_analyzed: int = 0
    n_papers_high_confidence: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# LABEL NORMALIZATION
# ============================================================================

class LabelNormalizer:
    """Normalize variant labels across papers."""

    def __init__(self, llm=None):
        self.llm = llm
        self.alias_maps: dict[str, dict[str, str]] = {}

    def normalize(self, labels: list[str], category: str) -> dict[str, str]:
        """
        Normalize labels within a category.

        Returns: dict mapping original label -> normalized label
        """
        if not labels:
            return {}

        # Check for existing alias map
        if category in self.alias_maps:
            return self._apply_alias_map(labels, self.alias_maps[category])

        # Build new alias map
        unique_labels = list(set(labels))
        alias_map = self._build_alias_map(unique_labels, category)
        self.alias_maps[category] = alias_map

        return self._apply_alias_map(labels, alias_map)

    def _build_alias_map(self, labels: list[str], category: str) -> dict[str, str]:
        """Build alias map using LLM or fuzzy matching."""
        if self.llm:
            return self._build_alias_map_with_llm(labels, category)
        else:
            return self._build_alias_map_fuzzy(labels)

    def _build_alias_map_with_llm(self, labels: list[str], category: str) -> dict[str, str]:
        """Use LLM to identify aliases."""
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        class AliasMap(BaseModel):
            canonical_labels: list[str]
            aliases: dict[str, str]

        parser = PydanticOutputParser(pydantic_object=AliasMap)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are organizing research metadata. Given a list of {category} labels,
identify which are aliases of the same thing and create a mapping.

Return:
1. A list of canonical (standard) labels
2. A mapping from each alias to its canonical form

{format_instructions}"""),
            ("human", "Labels: {labels}")
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "category": category,
                "labels": ", ".join(labels)
            })

            return result.aliases

        except Exception as e:
            print(f"[Normalizer] LLM alias building failed: {e}")
            return self._build_alias_map_fuzzy(labels)

    def _build_alias_map_fuzzy(self, labels: list[str]) -> dict[str, str]:
        """Build alias map using fuzzy matching."""
        alias_map = {}
        canonical_labels = []

        for label in labels:
            matched = False
            for canonical in canonical_labels:
                similarity = fuzz.ratio(label.lower(), canonical.lower())
                if similarity >= 85:
                    alias_map[label] = canonical
                    matched = True
                    break

            if not matched:
                canonical_labels.append(label)
                alias_map[label] = label

        return alias_map

    def _apply_alias_map(self, labels: list[str], alias_map: dict[str, str]) -> dict[str, str]:
        """Apply alias map to labels."""
        return {label: alias_map.get(label, label) for label in labels}

    def save_alias_maps(self, path: str):
        """Save alias maps to file for approval."""
        with open(path, "w") as f:
            json.dump(self.alias_maps, f, indent=2)

    def load_alias_maps(self, path: str):
        """Load approved alias maps."""
        with open(path, "r") as f:
            self.alias_maps = json.load(f)


# ============================================================================
# COVERAGE MATRIX BUILDER
# ============================================================================

class CoverageMatrixBuilder:
    """Build coverage matrices programmatically."""

    def __init__(self, normalizer: LabelNormalizer):
        self.normalizer = normalizer

    def build_method_dataset_matrix(
        self,
        paper_notes: list
    ) -> CoverageMatrix:
        """Build method × dataset coverage matrix."""
        # Collect all methods and datasets
        all_methods = []
        all_datasets = []
        method_dataset_papers: dict[str, list[str]] = defaultdict(list)

        for notes in paper_notes:
            method = notes.method_category
            datasets = notes.datasets or []

            if method:
                all_methods.append(method)

                for dataset in datasets:
                    all_datasets.append(dataset)
                    key = f"{method}:{dataset}"
                    method_dataset_papers[key].append(notes.paper_id)

        # Normalize labels
        method_map = self.normalizer.normalize(all_methods, "method")
        dataset_map = self.normalizer.normalize(all_datasets, "dataset")

        # Get unique normalized labels
        unique_methods = sorted(set(method_map.values()))
        unique_datasets = sorted(set(dataset_map.values()))

        # Build matrix
        cells = {}
        empty_cells = []

        for method in unique_methods:
            for dataset in unique_datasets:
                key = f"{method}:{dataset}"

                # Find papers for this cell
                papers_for_cell = []
                for orig_method, norm_method in method_map.items():
                    if norm_method != method:
                        continue
                    for orig_dataset, norm_dataset in dataset_map.items():
                        if norm_dataset != dataset:
                            continue
                        lookup_key = f"{orig_method}:{orig_dataset}"
                        papers_for_cell.extend(method_dataset_papers.get(lookup_key, []))

                cells[key] = list(set(papers_for_cell))

                if not papers_for_cell:
                    empty_cells.append(key)

        return CoverageMatrix(
            matrix_type="method_dataset",
            row_labels=unique_methods,
            column_labels=unique_datasets,
            cells=cells,
            empty_cells=empty_cells
        )

    def build_method_metric_matrix(
        self,
        paper_notes: list
    ) -> CoverageMatrix:
        """Build method × metric coverage matrix."""
        # Extract metrics from results
        all_methods = []
        all_metrics = []
        method_metric_papers: dict[str, list[str]] = defaultdict(list)

        for notes in paper_notes:
            method = notes.method_category
            results = notes.main_results or []

            if method:
                all_methods.append(method)

                # Extract metric names from results
                metrics = self._extract_metrics(results)
                for metric in metrics:
                    all_metrics.append(metric)
                    key = f"{method}:{metric}"
                    method_metric_papers[key].append(notes.paper_id)

        # Normalize
        method_map = self.normalizer.normalize(all_methods, "method")
        metric_map = self.normalizer.normalize(all_metrics, "metric")

        unique_methods = sorted(set(method_map.values()))
        unique_metrics = sorted(set(metric_map.values()))

        # Build matrix
        cells = {}
        empty_cells = []

        for method in unique_methods:
            for metric in unique_metrics:
                key = f"{method}:{metric}"

                papers_for_cell = []
                for orig_method, norm_method in method_map.items():
                    if norm_method != method:
                        continue
                    for orig_metric, norm_metric in metric_map.items():
                        if norm_metric != metric:
                            continue
                        lookup_key = f"{orig_method}:{orig_metric}"
                        papers_for_cell.extend(method_metric_papers.get(lookup_key, []))

                cells[key] = list(set(papers_for_cell))

                if not papers_for_cell:
                    empty_cells.append(key)

        return CoverageMatrix(
            matrix_type="method_metric",
            row_labels=unique_methods,
            column_labels=unique_metrics,
            cells=cells,
            empty_cells=empty_cells
        )

    def _extract_metrics(self, results: list[str]) -> list[str]:
        """Extract metric names from result strings."""
        metrics = []

        # Common metric patterns
        metric_patterns = [
            r"accuracy", r"precision", r"recall", r"F1", r"F1-score",
            r"BLEU", r"ROUGE", r"perplexity", r"accuracy@k",
            r"exact match", r"EM", r"MRR", r"NDCG"
        ]

        for result in results:
            result_lower = result.lower()
            for pattern in metric_patterns:
                if pattern.lower() in result_lower:
                    metrics.append(pattern)

        return list(set(metrics))


# ============================================================================
# CLUSTERING
# ============================================================================

class ClusterBuilder:
    """Build clusters for taxonomy, limitations, and future work."""

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def cluster_texts(
        self,
        texts: list[str],
        paper_ids: list[str],
        n_clusters: int = 5
    ) -> list[tuple[str, list[str]]]:
        """
        Cluster texts using embeddings.

        Returns: list of (cluster_label, list of paper_ids)
        """
        if not texts:
            return []

        # Generate embeddings
        if self.embedding_model:
            embeddings = [self.embedding_model.embed(t) for t in texts]
        else:
            embeddings = [self._dummy_embedding(t) for t in texts]

        embeddings = np.array(embeddings)

        # Simple k-means clustering
        clusters = self._kmeans(embeddings, n_clusters)

        # Group paper_ids by cluster
        cluster_papers = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_papers[cluster_id].append(paper_ids[i])

        return [(f"cluster_{k}", v) for k, v in cluster_papers.items()]

    def _kmeans(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple k-means implementation."""
        from sklearn.cluster import KMeans

        if len(embeddings) < n_clusters:
            n_clusters = max(1, len(embeddings))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)

    def _dummy_embedding(self, text: str) -> list[float]:
        """Generate dummy embedding for testing."""
        h = hashlib.sha256(text.encode()).hexdigest()
        return [float(int(h[i:i+2], 16)) / 255.0 for i in range(0, 64, 4)]

    def build_taxonomy(
        self,
        paper_notes: list,
        n_clusters: int = 5,
        llm=None
    ) -> list[TaxonomyCluster]:
        """Build method taxonomy for Related Work structure."""
        # Extract method descriptions
        texts = []
        paper_ids = []

        for notes in paper_notes:
            if notes.method_description:
                texts.append(notes.method_description)
                paper_ids.append(notes.paper_id)

        if not texts:
            return []

        # Cluster
        clusters = self.cluster_texts(texts, paper_ids, n_clusters)

        # Build taxonomy clusters
        taxonomy = []
        for cluster_id, cluster_paper_ids in clusters:
            # Get representative texts
            cluster_texts = [
                notes.method_description
                for notes in paper_notes
                if notes.paper_id in cluster_paper_ids and notes.method_description
            ]

            # Generate cluster name with LLM
            if llm:
                cluster_name = self._name_cluster_with_llm(llm, cluster_texts)
            else:
                cluster_name = cluster_id

            cluster = TaxonomyCluster(
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                description=f"Methods including: {', '.join(cluster_paper_ids[:3])}",
                paper_ids=cluster_paper_ids,
                representative_quotes=cluster_texts[:3]
            )
            taxonomy.append(cluster)

        return taxonomy

    def _name_cluster_with_llm(self, llm, texts: list[str]) -> str:
        """Generate cluster name using LLM."""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Give a short name (2-4 words) for a cluster of research methods described by these summaries."),
            ("human", "Summaries:\n{texts}")
        ])

        chain = prompt | llm

        try:
            result = chain.invoke({"texts": "\n".join(texts[:5])})
            return result.content.strip()[:50]
        except:
            return "Unknown Cluster"

    def build_limitation_clusters(
        self,
        paper_notes: list,
        min_papers: int = 4,
        llm=None
    ) -> list[LimitationCluster]:
        """Cluster limitations and find those reported by multiple papers."""
        # Extract limitations with paper IDs
        limitation_texts = []
        limitation_paper_ids = []

        for notes in paper_notes:
            for limitation in (notes.limitations or []):
                limitation_texts.append(limitation)
                limitation_paper_ids.append(notes.paper_id)

        if not limitation_texts:
            return []

        # Cluster
        clusters = self.cluster_texts(limitation_texts, limitation_paper_ids, n_clusters=10)

        # Build limitation clusters with count
        limitation_clusters = []
        cluster_num = 0

        for cluster_id, cluster_paper_ids in clusters:
            if len(set(cluster_paper_ids)) >= min_papers:
                cluster_texts = [
                    limitation_texts[i]
                    for i, pid in enumerate(limitation_paper_ids)
                    if pid in cluster_paper_ids
                ]

                if llm:
                    cluster_name = self._name_cluster_with_llm(llm, cluster_texts)
                else:
                    cluster_name = f"Limitation {cluster_num + 1}"

                cluster = LimitationCluster(
                    cluster_id=cluster_id,
                    cluster_name=cluster_name,
                    limitation_summary=cluster_texts[0] if cluster_texts else "",
                    paper_ids=list(set(cluster_paper_ids)),
                    count=len(set(cluster_paper_ids))
                )
                limitation_clusters.append(cluster)
                cluster_num += 1

        # Sort by count
        limitation_clusters.sort(key=lambda c: c.count, reverse=True)

        return limitation_clusters


# ============================================================================
# CONTRADICTION DETECTION
# ============================================================================

class ContradictionDetector:
    """Detect contradictions between papers."""

    def __init__(self, llm=None):
        self.llm = llm

    def detect_contradictions(
        self,
        paper_notes: list
    ) -> list[Contradiction]:
        """Find pairs of papers with contradictory claims."""
        contradictions = []

        # Group papers by dataset
        dataset_papers: dict[str, list] = defaultdict(list)
        for notes in paper_notes:
            for dataset in (notes.datasets or []):
                dataset_papers[dataset].append(notes)

        # Check for contradictions within each dataset
        for dataset, papers in dataset_papers.items():
            if len(papers) < 2:
                continue

            # Compare results on same metric
            for i, paper_1 in enumerate(papers):
                for paper_2 in papers[i+1:]:
                    contradiction = self._check_pair(paper_1, paper_2, dataset)
                    if contradiction:
                        contradictions.append(contradiction)

        return contradictions

    def _check_pair(
        self,
        paper_1,
        paper_2,
        dataset: str
    ) -> Optional[Contradiction]:
        """Check if two papers contradict each other."""
        # Extract numerical results
        results_1 = self._extract_numerical_results(paper_1.main_results or [])
        results_2 = self._extract_numerical_results(paper_2.main_results or [])

        # Check for conflicting claims on same metric
        for metric_1, value_1 in results_1.items():
            for metric_2, value_2 in results_2.items():
                if self._same_metric(metric_1, metric_2):
                    # Check for significant difference (>10%)
                    if abs(value_1 - value_2) / max(value_1, value_2) > 0.1:
                        # Potential contradiction
                        return Contradiction(
                            paper_id_1=paper_1.paper_id,
                            paper_id_2=paper_2.paper_id,
                            topic=f"Results on {dataset} for {metric_1}",
                            claim_1=f"{metric_1}: {value_1}",
                            claim_2=f"{metric_2}: {value_2}",
                            evidence_1=paper_1.main_results[0] if paper_1.main_results else "",
                            evidence_2=paper_2.main_results[0] if paper_2.main_results else "",
                            dataset=dataset,
                            metric=metric_1
                        )

        return None

    def _extract_numerical_results(self, results: list[str]) -> dict[str, float]:
        """Extract metric: value pairs from result strings."""
        import re

        metric_values = {}

        for result in results:
            # Find patterns like "accuracy: 85.2%" or "F1 = 0.92"
            matches = re.findall(r"(\w+)\s*[:=]\s*([\d.]+)\s*%?", result)
            for metric, value in matches:
                try:
                    metric_values[metric.lower()] = float(value)
                except ValueError:
                    continue

        return metric_values

    def _same_metric(self, metric_1: str, metric_2: str) -> bool:
        """Check if two metric names refer to the same metric."""
        m1 = metric_1.lower().replace("-", "").replace("_", "")
        m2 = metric_2.lower().replace("-", "").replace("_", "")

        return m1 == m2 or fuzz.ratio(m1, m2) >= 90


# ============================================================================
# GAP GENERATION
# ============================================================================

class GapGenerator:
    """Generate candidate gaps from analysis results."""

    def __init__(self, llm, corpus_db):
        self.llm = llm
        self.corpus_db = corpus_db

    def generate_gaps(
        self,
        taxonomy: list[TaxonomyCluster],
        coverage_matrices: list[CoverageMatrix],
        limitation_clusters: list[LimitationCluster],
        contradictions: list[Contradiction],
        paper_notes: list
    ) -> list[Gap]:
        """Generate candidate gaps using LLM."""
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        class GapList(BaseModel):
            gaps: list[Gap]

        parser = PydanticOutputParser(pydantic_object=GapList)

        # Build context for LLM
        context = self._build_context(
            taxonomy, coverage_matrices, limitation_clusters, contradictions
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are identifying gaps in research literature based on structured analysis.

RULES:
1. Use ONLY the provided matrix, clusters, and contradictions. No outside knowledge.
2. Every gap must cite at least 2 paper_ids and one verified quote, unless it's a coverage gap.
3. State the scope of absence: "No paper in the reviewed set of N papers evaluates X"
4. Do NOT claim "No one has studied X" - only claim absence within your corpus.
5. Return the gap type and say why it matters in one sentence.

GAP TYPES:
- method: Missing methodological approach
- evaluation: Missing evaluation dimension
- domain: Missing application domain
- contradiction: Conflicting results requiring resolution
- reproducibility: Reproducibility issues
- scalability: Scalability not addressed
- coverage: Empty cell in coverage matrix

{format_instructions}"""),
            ("human", """Based on this analysis, identify 8-15 gaps in the literature:

{n_papers} papers analyzed.

{context}

Generate gaps with supporting evidence.""")
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "n_papers": len(paper_notes),
                "context": context
            })

            # Validate evidence for each gap
            for gap in result.gaps:
                gap = self._validate_gap_evidence(gap, paper_notes)
                gap.gap_id = self._generate_gap_id(gap)

            return result.gaps

        except Exception as e:
            print(f"[GapGenerator] LLM gap generation failed: {e}")
            return self._generate_gaps_from_coverage(coverage_matrices, paper_notes)

    def _build_context(
        self,
        taxonomy: list[TaxonomyCluster],
        coverage_matrices: list[CoverageMatrix],
        limitation_clusters: list[LimitationCluster],
        contradictions: list[Contradiction]
    ) -> str:
        """Build context string for LLM."""
        parts = []

        # Taxonomy
        parts.append("## Method Taxonomy")
        for cluster in taxonomy:
            parts.append(f"- {cluster.cluster_name}: {len(cluster.paper_ids)} papers")

        # Coverage gaps
        parts.append("\n## Coverage Gaps (Empty Matrix Cells)")
        for matrix in coverage_matrices:
            parts.append(f"\n{matrix.matrix_type}:")
            for empty_cell in matrix.empty_cells[:10]:  # Limit for context
                parts.append(f"  - {empty_cell}")

        # Limitation clusters
        parts.append("\n## Limitation Clusters (≥4 papers)")
        for cluster in limitation_clusters:
            parts.append(f"- {cluster.cluster_name}: {cluster.count} papers")
            parts.append(f"  Papers: {', '.join(cluster.paper_ids[:5])}")

        # Contradictions
        parts.append("\n## Contradictions")
        for contra in contradictions[:5]:
            parts.append(f"- {contra.topic}: {contra.paper_id_1} vs {contra.paper_id_2}")

        return "\n".join(parts)

    def _validate_gap_evidence(
        self,
        gap: Gap,
        paper_notes: list
    ) -> Gap:
        """Validate that gap evidence exists in corpus."""
        # Build paper lookup
        paper_lookup = {notes.paper_id: notes for notes in paper_notes}

        validated_evidence = []

        for evidence in gap.evidence:
            # Check paper exists
            if evidence.paper_id not in paper_lookup:
                print(f"[GapValidation] Paper {evidence.paper_id} not in corpus")
                continue

            # Check quote exists (simplified check)
            paper = paper_lookup[evidence.paper_id]
            all_evidence = (
                paper.contributions_evidence +
                paper.results_evidence +
                paper.limitations_evidence
            )

            quote_found = any(
                fuzz.ratio(evidence.quote, ev.quote) >= 90
                for ev in all_evidence
            )

            if quote_found:
                validated_evidence.append(evidence)
            else:
                print(f"[GapValidation] Quote not found in paper {evidence.paper_id}")

        gap.evidence = validated_evidence

        return gap

    def _generate_gap_id(self, gap: Gap) -> str:
        """Generate unique gap ID."""
        content = f"{gap.gap_type}:{gap.statement}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _generate_gaps_from_coverage(
        self,
        coverage_matrices: list[CoverageMatrix],
        paper_notes: list
    ) -> list[Gap]:
        """Generate gaps from empty coverage cells (fallback)."""
        gaps = []

        for matrix in coverage_matrices:
            for empty_cell in matrix.empty_cells:
                row, col = empty_cell.split(":")

                gap = Gap(
                    gap_id=hashlib.sha256(empty_cell.encode()).hexdigest()[:12],
                    gap_type=GapType.COVERAGE,
                    statement=f"No paper in the reviewed set evaluates {row} on {col}",
                    why_it_matters=f"Understanding {row} performance on {col} is important for comprehensive evaluation",
                    coverage_cell=empty_cell,
                    importance_score=3,
                    feasibility_score=4,
                    alignment_score=3
                )
                gaps.append(gap)

        return gaps[:15]


# ============================================================================
# NOVELTY CHECK
# ============================================================================

class GapNoveltyChecker:
    """Check if gaps are actually novel or already addressed."""

    def __init__(self, retrieval_agent):
        self.retrieval_agent = retrieval_agent

    def check_novelty(
        self,
        gap: Gap,
        year_range: tuple[int, int],
        similarity_threshold: float = 0.85
    ) -> Gap:
        """Check if gap has been addressed in literature."""
        # Search for papers addressing this gap
        # In production, use the Stage 2 retrieval functions

        # For now, mark as novel
        gap.addressed = False
        gap.partially_addressed = False

        return gap


# ============================================================================
# GAP SCORING AND SELECTION
# ============================================================================

class GapScorer:
    """Score and select top gaps."""

    def score_gaps(
        self,
        gaps: list[Gap],
        research_question: str
    ) -> list[Gap]:
        """Score gaps by importance, feasibility, and alignment."""
        for gap in gaps:
            # Calculate total score
            gap.total_score = (
                gap.importance_score * 0.4 +
                gap.feasibility_score * 0.3 +
                gap.alignment_score * 0.3
            )

        # Sort by score
        gaps.sort(key=lambda g: g.total_score, reverse=True)

        return gaps

    def select_top_gaps(
        self,
        gaps: list[Gap],
        top_n: int = 3
    ) -> list[Gap]:
        """Select top N gaps."""
        for gap in gaps:
            gap.selected = False

        for gap in gaps[:top_n]:
            gap.selected = True

        return [g for g in gaps if g.selected]


# ============================================================================
# GAP ANALYSIS AGENT
# ============================================================================

class GapAnalysisAgent:
    """
    Main agent for Stage 4: Gap Analysis.

    Orchestrates the gap analysis pipeline.
    """

    def __init__(
        self,
        llm,
        extraction_db_path: str = "extraction.db",
        corpus_db_path: str = "corpus.db"
    ):
        self.llm = llm

        # Connect to databases
        self.extraction_db = self._connect_extraction_db(extraction_db_path)
        self.corpus_db_path = corpus_db_path

        # Initialize components
        self.normalizer = LabelNormalizer(llm=llm)
        self.matrix_builder = CoverageMatrixBuilder(self.normalizer)
        self.cluster_builder = ClusterBuilder()
        self.contradiction_detector = ContradictionDetector(llm=llm)
        self.gap_generator = GapGenerator(llm, corpus_db_path)
        self.novelty_checker = GapNoveltyChecker(None)  # Placeholder
        self.scorer = GapScorer()

    def _connect_extraction_db(self, db_path: str):
        """Connect to extraction database."""
        from extraction import ExtractionDatabase
        return ExtractionDatabase(db_path)

    def run(
        self,
        scope_doc,
        top_n_gaps: int = 3
    ) -> GapReport:
        """
        Execute the full gap analysis pipeline.

        Steps 1-14.
        """
        print(f"[Stage 4] Starting gap analysis")

        # Step 1: Load notes (high and medium confidence only for claims)
        paper_notes = self._load_notes()
        print(f"[Stage 4] Loaded {len(paper_notes)} paper notes")

        high_confidence = [n for n in paper_notes if n.confidence in ["high", "medium"]]
        print(f"[Stage 4] {len(high_confidence)} papers with high/medium confidence")

        if len(paper_notes) < 20:
            print(f"[Stage 4] Warning: Corpus too small ({len(paper_notes)} papers)")

        # Step 2: Normalize labels
        print(f"[Stage 4] Normalizing labels...")
        self._normalize_all_labels(paper_notes)

        # Step 3: Build taxonomy
        print(f"[Stage 4] Building taxonomy...")
        taxonomy = self.cluster_builder.build_taxonomy(
            high_confidence, n_clusters=5, llm=self.llm
        )
        print(f"[Stage 4] Created {len(taxonomy)} taxonomy clusters")

        # Step 4: Build coverage matrices
        print(f"[Stage 4] Building coverage matrices...")
        method_dataset_matrix = self.matrix_builder.build_method_dataset_matrix(high_confidence)
        method_metric_matrix = self.matrix_builder.build_method_metric_matrix(high_confidence)
        coverage_matrices = [method_dataset_matrix, method_metric_matrix]
        print(f"[Stage 4] Found {len(method_dataset_matrix.empty_cells)} method×dataset gaps")
        print(f"[Stage 4] Found {len(method_metric_matrix.empty_cells)} method×metric gaps")

        # Step 5: Aggregate limitations
        print(f"[Stage 4] Clustering limitations...")
        limitation_clusters = self.cluster_builder.build_limitation_clusters(
            high_confidence, min_papers=4, llm=self.llm
        )
        print(f"[Stage 4] Found {len(limitation_clusters)} limitation clusters (≥4 papers)")

        # Step 7: Detect contradictions
        print(f"[Stage 4] Detecting contradictions...")
        contradictions = self.contradiction_detector.detect_contradictions(high_confidence)
        print(f"[Stage 4] Found {len(contradictions)} potential contradictions")

        # Step 8: Generate gaps
        print(f"[Stage 4] Generating candidate gaps...")
        candidate_gaps = self.gap_generator.generate_gaps(
            taxonomy, coverage_matrices, limitation_clusters, contradictions, high_confidence
        )
        print(f"[Stage 4] Generated {len(candidate_gaps)} candidate gaps")

        # Step 10: Novelty check
        print(f"[Stage 4] Checking novelty...")
        for gap in candidate_gaps:
            gap = self.novelty_checker.check_novelty(gap, scope_doc.year_range)

        # Step 12: Score and select
        print(f"[Stage 4] Scoring gaps...")
        candidate_gaps = self.scorer.score_gaps(candidate_gaps, scope_doc.research_question)
        selected_gaps = self.scorer.select_top_gaps(candidate_gaps, top_n_gaps)
        print(f"[Stage 4] Selected top {len(selected_gaps)} gaps")

        # Step 13: Human checkpoint
        selected_gaps = self._human_checkpoint(selected_gaps, paper_notes)

        # Step 14: Draft motivation
        motivation = self._draft_motivation(selected_gaps, scope_doc, len(paper_notes))

        # Build report
        report = GapReport(
            taxonomy=taxonomy,
            coverage_matrices=coverage_matrices,
            limitation_clusters=limitation_clusters,
            contradictions=contradictions,
            candidate_gaps=candidate_gaps,
            selected_gaps=selected_gaps,
            motivation_paragraph=motivation,
            n_papers_analyzed=len(paper_notes),
            n_papers_high_confidence=len(high_confidence)
        )

        print(f"[Stage 4] Gap analysis complete")

        return report

    def _load_notes(self):
        """Load paper notes from extraction database."""
        return self.extraction_db.get_all_notes()

    def _normalize_all_labels(self, paper_notes: list):
        """Normalize all label categories."""
        # Collect all labels
        methods = [n.method_category for n in paper_notes if n.method_category]
        datasets = []
        for n in paper_notes:
            datasets.extend(n.datasets or [])

        # Normalize
        if methods:
            self.normalizer.normalize(methods, "method")
        if datasets:
            self.normalizer.normalize(datasets, "dataset")

    def _human_checkpoint(
        self,
        gaps: list[Gap],
        paper_notes: list
    ) -> list[Gap]:
        """Step 13: Human checkpoint for gap approval."""
        print(f"\n{'='*70}")
        print("SELECTED GAPS - AWAITING APPROVAL")
        print(f"{'='*70}")

        for i, gap in enumerate(gaps, 1):
            print(f"\n{i}. [{gap.gap_type.value}] {gap.statement}")
            print(f"   Why it matters: {gap.why_it_matters}")
            print(f"   Score: {gap.total_score:.2f}")

            if gap.evidence:
                print(f"   Evidence:")
                for ev in gap.evidence[:2]:
                    print(f"     - [{ev.paper_id}] \"{ev.quote[:60]}...\"")

        print(f"\n{'-'*70}")

        approved_gaps = []

        for gap in gaps:
            response = input(f"\nApprove gap: '{gap.statement[:50]}...'? (yes/edit/reject): ").strip().lower()

            if response in ["yes", "y"]:
                approved_gaps.append(gap)
            elif response == "edit":
                new_statement = input("Enter revised statement: ").strip()
                gap.statement = new_statement
                approved_gaps.append(gap)
            else:
                print(f"Gap rejected")

        return approved_gaps if approved_gaps else gaps

    def _draft_motivation(
        self,
        gaps: list[Gap],
        scope_doc,
        n_papers: int
    ) -> str:
        """Step 14: Draft motivation paragraph."""
        if not gaps:
            return ""

        # Build motivation from selected gaps
        gap_statements = [gap.statement for gap in gaps[:3]]

        motivation = f"""Despite extensive research on {scope_doc.original_topic},
our analysis of {n_papers} papers reveals several critical gaps.
{gap_statements[0] if gap_statements else ''}"""
        if len(gap_statements) > 1:
            motivation += f" Additionally, {gap_statements[1].lower()}."

        motivation += f""" These gaps motivate our contribution: to address {scope_doc.research_question.lower()}."""

        return motivation


# ============================================================================
# DEMO
# ============================================================================

def demo_gap_analysis():
    """Demo function showing gap analysis."""
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

    # Create sample scope document
    scope_doc = ScopeDocument(
        original_topic="LLMs for education",
        research_question="How are large language models being applied in educational contexts?",
        contribution_type=ContributionType.SURVEY,
        year_range=(2020, 2026),
        approved=True
    )

    agent = GapAnalysisAgent(
        llm=llm,
        extraction_db_path="demo_extraction.db"
    )

    report = agent.run(scope_doc, top_n_gaps=3)

    print(f"\nMotivation: {report.motivation_paragraph}")

    return agent


if __name__ == "__main__":
    demo_gap_analysis()
