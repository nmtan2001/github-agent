"""
Documentation Comparison Module

This module provides tools for comparing generated documentation with existing documentation,
calculating similarity metrics, and providing insights about differences.
"""

import re
import difflib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import bert_score


@dataclass
class ComparisonMetrics:
    """Container for comparison metrics"""

    semantic_similarity: float
    structural_similarity: float
    content_coverage: float
    rouge_scores: Dict[str, float]
    bert_score: float
    readability_score: float
    word_count_ratio: float
    section_coverage: float


@dataclass
class ComparisonResult:
    """Result of documentation comparison"""

    metrics: ComparisonMetrics
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    diff_summary: str
    missing_sections: List[str]
    additional_sections: List[str]


class DocumentationComparator:
    """Advanced documentation comparison using multiple similarity metrics"""

    def __init__(self):
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        # Download required NLTK data
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        self.stop_words = set(stopwords.words("english"))

    def compare_documents(self, generated_doc: str, existing_doc: str, doc_type: str = "general") -> ComparisonResult:
        """Compare generated documentation with existing documentation"""

        # Calculate various similarity metrics
        semantic_similarity = self._calculate_semantic_similarity(generated_doc, existing_doc)
        structural_similarity = self._calculate_structural_similarity(generated_doc, existing_doc)
        content_coverage = self._calculate_content_coverage(generated_doc, existing_doc)
        rouge_scores = self._calculate_rouge_scores(generated_doc, existing_doc)
        bert_score_result = self._calculate_bert_score(generated_doc, existing_doc)
        readability_score = self._calculate_readability_score(generated_doc, existing_doc)
        word_count_ratio = self._calculate_word_count_ratio(generated_doc, existing_doc)
        section_coverage = self._calculate_section_coverage(generated_doc, existing_doc)

        # Create metrics object
        metrics = ComparisonMetrics(
            semantic_similarity=semantic_similarity,
            structural_similarity=structural_similarity,
            content_coverage=content_coverage,
            rouge_scores=rouge_scores,
            bert_score=bert_score_result,
            readability_score=readability_score,
            word_count_ratio=word_count_ratio,
            section_coverage=section_coverage,
        )

        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(generated_doc, existing_doc, metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, detailed_analysis)

        # Generate diff summary
        diff_summary = self._generate_diff_summary(generated_doc, existing_doc)

        # Find missing and additional sections
        missing_sections, additional_sections = self._analyze_section_differences(generated_doc, existing_doc)

        return ComparisonResult(
            metrics=metrics,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            diff_summary=diff_summary,
            missing_sections=missing_sections,
            additional_sections=additional_sections,
        )

    def _calculate_semantic_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        try:
            # Create embeddings for both documents
            embeddings1 = self.sentence_model.encode([doc1])
            embeddings2 = self.sentence_model.encode([doc2])

            # Calculate cosine similarity
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def _calculate_structural_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate structural similarity based on document organization"""

        def extract_structure(doc: str) -> List[str]:
            """Extract structural elements (headers, lists, etc.)"""
            structure = []

            # Extract headers
            headers = re.findall(r"^#+\s+(.+)$", doc, re.MULTILINE)
            structure.extend([f"header:{h.strip()}" for h in headers])

            # Extract list items
            list_items = re.findall(r"^\s*[-*+]\s+(.+)$", doc, re.MULTILINE)
            structure.extend([f"list:{item.strip()}" for item in list_items])

            # Extract code blocks
            code_blocks = re.findall(r"```[\s\S]*?```", doc)
            structure.extend([f"code:block" for _ in code_blocks])

            return structure

        struct1 = extract_structure(doc1)
        struct2 = extract_structure(doc2)

        if not struct1 and not struct2:
            return 1.0
        if not struct1 or not struct2:
            return 0.0

        # Calculate Jaccard similarity
        set1, set2 = set(struct1), set(struct2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _calculate_content_coverage(self, generated_doc: str, existing_doc: str) -> float:
        """Calculate how much of the existing content is covered in the generated doc"""

        def extract_key_concepts(doc: str) -> set:
            """Extract key concepts from document"""
            # Tokenize and filter
            words = word_tokenize(doc.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words and len(w) > 3]

            # Extract important phrases (simple bigrams)
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]

            return set(words + bigrams)

        concepts_existing = extract_key_concepts(existing_doc)
        concepts_generated = extract_key_concepts(generated_doc)

        if not concepts_existing:
            return 1.0

        covered = len(concepts_existing.intersection(concepts_generated))
        total = len(concepts_existing)

        return covered / total

    def _calculate_rouge_scores(self, generated_doc: str, existing_doc: str) -> Dict[str, float]:
        """Calculate ROUGE scores for document comparison"""
        try:
            scores = self.rouge_scorer.score(existing_doc, generated_doc)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def _calculate_bert_score(self, generated_doc: str, existing_doc: str) -> float:
        """Calculate BERTScore for semantic similarity"""
        try:
            # Split documents into sentences for better comparison
            generated_sents = sent_tokenize(generated_doc)
            existing_sents = sent_tokenize(existing_doc)

            if not generated_sents or not existing_sents:
                return 0.0

            # Calculate BERTScore
            P, R, F1 = bert_score.score(generated_sents, existing_sents, lang="en", verbose=False)

            return float(F1.mean())
        except Exception:
            return 0.0

    def _calculate_readability_score(self, generated_doc: str, existing_doc: str) -> float:
        """Compare readability using Flesch-Kincaid metrics"""

        def flesch_kincaid_score(text: str) -> float:
            """Calculate Flesch-Kincaid readability score"""
            sentences = sent_tokenize(text)
            words = word_tokenize(text)

            if len(sentences) == 0 or len(words) == 0:
                return 0.0

            avg_sentence_length = len(words) / len(sentences)

            # Count syllables (simple approximation)
            syllables = sum([self._count_syllables(word) for word in words])
            avg_syllables = syllables / len(words)

            # Flesch-Kincaid formula
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            return max(0, min(100, score))  # Clamp between 0-100

        score_generated = flesch_kincaid_score(generated_doc)
        score_existing = flesch_kincaid_score(existing_doc)

        # Return similarity in readability scores
        if score_existing == 0:
            return 1.0

        return 1.0 - abs(score_generated - score_existing) / 100.0

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not previous_char_was_vowel:
                    syllable_count += 1
                previous_char_was_vowel = True
            else:
                previous_char_was_vowel = False

        # Handle silent 'e'
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _calculate_word_count_ratio(self, generated_doc: str, existing_doc: str) -> float:
        """Calculate ratio of word counts"""
        words_generated = len(word_tokenize(generated_doc))
        words_existing = len(word_tokenize(existing_doc))

        if words_existing == 0:
            return 1.0 if words_generated == 0 else 0.0

        ratio = words_generated / words_existing
        # Convert to similarity score (closer to 1.0 is better)
        return 1.0 - abs(1.0 - ratio)

    def _calculate_section_coverage(self, generated_doc: str, existing_doc: str) -> float:
        """Calculate how well the generated doc covers existing sections"""

        def extract_sections(doc: str) -> set:
            """Extract section headers"""
            headers = re.findall(r"^#+\s+(.+)$", doc, re.MULTILINE)
            return set([h.strip().lower() for h in headers])

        sections_existing = extract_sections(existing_doc)
        sections_generated = extract_sections(generated_doc)

        if not sections_existing:
            return 1.0

        covered = len(sections_existing.intersection(sections_generated))
        total = len(sections_existing)

        return covered / total

    def _generate_detailed_analysis(
        self, generated_doc: str, existing_doc: str, metrics: ComparisonMetrics
    ) -> Dict[str, Any]:
        """Generate detailed analysis of the comparison"""

        analysis = {
            "length_comparison": {
                "generated_words": len(word_tokenize(generated_doc)),
                "existing_words": len(word_tokenize(existing_doc)),
                "length_ratio": metrics.word_count_ratio,
            },
            "structure_analysis": {
                "generated_headers": len(re.findall(r"^#+", generated_doc, re.MULTILINE)),
                "existing_headers": len(re.findall(r"^#+", existing_doc, re.MULTILINE)),
                "structural_similarity": metrics.structural_similarity,
            },
            "content_analysis": {
                "semantic_similarity": metrics.semantic_similarity,
                "content_coverage": metrics.content_coverage,
                "rouge_scores": metrics.rouge_scores,
            },
            "quality_metrics": {
                "bert_score": metrics.bert_score,
                "readability_similarity": metrics.readability_score,
                "section_coverage": metrics.section_coverage,
            },
        }

        return analysis

    def _generate_recommendations(self, metrics: ComparisonMetrics, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results"""

        recommendations = []

        # Semantic similarity recommendations
        if metrics.semantic_similarity < 0.7:
            recommendations.append(
                "Consider improving semantic similarity by including more relevant technical terms and concepts from the original documentation."
            )

        # Content coverage recommendations
        if metrics.content_coverage < 0.6:
            recommendations.append(
                "The generated documentation misses significant content from the original. Consider expanding coverage of key topics."
            )

        # Structural recommendations
        if metrics.structural_similarity < 0.5:
            recommendations.append(
                "Consider reorganizing the document structure to better match the original format and section organization."
            )

        # Length recommendations
        if metrics.word_count_ratio < 0.7:
            recommendations.append(
                "The generated documentation is significantly shorter. Consider adding more detailed explanations and examples."
            )
        elif metrics.word_count_ratio > 1.3:
            recommendations.append(
                "The generated documentation is verbose. Consider condensing content while maintaining key information."
            )

        # Section coverage recommendations
        if metrics.section_coverage < 0.8:
            recommendations.append(
                "Some important sections from the original documentation are missing. Review and add missing sections."
            )

        # ROUGE score recommendations
        if metrics.rouge_scores.get("rouge1", 0) < 0.4:
            recommendations.append(
                "Low word overlap with original documentation. Consider using more consistent terminology."
            )

        # Overall quality recommendation
        overall_score = (metrics.semantic_similarity + metrics.content_coverage + metrics.structural_similarity) / 3

        if overall_score < 0.6:
            recommendations.append(
                "Overall documentation quality needs improvement. Consider regenerating with more specific prompts or manual review."
            )
        elif overall_score > 0.8:
            recommendations.append(
                "Excellent documentation quality! Minor refinements may further improve alignment with original."
            )

        return recommendations

    def _generate_diff_summary(self, generated_doc: str, existing_doc: str) -> str:
        """Generate a summary of differences between documents"""

        # Split documents into lines for comparison
        generated_lines = generated_doc.splitlines()
        existing_lines = existing_doc.splitlines()

        # Generate unified diff
        diff = list(
            difflib.unified_diff(
                existing_lines, generated_lines, fromfile="existing_doc", tofile="generated_doc", lineterm=""
            )
        )

        # Count changes
        additions = len([line for line in diff if line.startswith("+")])
        deletions = len([line for line in diff if line.startswith("-")])

        summary = f"""
Diff Summary:
- Lines added: {additions}
- Lines removed: {deletions}
- Total changes: {additions + deletions}

Key differences:
{chr(10).join(diff[:20])}  # Show first 20 diff lines
"""

        return summary

    def _analyze_section_differences(self, generated_doc: str, existing_doc: str) -> Tuple[List[str], List[str]]:
        """Analyze differences in sections between documents"""

        def extract_sections(doc: str) -> List[str]:
            """Extract section headers with context"""
            sections = []
            lines = doc.split("\n")

            for i, line in enumerate(lines):
                if re.match(r"^#+\s+", line):
                    # Get some context around the header
                    context_start = max(0, i - 1)
                    context_end = min(len(lines), i + 3)
                    context = "\n".join(lines[context_start:context_end])
                    sections.append(context)

            return sections

        existing_sections = extract_sections(existing_doc)
        generated_sections = extract_sections(generated_doc)

        # Simple string matching for section comparison
        existing_headers = [
            re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)[0].strip()
            for section in existing_sections
            if re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)
        ]

        generated_headers = [
            re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)[0].strip()
            for section in generated_sections
            if re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)
        ]

        missing_sections = [h for h in existing_headers if h not in generated_headers]
        additional_sections = [h for h in generated_headers if h not in existing_headers]

        return missing_sections, additional_sections

    def compare_multiple_documents(
        self, generated_docs: List[Tuple[str, str]], existing_docs: List[Tuple[str, str]]
    ) -> Dict[str, ComparisonResult]:
        """Compare multiple document pairs"""

        results = {}
        existing_dict = {doc_type: content for doc_type, content in existing_docs}

        for doc_type, generated_content in generated_docs:
            if doc_type == "comprehensive":
                # Special handling for the comprehensive document
                return self.compare_comprehensive_documentation(generated_content, existing_docs)

            if doc_type in existing_dict:
                result = self.compare_documents(generated_content, existing_dict[doc_type], doc_type)
                results[doc_type] = result

        return results

    def compare_comprehensive_documentation(
        self, comprehensive_doc: str, existing_docs: List[Tuple[str, str]]
    ) -> Dict[str, ComparisonResult]:
        """Compare a single comprehensive document against all existing documentation combined."""
        if not existing_docs:
            return {}

        # Aggregate all existing documentation into a single string
        # We can add separators to give the model some context on the document boundaries
        aggregated_existing_content = "\n\n--- (New Document) ---\n\n".join(
            [f"# Existing Document: {doc_type}\n\n{content}" for doc_type, content in existing_docs]
        )

        # Now compare the comprehensive doc against the aggregated content
        # We'll store the result under the 'comprehensive' key
        result = self.compare_documents(comprehensive_doc, aggregated_existing_content, "comprehensive")

        return {"comprehensive": result}

    def generate_comparison_report(self, comparison_results: Dict[str, ComparisonResult]) -> str:
        """Generate a comprehensive comparison report"""

        report = "# Documentation Comparison Report\n\n"

        # Overall summary
        overall_scores = []
        for doc_type, result in comparison_results.items():
            overall_score = (
                result.metrics.semantic_similarity
                + result.metrics.content_coverage
                + result.metrics.structural_similarity
            ) / 3
            overall_scores.append(overall_score)

        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        report += f"## Overall Assessment\n"
        report += f"Average Quality Score: {avg_score:.2f}/1.0\n\n"

        # Individual document reports
        for doc_type, result in comparison_results.items():
            report += f"## {doc_type.title()} Documentation\n\n"

            report += f"### Metrics\n"
            report += f"- Semantic Similarity: {result.metrics.semantic_similarity:.3f}\n"
            report += f"- Content Coverage: {result.metrics.content_coverage:.3f}\n"
            report += f"- Structural Similarity: {result.metrics.structural_similarity:.3f}\n"
            report += f"- ROUGE-1: {result.metrics.rouge_scores.get('rouge1', 0):.3f}\n"
            report += f"- BERTScore: {result.metrics.bert_score:.3f}\n"
            report += f"- Section Coverage: {result.metrics.section_coverage:.3f}\n\n"

            report += f"### Recommendations\n"
            for rec in result.recommendations:
                report += f"- {rec}\n"
            report += "\n"

            if result.missing_sections:
                report += f"### Missing Sections\n"
                for section in result.missing_sections:
                    report += f"- {section}\n"
                report += "\n"

            if result.additional_sections:
                report += f"### Additional Sections\n"
                for section in result.additional_sections:
                    report += f"- {section}\n"
                report += "\n"

        return report
