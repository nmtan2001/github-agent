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

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, answer_correctness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os


@dataclass
class ComparisonMetrics:
    """Container for comparison metrics"""

    semantic_similarity: float
    rouge_scores: Dict[str, float]
    bert_score: float
    readability_score: float
    # word_count_ratio: float
    ragas_relevancy: Optional[float] = None
    ragas_correctness: Optional[float] = None


@dataclass
class ComparisonResult:
    """Result of documentation comparison"""

    metrics: ComparisonMetrics
    detailed_analysis: Dict[str, Any]


class DocumentationComparator:
    """Advanced documentation comparison using multiple similarity metrics"""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        self.ragas_llm = None
        used_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if used_api_key:
            self.ragas_llm = LangchainLLMWrapper(ChatOpenAI(model=model_name, api_key=used_api_key))

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

    def compare_documents(self, generated_doc: str, existing_doc: str) -> ComparisonResult:
        """Compare generated documentation with existing documentation"""

        # Calculate various similarity metrics
        semantic_similarity = self._calculate_semantic_similarity(generated_doc, existing_doc)
        rouge_scores = self._calculate_rouge_scores(generated_doc, existing_doc)
        bert_score_result = self._calculate_bert_score(generated_doc, existing_doc)
        readability_score = self._calculate_readability_score(generated_doc, existing_doc)

        # Ragas evaluation
        ragas_scores = self.evaluate_with_ragas(
            generated_doc=generated_doc,
            existing_doc=existing_doc,
            question="Provide a comprehensive documentation for the project, including overview, API reference, usage examples, and architecture.",
        )
        ragas_relevancy = ragas_scores.get("answer_relevancy")
        ragas_correctness = ragas_scores.get("answer_correctness")

        # Create metrics object
        metrics = ComparisonMetrics(
            semantic_similarity=semantic_similarity,
            rouge_scores=rouge_scores,
            bert_score=bert_score_result,
            readability_score=readability_score,
            ragas_relevancy=ragas_relevancy,
            ragas_correctness=ragas_correctness,
        )

        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(generated_doc, existing_doc, metrics)

        return ComparisonResult(
            metrics=metrics,
            detailed_analysis=detailed_analysis,
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

    def _calculate_rouge_scores(self, generated_doc: str, existing_doc: str) -> Dict[str, float]:
        """Calculate ROUGE scores for document comparison"""
        try:
            scores = self.rouge_scorer.score(existing_doc, generated_doc)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception:
            return {"rouge1": 0.0, "rougeL": 0.0}

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

    # def _calculate_word_count_ratio(self, generated_doc: str, existing_doc: str) -> float:
    #     """Calculate ratio of word counts"""
    #     words_generated = len(word_tokenize(generated_doc))
    #     words_existing = len(word_tokenize(existing_doc))

    #     if words_existing == 0:
    #         return 1.0 if words_generated == 0 else 0.0

    #     ratio = words_generated / words_existing
    #     # Convert to similarity score (closer to 1.0 is better)
    #     return 1.0 - abs(1.0 - ratio)

    def _generate_detailed_analysis(
        self, generated_doc: str, existing_doc: str, metrics: ComparisonMetrics
    ) -> Dict[str, Any]:
        """Generate detailed analysis of the comparison"""

        analysis = {
            "length_comparison": {
                "generated_words": len(word_tokenize(generated_doc)),
                "existing_words": len(word_tokenize(existing_doc)),
                # "length_ratio": metrics.word_count_ratio,
            },
            "structure_analysis": {
                "generated_headers": len(re.findall(r"^#+", generated_doc, re.MULTILINE)),
                "existing_headers": len(re.findall(r"^#+", existing_doc, re.MULTILINE)),
            },
            "content_analysis": {
                "semantic_similarity": metrics.semantic_similarity,
                "rouge_scores": metrics.rouge_scores,
            },
            "quality_metrics": {
                "bert_score": metrics.bert_score,
                "readability_similarity": metrics.readability_score,
            },
        }

        return analysis

    #     def _generate_diff_summary(self, generated_doc: str, existing_doc: str) -> str:
    #         """Generate a summary of differences between documents"""

    #         # Split documents into lines for comparison
    #         generated_lines = generated_doc.splitlines()
    #         existing_lines = existing_doc.splitlines()

    #         # Generate unified diff
    #         diff = list(
    #             difflib.unified_diff(
    #                 existing_lines, generated_lines, fromfile="existing_doc", tofile="generated_doc", lineterm=""
    #             )
    #         )

    #         # Count changes
    #         additions = len([line for line in diff if line.startswith("+")])
    #         deletions = len([line for line in diff if line.startswith("-")])

    #         summary = f"""
    # Diff Summary:
    # - Lines added: {additions}
    # - Lines removed: {deletions}
    # - Total changes: {additions + deletions}

    # Key differences:
    # {chr(10).join(diff[:20])}  # Show first 20 diff lines
    # """

    #         return summary

    # def _analyze_section_differences(self, generated_doc: str, existing_doc: str) -> Tuple[List[str], List[str]]:
    #     """Analyze differences in sections between documents"""

    #     def extract_sections(doc: str) -> List[str]:
    #         """Extract section headers with context"""
    #         sections = []
    #         lines = doc.split("\n")

    #         for i, line in enumerate(lines):
    #             if re.match(r"^#+\s+", line):
    #                 # Get some context around the header
    #                 context_start = max(0, i - 1)
    #                 context_end = min(len(lines), i + 3)
    #                 context = "\n".join(lines[context_start:context_end])
    #                 sections.append(context)

    #         return sections

    #     existing_sections = extract_sections(existing_doc)
    #     generated_sections = extract_sections(generated_doc)

    #     # Simple string matching for section comparison
    #     existing_headers = [
    #         re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)[0].strip()
    #         for section in existing_sections
    #         if re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)
    #     ]

    #     generated_headers = [
    #         re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)[0].strip()
    #         for section in generated_sections
    #         if re.findall(r"^#+\s+(.+)$", section, re.MULTILINE)
    #     ]

    #     missing_sections = [h for h in existing_headers if h not in generated_headers]
    #     additional_sections = [h for h in generated_headers if h not in existing_headers]

    #     return missing_sections, additional_sections

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
                result = self.compare_documents(generated_content, existing_dict[doc_type])
                results[doc_type] = result

        return results

    def compare_comprehensive_documentation(
        self, comprehensive_doc: str, existing_docs: List[Tuple[str, str]]
    ) -> Dict[str, ComparisonResult]:
        """Compare a single comprehensive document against all existing documentation combined."""
        if not existing_docs:
            return {}

        aggregated_existing_content = "\n\n--- (New Document) ---\n\n".join(
            [f"# Existing Document: {doc_type}\n\n{content}" for doc_type, content in existing_docs]
        )

        result = self.compare_documents(comprehensive_doc, aggregated_existing_content)

        return {"comprehensive": result}

    def generate_comparison_report(self, comparison_results: Dict[str, ComparisonResult]) -> str:
        """Generate a comprehensive comparison report"""

        report = "# Documentation Comparison Report\n\n"

        # Overall summary
        overall_scores = []
        for doc_type, result in comparison_results.items():
            quality_scores = [
                result.metrics.semantic_similarity,
                result.metrics.bert_score,
                result.metrics.readability_score,
                result.metrics.rouge_scores.get("rouge1", 0),
                result.metrics.rouge_scores.get("rougeL", 0),
            ]
            if result.metrics.ragas_relevancy is not None:
                quality_scores.append(result.metrics.ragas_relevancy)
            if result.metrics.ragas_correctness is not None:
                quality_scores.append(result.metrics.ragas_correctness)

            if quality_scores:
                overall_score = sum(quality_scores) / len(quality_scores)
                overall_scores.append(overall_score)

        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        report += f"## Overall Assessment\n"
        report += f"Average Quality Score: {avg_score:.2f}/1.0\n\n"

        # Individual document reports
        for doc_type, result in comparison_results.items():
            report += f"## {doc_type.title()} Documentation\n\n"

            report += f"### Metrics\n"
            report += f"- Semantic Similarity: {result.metrics.semantic_similarity:.3f}\n"
            report += f"- ROUGE-1: {result.metrics.rouge_scores.get('rouge1', 0):.3f}\n"
            report += f"- BERTScore: {result.metrics.bert_score:.3f}\n"
            if result.metrics.ragas_relevancy is not None:
                report += f"- Ragas Answer Relevancy: {result.metrics.ragas_relevancy:.3f}\n"
            if result.metrics.ragas_correctness is not None:
                report += f"- Ragas Answer Correctness: {result.metrics.ragas_correctness:.3f}\n"
            report += "\n"

        return report

    def evaluate_with_ragas(self, generated_doc: str, existing_doc: str, question: str) -> Dict[str, float]:
        """Evaluate documentation using Ragas"""
        if not self.ragas_llm:
            print("⚠️ Ragas LLM not initialized. Skipping Ragas evaluation.")
            return {}

        dataset_dict = {"question": [question], "answer": [generated_doc], "ground_truth": [existing_doc]}
        dataset = Dataset.from_dict(dataset_dict)

        metrics_to_use = [
            answer_relevancy,
            answer_correctness,
        ]

        try:
            print("🤖 Running Ragas evaluation...")
            result = evaluate(dataset, metrics=metrics_to_use, llm=self.ragas_llm)
            print(f"✅ Ragas evaluation successful: {result}")
            scores = {}

            # Safely access Ragas results, which returns a list of scores.
            try:
                relevancy_list = result["answer_relevancy"]
                if relevancy_list and len(relevancy_list) > 0:
                    relevancy = relevancy_list[0]
                    if isinstance(relevancy, float) and not np.isnan(relevancy):
                        scores["answer_relevancy"] = relevancy
                    else:
                        scores["answer_relevancy"] = None
                else:
                    scores["answer_relevancy"] = None
            except (KeyError, IndexError):
                scores["answer_relevancy"] = None

            try:
                correctness_list = result["answer_correctness"]
                if correctness_list and len(correctness_list) > 0:
                    correctness = correctness_list[0]
                    if isinstance(correctness, float) and not np.isnan(correctness):
                        scores["answer_correctness"] = correctness
                    else:
                        scores["answer_correctness"] = None
                else:
                    scores["answer_correctness"] = None
            except (KeyError, IndexError):
                scores["answer_correctness"] = None

            return scores
        except Exception as e:
            import traceback

            print(f"❌ Ragas evaluation failed: {e}")
            print(traceback.format_exc())
            return {}
