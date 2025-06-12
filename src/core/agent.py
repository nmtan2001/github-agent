"""
Main Documentation Agent

This module contains the main DocumentationAgent class that orchestrates
the entire documentation generation and comparison process.
"""

import os
import json
import re
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .analyzer import CodeAnalyzer, RepositoryMetadata, ModuleInfo
from .generator import DocumentationGenerator, DocumentationConfig, GeneratedDocument, EnhancedDocumentationGenerator
from .comparator import DocumentationComparator, ComparisonResult
from .document_reader import DocumentReader, DocumentChunk
from ..utils.llm import LLMManager
from ..utils.templates import TemplateManager

try:
    from git import Repo

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


@dataclass
class AgentConfig:
    """Configuration for the Documentation Agent"""

    repo_path: str
    output_dir: str = "generated_docs"
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    doc_types: List[str] = None
    include_comparison: bool = True
    save_intermediate: bool = True
    auto_cleanup: bool = True  # Whether to cleanup cloned repos automatically

    # Enhanced documentation settings
    use_enhanced_generator: bool = True  # Use enhanced generator with README embedding logic
    auto_discover_docs: bool = True
    docs_paths: List[str] = None
    exclude_patterns: List[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7

    def __post_init__(self):
        if self.doc_types is None:
            self.doc_types = ["readme", "api", "tutorial", "architecture"]

        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

        if self.docs_paths is None:
            self.docs_paths = []

        if self.exclude_patterns is None:
            self.exclude_patterns = ["*.git/*", "*.node_modules/*", "*.venv/*", "*/__pycache__/*"]

    @staticmethod
    def is_github_url(path: str) -> bool:
        """Check if the provided path is a GitHub URL"""
        github_patterns = [
            r"^https://github\.com/[\w\-_]+/[\w\-_]+/?$",
            r"^https://github\.com/[\w\-_]+/[\w\-_]+\.git/?$",
            r"^git@github\.com:[\w\-_]+/[\w\-_]+\.git$",
            r"^github\.com/[\w\-_]+/[\w\-_]+/?$",
        ]
        return any(re.match(pattern, path.strip()) for pattern in github_patterns)

    @staticmethod
    def normalize_github_url(url: str) -> str:
        """Normalize GitHub URL to HTTPS format"""
        url = url.strip()

        # Handle SSH format
        if url.startswith("git@github.com:"):
            url = url.replace("git@github.com:", "https://github.com/")

        # Add https:// if missing
        if url.startswith("github.com/"):
            url = "https://" + url

        # Remove .git suffix if present
        if url.endswith(".git"):
            url = url[:-4]

        # Remove trailing slash
        if url.endswith("/"):
            url = url[:-1]

        return url


@dataclass
class DocumentationReport:
    """Complete documentation analysis and generation report"""

    repository_metadata: RepositoryMetadata
    modules_analyzed: int
    generated_documents: List[GeneratedDocument]
    comparison_results: Optional[Dict[str, ComparisonResult]]
    execution_time: float
    success_rate: float
    recommendations: List[str]


class DocumentationAgent:
    """Main agent for automated documentation generation and analysis"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._cloned_repo_path = None  # Track if we cloned a repo for cleanup

        # Create output directory first (needed for logging)
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()

        # Handle GitHub URL automatic cloning
        actual_repo_path = self._handle_repo_path(config.repo_path)

        # Initialize components
        self.analyzer = CodeAnalyzer(actual_repo_path)

        # Choose generator based on configuration
        if config.use_enhanced_generator:
            self.llm_manager = LLMManager()
            self.template_manager = TemplateManager()
            self.enhanced_generator = EnhancedDocumentationGenerator(
                llm_manager=self.llm_manager,
                template_manager=self.template_manager,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        else:
            self.enhanced_generator = None

        self.generator = DocumentationGenerator(
            DocumentationConfig(
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                documentation_types=config.doc_types,
            )
        )
        self.comparator = DocumentationComparator() if config.include_comparison else None

        # Initialize state
        self.repository_metadata = None
        self.modules = []
        self.generated_docs = []
        self.comparison_results = {}
        self.existing_docs_loaded = False
        self.docs_summary = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the agent"""
        logger = logging.getLogger("DocumentationAgent")
        logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create file handler (output directory already exists)
        log_file = self.output_path / "agent.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def _handle_repo_path(self, repo_path: str) -> str:
        """Handle repository path - clone if it's a GitHub URL, otherwise return as-is"""
        if AgentConfig.is_github_url(repo_path):
            return self._clone_github_repo(repo_path)
        else:
            return repo_path

    def _clone_github_repo(self, github_url: str) -> str:
        """Clone a GitHub repository to a temporary directory"""
        if not GIT_AVAILABLE:
            raise RuntimeError(
                "GitPython is required for automatic cloning. Please install it with: pip install gitpython"
            )

        try:
            # Normalize the URL
            normalized_url = AgentConfig.normalize_github_url(github_url)

            # Extract repository name for the local directory
            repo_name = normalized_url.split("/")[-1]

            # Create a temporary directory in the output directory
            clone_dir = self.output_path / "cloned_repos" / repo_name
            clone_dir.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing directory if it exists
            if clone_dir.exists():
                shutil.rmtree(clone_dir)

            self.logger.info(f"Cloning repository from {normalized_url} to {clone_dir}")

            # Clone the repository
            repo = Repo.clone_from(normalized_url, clone_dir)

            # Store the cloned path for cleanup
            self._cloned_repo_path = str(clone_dir)

            self.logger.info(f"Successfully cloned repository to {clone_dir}")
            return str(clone_dir)

        except Exception as e:
            self.logger.error(f"Failed to clone repository {github_url}: {str(e)}")
            raise RuntimeError(f"Failed to clone repository: {str(e)}")

    def auto_discover_documentation(self, repo_path: Optional[str] = None) -> List[str]:
        """
        Automatically discover documentation files in the repository.

        Args:
            repo_path: Path to repository (uses config if not provided)

        Returns:
            List of discovered documentation paths
        """
        if not repo_path:
            repo_path = self.config.repo_path if not self._cloned_repo_path else self._cloned_repo_path

        doc_paths = []
        repo_root = Path(repo_path)

        # Common documentation directories
        doc_dirs = ["docs", "doc", "documentation", "guide", "guides"]

        # Common documentation files
        doc_files = [
            "README.md",
            "README.rst",
            "README.txt",
            "INSTALL.md",
            "INSTALLATION.md",
            "USAGE.md",
            "TUTORIAL.md",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "API.md",
            "REFERENCE.md",
        ]

        # Check for documentation directories
        for doc_dir in doc_dirs:
            dir_path = repo_root / doc_dir
            if dir_path.exists() and dir_path.is_dir():
                doc_paths.append(str(dir_path))
                self.logger.info(f"Found documentation directory: {dir_path}")

        # Check for documentation files in root
        for doc_file in doc_files:
            file_path = repo_root / doc_file
            if file_path.exists() and file_path.is_file():
                doc_paths.append(str(file_path))
                self.logger.info(f"Found documentation file: {file_path}")

        # Check for documentation in common subdirectories
        for subdir in ["examples", "tutorials", "samples"]:
            subdir_path = repo_root / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                # Look for markdown/rst files
                for ext in ["*.md", "*.rst", "*.txt"]:
                    found_files = list(subdir_path.rglob(ext))
                    if found_files:
                        doc_paths.extend([str(f) for f in found_files])
                        self.logger.info(f"Found {len(found_files)} documentation files in {subdir}")

        return doc_paths

    def load_existing_documentation(
        self, docs_paths: Optional[List[str]] = None, exclude_patterns: Optional[List[str]] = None
    ) -> bool:
        """
        Load existing documentation from specified paths.

        Args:
            docs_paths: List of paths to documentation directories/files
            exclude_patterns: Patterns to exclude from loading

        Returns:
            True if documentation was loaded successfully
        """
        if not self.enhanced_generator:
            self.logger.warning("Enhanced generator not available. Cannot load existing documentation.")
            return False

        docs_paths = docs_paths or self.config.docs_paths
        exclude_patterns = exclude_patterns or self.config.exclude_patterns

        if not docs_paths and self.config.auto_discover_docs:
            # Auto-discover documentation
            docs_paths = self.auto_discover_documentation()
            self.config.docs_paths.extend(docs_paths)

        if not docs_paths:
            self.logger.warning("No documentation paths provided")
            return False

        # Filter to existing paths
        existing_paths = [path for path in docs_paths if os.path.exists(path)]
        if not existing_paths:
            self.logger.warning(f"None of the provided documentation paths exist: {docs_paths}")
            return False

        self.logger.info(f"Loading existing documentation from: {existing_paths}")

        # Load documentation using enhanced generator
        success = self.enhanced_generator.load_existing_documentation(existing_paths, exclude_patterns)

        if success:
            self.existing_docs_loaded = True
            self.docs_summary = self.enhanced_generator.document_reader.get_documentation_summary(
                self.enhanced_generator.existing_docs
            )
            self.logger.info(f"Successfully loaded documentation: {self.docs_summary}")

        return success

    def generate_enhanced_documentation(
        self, doc_types: Optional[List[str]] = None, load_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Generate documentation using the enhanced generator with existing documentation context.

        Args:
            doc_types: Types of documentation to generate
            load_existing: Whether to load existing documentation first

        Returns:
            Enhanced generation results
        """
        if not self.enhanced_generator:
            raise RuntimeError("Enhanced generator not available. Set use_enhanced_generator=True in config.")

        doc_types = doc_types or self.config.doc_types

        # Load existing documentation if requested and not already loaded
        if load_existing and not self.existing_docs_loaded:
            self.load_existing_documentation()

        # Get the repository path (use cloned path if available)
        repo_path = self._cloned_repo_path or self.config.repo_path

        self.logger.info(f"Generating enhanced documentation for: {repo_path}")

        # Generate documentation with context
        result = self.enhanced_generator.generate_documentation_with_context(
            repo_path=repo_path,
            output_format="markdown",
            doc_types=doc_types,
            context_similarity_threshold=self.config.similarity_threshold,
        )

        self.logger.info("Enhanced documentation generation completed")
        return result

    def run_enhanced_pipeline(self) -> Dict[str, Any]:
        """Run the enhanced documentation generation pipeline with context awareness"""

        # Load existing documentation if enabled
        if self.config.auto_discover_docs:
            self._discover_and_load_documentation()

        # Generate documentation with context
        generated_docs = self.enhanced_generator.generate_documentation_with_context(
            repo_path=self.config.repo_path,
            output_format="markdown",
            doc_types=self.config.doc_types,
            context_similarity_threshold=0.7,
        )

        # Convert to format expected by the app (similar to standard pipeline)
        return {
            "documents": generated_docs,  # List of GeneratedDocument objects
            "metadata": {
                "total_documents": len(generated_docs),
                "enhanced_with_embeddings": True,
                "existing_docs_loaded": len(self.enhanced_generator.existing_docs) > 0,
                "generation_method": "enhanced_with_context",
            },
        }

    def _discover_and_load_documentation(self):
        """Discover and load existing documentation for the enhanced generator"""
        if not self.enhanced_generator:
            return

        try:
            # Auto-discover documentation paths
            discovered_paths = self.auto_discover_documentation()

            # Combine with manually configured paths
            all_paths = (self.config.docs_paths or []) + discovered_paths

            if all_paths:
                # Load documentation into the enhanced generator
                success = self.enhanced_generator.load_existing_documentation(all_paths)
                if success:
                    self.logger.info(f"Loaded existing documentation from {len(all_paths)} paths")
                else:
                    self.logger.warning("Failed to load existing documentation")
            else:
                self.logger.info("No existing documentation paths found")

        except Exception as e:
            self.logger.warning(f"Error discovering/loading documentation: {e}")

    def __del__(self):
        """Cleanup cloned repositories on object destruction"""
        self._cleanup_cloned_repo()

    def _cleanup_cloned_repo(self):
        """Clean up cloned repository if auto_cleanup is enabled"""
        if self._cloned_repo_path and self.config.auto_cleanup and Path(self._cloned_repo_path).exists():
            try:
                shutil.rmtree(self._cloned_repo_path)
                self.logger.info(f"Cleaned up cloned repository: {self._cloned_repo_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup cloned repository {self._cloned_repo_path}: {e}")

    def analyze_repository(self) -> Tuple[RepositoryMetadata, List[ModuleInfo]]:
        """Analyze the repository structure and extract metadata"""
        self.logger.info(f"Starting repository analysis: {self.config.repo_path}")

        try:
            self.repository_metadata, self.modules = self.analyzer.analyze_repository()

            self.logger.info(f"Analysis complete: {len(self.modules)} modules found")
            self.logger.info(f"Repository: {self.repository_metadata.name}")
            self.logger.info(f"Language: {self.repository_metadata.language}")
            self.logger.info(f"Files: {self.repository_metadata.file_count}")

            # Save analysis results if requested
            if self.config.save_intermediate:
                self._save_analysis_results()

            return self.repository_metadata, self.modules

        except Exception as e:
            self.logger.error(f"Repository analysis failed: {str(e)}")
            raise

    def generate_documentation(self, doc_types: Optional[List[str]] = None) -> List[GeneratedDocument]:
        """Generate documentation for the analyzed repository"""
        if not self.repository_metadata or not self.modules:
            raise ValueError("Repository must be analyzed before generating documentation")

        doc_types = doc_types or self.config.doc_types
        self.logger.info(f"Generating documentation types: {doc_types}")

        try:
            import time

            start_time = time.time()

            self.generated_docs = self.generator.generate_documentation(
                self.repository_metadata, self.modules, doc_types
            )

            generation_time = time.time() - start_time
            self.logger.info(f"Documentation generation completed in {generation_time:.2f} seconds")
            self.logger.info(f"Generated {len(self.generated_docs)} documents")

            # Save generated documents
            self._save_generated_documents()

            return self.generated_docs

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {str(e)}")
            raise

    def compare_with_existing(self, existing_docs_path: Optional[str] = None) -> Dict[str, ComparisonResult]:
        """Compare generated documentation with existing documentation"""
        if not self.comparator:
            self.logger.warning("Comparison disabled in configuration")
            return {}

        if not self.generated_docs:
            raise ValueError("Documentation must be generated before comparison")

        self.logger.info("Starting documentation comparison")

        try:
            # Find existing documentation
            existing_docs = self._find_existing_docs(existing_docs_path)

            if not existing_docs:
                self.logger.warning("No existing documentation found for comparison")
                return {}

            # Perform comparison
            generated_doc_pairs = [(doc.doc_type, doc.content) for doc in self.generated_docs]

            self.comparison_results = self.comparator.compare_multiple_documents(generated_doc_pairs, existing_docs)

            self.logger.info(f"Comparison completed for {len(self.comparison_results)} document types")

            # Save comparison results
            if self.config.save_intermediate:
                self._save_comparison_results()

            return self.comparison_results

        except Exception as e:
            self.logger.error(f"Documentation comparison failed: {str(e)}")
            raise

    def generate_report(self) -> DocumentationReport:
        """Generate a comprehensive report of the documentation process"""
        if not self.repository_metadata:
            raise ValueError("Repository analysis must be completed first")

        self.logger.info("Generating comprehensive report")

        # Calculate success rate
        total_requested = len(self.config.doc_types)
        total_generated = len(self.generated_docs)
        success_rate = total_generated / total_requested if total_requested > 0 else 0.0

        # Generate overall recommendations
        recommendations = self._generate_overall_recommendations()

        report = DocumentationReport(
            repository_metadata=self.repository_metadata,
            modules_analyzed=len(self.modules),
            generated_documents=self.generated_docs,
            comparison_results=self.comparison_results if self.comparison_results else None,
            execution_time=0.0,  # Would need to track this
            success_rate=success_rate,
            recommendations=recommendations,
        )

        # Save the report
        self._save_report(report)

        self.logger.info("Report generation completed")
        return report

    def run_full_pipeline(self, existing_docs_path: Optional[str] = None) -> DocumentationReport:
        """Run the complete documentation generation and analysis pipeline"""
        self.logger.info("Starting full documentation pipeline")

        import time

        start_time = time.time()

        try:
            # Step 1: Analyze repository
            self.analyze_repository()

            # Step 2: Generate documentation
            self.generate_documentation()

            # Step 3: Compare with existing (if enabled)
            if self.config.include_comparison:
                self.compare_with_existing(existing_docs_path)

            # Step 4: Generate report
            report = self.generate_report()
            report.execution_time = time.time() - start_time

            self.logger.info(f"Pipeline completed successfully in {report.execution_time:.2f} seconds")
            return report

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def _find_existing_docs(self, docs_path: Optional[str] = None) -> List[Tuple[str, str]]:
        """Find existing documentation files"""
        if docs_path:
            base_path = Path(docs_path)
        else:
            if hasattr(self, "analyzer") and hasattr(self.analyzer, "repo_path"):
                base_path = Path(self.analyzer.repo_path)
                self.logger.info(f"Using analyzer repo path for existing docs: {base_path}")
            else:
                base_path = Path(self.config.repo_path)
                self.logger.info(f"Using config repo path for existing docs: {base_path}")

        existing_docs = []

        # Common documentation file patterns
        doc_patterns = {
            "readme": ["README.md", "README.rst", "README.txt", "readme.md"],
            "api": ["API.md", "api.md", "docs/api.md", "docs/API.md"],
            "tutorial": ["TUTORIAL.md", "tutorial.md", "docs/tutorial.md", "GETTING_STARTED.md"],
            "architecture": ["ARCHITECTURE.md", "architecture.md", "docs/architecture.md"],
        }

        self.logger.info(f"Searching for existing documentation in: {base_path}")

        # Debug: List what files actually exist in the repository
        if base_path.exists():
            try:
                files = list(base_path.glob("*"))
                self.logger.info(f"Files in repository root: {[f.name for f in files[:10]]}")
                # Check specifically for README files
                readme_files = list(base_path.glob("README*"))
                self.logger.info(f"README files found: {[f.name for f in readme_files]}")
            except Exception as e:
                self.logger.warning(f"Could not list repository files: {e}")
        else:
            self.logger.warning(f"Repository path does not exist: {base_path}")

        for doc_type, patterns in doc_patterns.items():
            for pattern in patterns:
                file_path = base_path / pattern
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        existing_docs.append((doc_type, content))
                        self.logger.info(f"Found existing {doc_type} documentation: {file_path}")
                        break  # Only take the first match per type
                    except (UnicodeDecodeError, PermissionError) as e:
                        self.logger.warning(f"Could not read {file_path}: {str(e)}")

        if not existing_docs:
            self.logger.warning(f"No existing documentation found in {base_path}")

        return existing_docs

    def _save_analysis_results(self):
        """Save repository analysis results to file"""
        analysis_file = self.output_path / "analysis_results.json"

        # Convert dataclasses to dictionaries for JSON serialization
        analysis_data = {
            "repository_metadata": asdict(self.repository_metadata),
            "modules": [asdict(module) for module in self.modules],
        }

        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, default=str)

        self.logger.info(f"Analysis results saved to {analysis_file}")

    def _save_generated_documents(self):
        """Save generated documents to files"""
        docs_dir = self.output_path / "generated"
        docs_dir.mkdir(exist_ok=True)

        for doc in self.generated_docs:
            # Save document content
            doc_file = docs_dir / f"{doc.doc_type}_documentation.md"
            with open(doc_file, "w", encoding="utf-8") as f:
                f.write(doc.content)

            # Save document metadata
            metadata_file = docs_dir / f"{doc.doc_type}_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(asdict(doc), f, indent=2, default=str)

            self.logger.info(f"Saved {doc.doc_type} documentation to {doc_file}")

    def _save_comparison_results(self):
        """Save comparison results to file"""
        if not self.comparison_results:
            return

        comparison_file = self.output_path / "comparison_results.json"

        # Convert comparison results to serializable format
        serializable_results = {}
        for doc_type, result in self.comparison_results.items():
            serializable_results[doc_type] = asdict(result)

        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Also save human-readable report
        if self.comparator:
            report_file = self.output_path / "comparison_report.md"
            report_content = self.comparator.generate_comparison_report(self.comparison_results)
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)

        self.logger.info(f"Comparison results saved to {comparison_file}")

    def _save_report(self, report: DocumentationReport):
        """Save the final report"""
        report_file = self.output_path / "documentation_report.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Also create a human-readable summary
        summary_file = self.output_path / "report_summary.md"
        summary_content = self._create_report_summary(report)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_content)

        self.logger.info(f"Final report saved to {report_file}")

    def _create_report_summary(self, report: DocumentationReport) -> str:
        """Create a human-readable report summary"""
        summary = f"""# Documentation Generation Report

## Repository Analysis
- **Repository**: {report.repository_metadata.name}
- **Description**: {report.repository_metadata.description}
- **Language**: {report.repository_metadata.language}
- **Files Analyzed**: {report.repository_metadata.file_count}
- **Modules Analyzed**: {report.modules_analyzed}
- **Dependencies**: {', '.join(report.repository_metadata.dependencies)}

## Generation Results
- **Documents Generated**: {len(report.generated_documents)}
- **Success Rate**: {report.success_rate:.1%}
- **Execution Time**: {report.execution_time:.2f} seconds

### Generated Documents
"""

        for doc in report.generated_documents:
            summary += f"- **{doc.doc_type.title()}**: {doc.word_count} words\n"

        if report.comparison_results:
            summary += "\n## Comparison Results\n"
            for doc_type, result in report.comparison_results.items():
                metrics = result.metrics
                summary += f"\n### {doc_type.title()} Documentation\n"
                summary += f"- Semantic Similarity: {metrics.semantic_similarity:.3f}\n"
                summary += f"- Content Coverage: {metrics.content_coverage:.3f}\n"
                summary += f"- Structural Similarity: {metrics.structural_similarity:.3f}\n"

        if report.recommendations:
            summary += "\n## Recommendations\n"
            for i, rec in enumerate(report.recommendations, 1):
                summary += f"{i}. {rec}\n"

        return summary

    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on all analysis"""
        recommendations = []

        # Repository-level recommendations
        if self.repository_metadata:
            if self.repository_metadata.complexity_score > 10:
                recommendations.append(
                    "High code complexity detected. Consider refactoring complex functions and adding more detailed documentation."
                )

            if len(self.repository_metadata.dependencies) > 20:
                recommendations.append(
                    "Large number of dependencies detected. Consider documenting dependency management and version requirements."
                )

            if not self.repository_metadata.entry_points:
                recommendations.append(
                    "No clear entry points found. Consider adding main functions or improving code structure documentation."
                )

        # Documentation quality recommendations
        if self.generated_docs:
            # Check documentation word count as a quality indicator
            avg_word_count = sum(doc.word_count for doc in self.generated_docs) / len(self.generated_docs)

            if avg_word_count < 100:
                recommendations.append(
                    "Generated documentation appears brief. Consider reviewing and expanding the documentation for completeness."
                )

        # Comparison-based recommendations
        if self.comparison_results:
            for doc_type, result in self.comparison_results.items():
                if result.metrics.semantic_similarity < 0.6:
                    recommendations.append(
                        f"The generated {doc_type} documentation differs significantly from existing documentation. Manual review recommended."
                    )

        # General recommendations
        recommendations.append("Regularly update documentation as code evolves to maintain accuracy and relevance.")

        return recommendations

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a quick summary of the analysis results"""
        if not self.repository_metadata:
            return {"status": "not_analyzed"}

        return {
            "repository_name": self.repository_metadata.name,
            "language": self.repository_metadata.language,
            "file_count": self.repository_metadata.file_count,
            "modules_count": len(self.modules),
            "complexity_score": self.repository_metadata.complexity_score,
            "dependencies_count": len(self.repository_metadata.dependencies),
            "generated_docs_count": len(self.generated_docs),
            "has_comparisons": bool(self.comparison_results),
        }
