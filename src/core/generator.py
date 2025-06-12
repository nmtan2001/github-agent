"""
Documentation Generation Module

This module handles the generation of documentation using Large Language Models,
with support for different documentation types and templates.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from .analyzer import RepositoryMetadata, ModuleInfo, FunctionInfo, ClassInfo, CodeAnalyzer
from .document_reader import DocumentReader, DocumentChunk
from .summarizer import ContentSummarizer, SummarizationConfig
from ..utils.templates import TemplateManager
from ..utils.llm import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation"""

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    include_examples: bool = True
    include_diagrams: bool = True
    format_type: str = "markdown"
    documentation_types: List[str] = None

    def __post_init__(self):
        if self.documentation_types is None:
            self.documentation_types = ["comprehensive"]


@dataclass
class GeneratedDocument:
    """Container for generated documentation"""

    title: str
    content: str
    doc_type: str
    metadata: Dict[str, Any]
    word_count: int


class LLMDocumentationChain:
    """LangChain-based documentation generation pipeline"""

    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        self.knowledge_base = None
        self.graph = self._create_documentation_graph()

    def _create_documentation_graph(self) -> StateGraph:
        """Create a LangGraph state machine for documentation generation"""

        class DocumentationState(TypedDict):
            code_analysis: RepositoryMetadata
            modules: List[ModuleInfo]
            current_doc_type: str
            generated_sections: Dict[str, str]
            context_chunks: List[str]
            final_document: str

        def analyze_context(state: DocumentationState) -> DocumentationState:
            """Analyze code context and prepare for generation"""
            context_chunks = []
            repo_context = f"""
            Repository: {state['code_analysis'].name}
            Description: {state['code_analysis'].description}
            Language: {state['code_analysis'].language}
            Files: {state['code_analysis'].file_count}
            Dependencies: {', '.join(state['code_analysis'].dependencies)}
            """
            context_chunks.append(repo_context)

            for module in state["modules"]:
                module_context = f"""
                Module: {module.name}
                Path: {module.path}
                Functions: {len(module.functions)}
                Classes: {len(module.classes)}
                """
                if module.docstring:
                    module_context += f"Description: {module.docstring}\n"
                context_chunks.append(module_context)

            state["context_chunks"] = context_chunks
            return state

        def generate_section(state: DocumentationState) -> DocumentationState:
            """Generate a specific documentation section"""
            doc_type = state["current_doc_type"]
            try:
                section = self._generate_comprehensive_section(state)
                if "generated_sections" not in state:
                    state["generated_sections"] = {}
                state["generated_sections"][doc_type] = section
            except Exception as e:
                print(f"ERROR: Error generating {doc_type} section: {str(e)}")
                import traceback

                traceback.print_exc()
                if "generated_sections" not in state:
                    state["generated_sections"] = {}
                state["generated_sections"][
                    doc_type
                ] = f"# Error generating {doc_type} documentation\n\nError: {str(e)}"
            return state

        def compile_document(state: DocumentationState) -> DocumentationState:
            """Compile final documentation from all sections"""
            doc_type = state["current_doc_type"]
            if doc_type in state["generated_sections"]:
                state["final_document"] = state["generated_sections"][doc_type]
            else:
                if state["generated_sections"]:
                    state["final_document"] = list(state["generated_sections"].values())[0]
                else:
                    state["final_document"] = f"# {state['code_analysis'].name} Documentation\n\nNo content generated."
            return state

        workflow = StateGraph(DocumentationState)
        workflow.add_node("analyze_context", analyze_context)
        workflow.add_node("generate_section", generate_section)
        workflow.add_node("compile_document", compile_document)
        workflow.add_edge("analyze_context", "generate_section")
        workflow.add_edge("generate_section", "compile_document")
        workflow.add_edge("compile_document", END)
        workflow.set_entry_point("analyze_context")
        return workflow.compile()

    def _format_modules_for_prompt(self, modules: List[ModuleInfo]) -> str:
        """Format module information for LLM prompt"""
        formatted = []
        for module in modules:
            module_text = f"Module: {module.name}\n"
            if module.docstring:
                module_text += f"Description: {module.docstring}\n"
            if module.functions:
                module_text += "Functions:\n"
                for func in module.functions:
                    func_sig = f"  - {func.name}({', '.join(func.parameters)})"
                    if func.return_type:
                        func_sig += f" -> {func.return_type}"
                    module_text += func_sig + "\n"
                    if func.docstring:
                        module_text += f"    {func.docstring}\n"
            if module.classes:
                module_text += "Classes:\n"
                for cls in module.classes:
                    module_text += f"  - {cls.name}\n"
                    if cls.docstring:
                        module_text += f"    {cls.docstring}\n"
            formatted.append(module_text)
        return "\n\n".join(formatted)

    def _create_module_structure_text(self, modules: List[ModuleInfo]) -> str:
        """Create a text representation of module structure"""
        structure = []
        for module in modules:
            module_info = f"{module.path}:"
            module_info += f"\n  Functions: {len(module.functions)}"
            module_info += f"\n  Classes: {len(module.classes)}"
            module_info += f"\n  Imports: {len(module.imports)}"
            structure.append(module_info)
        return "\n\n".join(structure)

    def _generate_comprehensive_section(self, state) -> str:
        repo_name = state["code_analysis"].name
        description = state["code_analysis"].description
        language = state["code_analysis"].language
        dependencies = state["code_analysis"].dependencies
        modules = state["modules"]
        core_modules = [
            m for m in modules if not any(skip in m.path.lower() for skip in ["test", "example", "__pycache__"])
        ]
        module_details = []
        key_classes = []
        key_functions = []
        for module in core_modules[:15]:
            module_detail = f"\n## Module: {module.name}\n"
            module_detail += f"**Path**: `{module.path}`\n"
            if module.docstring:
                module_detail += f"**Description**: {module.docstring[:300]}...\n"
            if module.classes:
                module_detail += f"**Classes** ({len(module.classes)}):\n"
                for cls in module.classes[:5]:
                    key_classes.append(f"{module.name}.{cls.name}")
                    module_detail += (
                        f"- `{cls.name}`: {cls.docstring[:150] if cls.docstring else 'No description'}...\n"
                    )
                    if cls.methods:
                        module_detail += f"  - Methods: {', '.join([m.name for m in cls.methods[:8]])}\n"
            if module.functions:
                module_detail += f"**Functions** ({len(module.functions)}):\n"
                for func in module.functions[:8]:
                    key_functions.append(f"{module.name}.{func.name}")
                    params = ", ".join(func.parameters) if func.parameters else "no parameters"
                    return_type = f" -> {func.return_type}" if func.return_type else ""
                    module_detail += f"- `{func.name}({params}){return_type}`: {func.docstring[:150] if func.docstring else 'No description'}...\n"
            module_details.append(module_detail)
        prompt = PromptTemplate(
            input_variables=[
                "repo_name",
                "description",
                "language",
                "dependencies",
                "module_details",
                "key_classes",
                "key_functions",
            ],
            template="""
            Generate a comprehensive, unified documentation for the {repo_name} project that combines all aspects into a single, cohesive document.
            
            **Project Overview:**
            - Name: {repo_name}
            - Description: {description}
            - Language: {language}
            - Key Dependencies: {dependencies}
            
            **Module Analysis:**
            {module_details}
            
            **Key Components:**
            - Classes: {key_classes}
            - Functions: {key_functions}
            
            Create a SINGLE comprehensive documentation that includes ALL of the following sections in this exact order:
            
            # {repo_name}
            
            ## Table of Contents
            [Generate a complete table of contents for all sections below]
            
            ## Overview
            [Provide a comprehensive overview of the project, its purpose, and key features based on the code analysis]
            
            ## Installation
            [Detailed installation instructions including prerequisites, dependencies, and setup steps]
            
            ## Quick Start
            [A concise getting-started guide with the most basic usage example]
            
            ## Features
            [List and explain all major features discovered in the codebase]
            
            ## API Reference
            [Comprehensive API documentation covering all major modules, classes, and functions with:
            - Module descriptions
            - Class documentation with methods
            - Function signatures and descriptions
            - Parameter types and return values
            - Usage examples for each major component]
            
            ## Usage Examples
            [Multiple practical examples showing different use cases and scenarios]
            
            ## Architecture
            [Technical architecture overview including:
            - System design and structure
            - Component relationships
            - Data flow
            - Design patterns used
            - Technology choices and rationale]
            
            ## Configuration
            [Any configuration options, environment variables, or settings]
            
            ## Advanced Usage
            [Advanced features, customization options, and power-user functionality]
            
            ## Troubleshooting
            [Common issues, solutions, and debugging tips]
            
            ## Performance Considerations
            [Performance tips, optimization strategies, and scalability notes]
            
            ## Security
            [Security considerations, best practices, and any security-related features]
            
            ## Contributing
            [Guidelines for contributing to the project]
            
            ## License
            [License information]
            
            ## Appendix
            [Additional technical details, glossary, or reference materials]
            
            **Requirements:**
            - Make this a SINGLE, unified document - not separate files
            - Use actual module names, classes, and functions from the code analysis
            - Include realistic, runnable code examples based on the actual codebase
            - Ensure all sections flow together cohesively
            - Use proper markdown formatting with appropriate headers and code blocks
            - Make it comprehensive enough to serve as the primary documentation
            - Include both high-level concepts and detailed technical information
            - Ensure examples use the actual API discovered in the code
            
            Generate the complete, production-ready comprehensive documentation:
            """,
        )
        formatted_prompt = prompt.format(
            repo_name=repo_name,
            description=description[:500] + "..." if len(description) > 500 else description,
            language=language,
            dependencies=", ".join(dependencies[:15]),
            module_details="\n".join(module_details),
            key_classes=", ".join(key_classes[:30]),
            key_functions=", ".join(key_functions[:40]),
        )
        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive, unified documentation that combines all aspects into a single cohesive document."
            ),
            HumanMessage(content=formatted_prompt),
        ]
        result = self.llm.invoke(messages)
        return result.content if hasattr(result, "content") else str(result)


class DocumentationGenerator:
    """Main documentation generator using LLM chains"""

    def __init__(self, config: Optional[DocumentationConfig] = None):
        self.config = config or DocumentationConfig()
        self.llm_chain = LLMDocumentationChain(self.config)
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def generate_documentation(
        self, repo_metadata: RepositoryMetadata, modules: List[ModuleInfo], doc_types: Optional[List[str]] = None
    ) -> List[GeneratedDocument]:
        """Generate documentation for the given repository"""
        if doc_types is None:
            doc_types = self.config.documentation_types
        generated_docs = []
        for doc_type in doc_types:
            print(f"Starting generation of {doc_type} documentation...")
            try:
                state = {
                    "code_analysis": repo_metadata,
                    "modules": modules,
                    "current_doc_type": doc_type,
                    "generated_sections": {},
                    "context_chunks": [],
                    "final_document": "",
                }
                result = self.llm_chain.graph.invoke(state)
                doc = GeneratedDocument(
                    title=f"{repo_metadata.name} - {doc_type.title()} Documentation",
                    content=result["final_document"],
                    doc_type=doc_type,
                    metadata={
                        "repository": repo_metadata.name,
                        "language": repo_metadata.language,
                        "generation_model": self.config.model_name,
                        "modules_count": len(modules),
                    },
                    word_count=len(result["final_document"].split()),
                )
                generated_docs.append(doc)
                print(f"Successfully generated {doc_type} documentation ({doc.word_count} words)")
            except Exception as e:
                print(f"Error generating {doc_type} documentation: {str(e)}")
                print(f"Full error details:")
                import traceback

                traceback.print_exc()
                continue
        print(f"Completed generation of {len(generated_docs)}/{len(doc_types)} documentation types")
        return generated_docs


@dataclass
class DocumentationContext:
    """Context information for documentation generation."""

    existing_docs: List[DocumentChunk]
    code_analysis: Dict[str, Any]
    similar_content: List[Dict[str, Any]]
    doc_summary: Dict[str, Any]


class EnhancedDocumentationGenerator:
    """
    Enhanced documentation generator that reads and incorporates existing
    documentation during the generation process.
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        template_manager: Optional[TemplateManager] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        summarization_config: Optional[SummarizationConfig] = None,
    ):
        """Initialize the enhanced documentation generator."""
        self.llm_manager = llm_manager or LLMManager()
        self.template_manager = template_manager or TemplateManager()
        self.document_reader = DocumentReader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.summarizer = ContentSummarizer(summarization_config)
        self.vector_index = None
        self.existing_docs = []

    def load_existing_documentation(self, docs_paths: List[str], exclude_patterns: Optional[List[str]] = None) -> bool:
        """
        Load existing documentation from specified paths.
        """
        try:
            self.existing_docs = []
            exclude_patterns = exclude_patterns or ["*.git/*", "*.node_modules/*", "*.venv/*", "*/__pycache__/*"]
            for docs_path in docs_paths:
                if not os.path.exists(docs_path):
                    logger.warning(f"Documentation path not found: {docs_path}")
                    continue
                logger.info(f"Loading documentation from: {docs_path}")
                if os.path.isfile(docs_path):
                    chunks = self.document_reader.read_with_llamaindex(
                        str(Path(docs_path).parent), recursive=False, exclude_patterns=exclude_patterns
                    )
                else:
                    chunks_langchain = self.document_reader.read_documentation_directory(
                        docs_path, recursive=True, exclude_patterns=exclude_patterns
                    )
                    chunks_llamaindex = self.document_reader.read_with_llamaindex(
                        docs_path, recursive=True, exclude_patterns=exclude_patterns
                    )
                    chunks = self._deduplicate_chunks(chunks_langchain + chunks_llamaindex)
                self.existing_docs.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {docs_path}")
            if self.existing_docs:
                self.vector_index = self.document_reader.create_vector_index(self.existing_docs)
                summary = self.document_reader.get_documentation_summary(self.existing_docs)
                logger.info(f"Documentation loaded: {summary}")
                return True
            else:
                logger.warning("No documentation content was loaded")
                return False
        except Exception as e:
            logger.error(f"Error loading existing documentation: {e}")
            return False

    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove duplicate chunks based on content and source."""
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            identifier = (hash(chunk.content), chunk.source, chunk.chunk_index)
            if identifier not in seen:
                seen.add(identifier)
                unique_chunks.append(chunk)
        logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
        return unique_chunks

    def generate_documentation_with_context(
        self,
        repo_path: str,
        output_format: str = "markdown",
        context_similarity_threshold: float = 0.7,
    ) -> List[GeneratedDocument]:
        """
        Generate comprehensive documentation using both code analysis and existing documentation context.
        """
        try:
            logger.info(f"Generating comprehensive documentation for: {repo_path}")
            logger.info("Analyzing code structure...")
            code_analyzer = CodeAnalyzer(repo_path)
            repository_metadata, modules = code_analyzer.analyze_repository()
            code_analysis = {
                "name": repository_metadata.name,
                "description": repository_metadata.description,
                "language": repository_metadata.language,
                "primary_language": repository_metadata.language,
                "size": repository_metadata.size,
                "file_count": repository_metadata.file_count,
                "dependencies": repository_metadata.dependencies,
                "entry_points": repository_metadata.entry_points,
                "test_coverage": repository_metadata.test_coverage,
                "complexity_score": repository_metadata.complexity_score,
                "modules": [asdict(module) for module in modules],
            }
            context = self._prepare_documentation_context(code_analysis, context_similarity_threshold)

            doc_type = "comprehensive"
            logger.info(f"Generating {doc_type} documentation with context...")

            section_content = self._generate_section_with_context(doc_type, context, output_format)

            doc = GeneratedDocument(
                title=f"{repository_metadata.name} - {doc_type.title()} Documentation",
                content=section_content,
                doc_type=doc_type,
                metadata={
                    "repository": repository_metadata.name,
                    "language": repository_metadata.language,
                    "generation_model": self.llm_manager.config.model,
                    "modules_count": len(modules),
                    "existing_docs_count": len(context.existing_docs),
                    "enhanced_with_embeddings": True,
                    "context_used": bool(context.existing_docs),
                },
                word_count=len(section_content.split()),
            )

            logger.info(f"Generated {doc_type} documentation with embedding enhancement ({doc.word_count} words)")
            return [doc]
        except Exception as e:
            logger.error(f"Error generating documentation with context: {e}")
            raise

    def _prepare_documentation_context(
        self, code_analysis: Dict[str, Any], similarity_threshold: float
    ) -> DocumentationContext:
        """Prepare comprehensive context for documentation generation."""
        similar_content = []
        if self.vector_index and self.existing_docs:
            search_queries = [
                f"installation guide for {code_analysis.get('name', 'project')}",
                f"API reference {code_analysis.get('primary_language', '')}",
                f"usage examples {code_analysis.get('name', '')}",
                "getting started tutorial",
                "contributing guidelines",
            ]
            for query in search_queries:
                results = self.document_reader.search_similar_content(query, self.vector_index, top_k=3)
                filtered_results = [r for r in results if r.get("score", 0) >= similarity_threshold]
                similar_content.extend(filtered_results)
        unique_similar = []
        seen_content = set()
        for item in similar_content:
            content_hash = hash(item["content"])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_similar.append(item)
        doc_summary = self.document_reader.get_documentation_summary(self.existing_docs)
        return DocumentationContext(
            existing_docs=self.existing_docs,
            code_analysis=code_analysis,
            similar_content=unique_similar,
            doc_summary=doc_summary,
        )

    def _generate_section_with_context(self, doc_type: str, context: DocumentationContext, output_format: str) -> str:
        """Generate a documentation section using context."""
        relevant_content = self._get_relevant_content_for_section(doc_type, context)
        prompt = self._build_context_aware_prompt(doc_type, context.code_analysis, relevant_content, output_format)
        print(f"-----PROMPT FOR {doc_type}-----")
        print(prompt)
        print("--------------------------")
        try:
            llm = ChatOpenAI(
                model=self.llm_manager.config.model,
                temperature=0.3,
                max_tokens=None,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            prompt_template = ChatPromptTemplate.from_template("{prompt}")
            chain = prompt_template | llm | StrOutputParser()
            return chain.invoke({"prompt": prompt})
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            return f"Error generating documentation: {e}"

    def _get_relevant_content_for_section(self, doc_type: str, context: DocumentationContext) -> List[Dict[str, Any]]:
        """Get existing content relevant to a specific documentation section."""
        return context.similar_content

    def _build_context_aware_prompt(
        self, doc_type: str, code_analysis: Dict[str, Any], relevant_content: List[Dict[str, Any]], output_format: str
    ) -> str:
        """Build a prompt that incorporates existing documentation context."""
        project_info = f"""
Project Information:
- Name: {code_analysis.get('name', 'Unknown')}
- Language: {code_analysis.get('primary_language', 'Unknown')}
- Description: {code_analysis.get('description', 'No description available')}
- Dependencies: {', '.join(code_analysis.get('dependencies', [])[:5])}
- File Count: {code_analysis.get('file_count', 0)}
- Entry Points: {', '.join(code_analysis.get('entry_points', [])[:3])}
"""
        modules_summary = ""
        if code_analysis.get("modules"):
            modules = code_analysis["modules"][:10]
            modules_summary = "\nKey Modules:\n"
            for module in modules:
                functions_count = len(module.get("functions", []))
                classes_count = len(module.get("classes", []))
                modules_summary += (
                    f"- {module.get('name', 'Unknown')}: {functions_count} functions, {classes_count} classes\n"
                )
        existing_context = ""
        if relevant_content:
            existing_context = "\nExisting Documentation Context:\n"
            for i, content in enumerate(relevant_content, 1):
                source = content["metadata"].get("file_name", "unknown file")
                if self.summarizer.should_summarize(content["content"]):
                    content_summary = self.summarizer.summarize_existing_documentation(content["content"])
                    existing_context += f"\n{i}. From {source} (summarized):\n{content_summary}\n"
                else:
                    existing_context += f"\n{i}. From {source}:\n{content['content']}\n"
        prompt = f"""
You are a technical documentation expert. Generate comprehensive {doc_type} documentation 
for the following project, incorporating insights from existing documentation where relevant.

{project_info}
{modules_summary}
{existing_context}

Instructions:
1. Create {doc_type} documentation in {output_format} format
2. Use information from the code analysis as the primary source
3. Incorporate relevant insights from existing documentation context when appropriate
4. Ensure the documentation is comprehensive, accurate, and well-structured
5. If existing documentation provides good examples or explanations, adapt them appropriately
6. Maintain consistency with the existing documentation style where possible

Generate the {doc_type} documentation:
"""
        return prompt
