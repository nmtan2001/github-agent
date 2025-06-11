"""
Documentation Generation Module

This module handles the generation of documentation using Large Language Models,
with support for different documentation types and templates.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.schema import BaseOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import Graph, StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from .analyzer import RepositoryMetadata, ModuleInfo, FunctionInfo, ClassInfo


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation"""

    model_name: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 2000
    include_examples: bool = True
    include_diagrams: bool = True
    format_type: str = "markdown"  # markdown, rst, html
    documentation_types: List[str] = None

    def __post_init__(self):
        if self.documentation_types is None:
            self.documentation_types = ["api", "readme", "tutorial", "architecture"]


@dataclass
class GeneratedDocument:
    """Container for generated documentation"""

    title: str
    content: str
    doc_type: str
    metadata: Dict[str, Any]
    confidence_score: float
    word_count: int


class DocumentationTemplates:
    """Collection of documentation templates"""

    @staticmethod
    def get_api_template() -> str:
        return """
# API Documentation for {module_name}

## Overview
{module_description}

## Classes

{classes_section}

## Functions

{functions_section}

## Constants

{constants_section}

## Usage Examples

{examples_section}
"""

    @staticmethod
    def get_readme_template() -> str:
        return """
# {project_name}

{project_description}

## Features

{features_section}

## Installation

{installation_section}

## Quick Start

{quickstart_section}

## API Reference

{api_reference_section}

## Contributing

{contributing_section}

## License

{license_section}
"""

    @staticmethod
    def get_tutorial_template() -> str:
        return """
# {project_name} Tutorial

## Introduction

{introduction}

## Prerequisites

{prerequisites}

## Step-by-Step Guide

{tutorial_steps}

## Advanced Usage

{advanced_usage}

## Troubleshooting

{troubleshooting}

## Next Steps

{next_steps}
"""

    @staticmethod
    def get_architecture_template() -> str:
        return """
# Architecture Documentation

## System Overview

{system_overview}

## Component Architecture

{component_architecture}

## Data Flow

{data_flow}

## Design Patterns

{design_patterns}

## Dependencies

{dependencies}

## Performance Considerations

{performance_considerations}
"""


class LLMDocumentationChain:
    """LangChain-based documentation generation pipeline"""

    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.llm = ChatOpenAI(model=config.model_name, temperature=config.temperature, max_tokens=config.max_tokens)

        # Initialize embeddings for context retrieval
        self.embeddings = OpenAIEmbeddings()
        self.knowledge_base = None

        # Create documentation generation graph
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
            # Create context chunks from code analysis
            context_chunks = []

            # Add repository overview
            repo_context = f"""
            Repository: {state['code_analysis'].name}
            Description: {state['code_analysis'].description}
            Language: {state['code_analysis'].language}
            Files: {state['code_analysis'].file_count}
            Dependencies: {', '.join(state['code_analysis'].dependencies)}
            """
            context_chunks.append(repo_context)

            # Add module information
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

            print(f"🔍 DEBUG: Generating {doc_type} section")
            print(f"🔍 DEBUG: Repository: {state['code_analysis'].name}")
            print(f"🔍 DEBUG: Modules count: {len(state['modules'])}")

            try:
                if doc_type == "api":
                    section = self._generate_api_section(state)
                elif doc_type == "readme":
                    section = self._generate_readme_section(state)
                elif doc_type == "tutorial":
                    section = self._generate_tutorial_section(state)
                elif doc_type == "architecture":
                    section = self._generate_architecture_section(state)
                else:
                    section = self._generate_generic_section(state)

                if "generated_sections" not in state:
                    state["generated_sections"] = {}
                state["generated_sections"][doc_type] = section

                print(f"✅ DEBUG: Successfully generated {doc_type} section ({len(section)} chars)")

            except Exception as e:
                print(f"❌ DEBUG: Error generating {doc_type} section: {str(e)}")
                import traceback

                traceback.print_exc()

                # Set a fallback content
                if "generated_sections" not in state:
                    state["generated_sections"] = {}
                state["generated_sections"][
                    doc_type
                ] = f"# Error generating {doc_type} documentation\n\nError: {str(e)}"

            return state

        def compile_document(state: DocumentationState) -> DocumentationState:
            """Compile final documentation from all sections"""
            doc_type = state["current_doc_type"]

            # Debug logging
            print(f"🔍 DEBUG: Compiling {doc_type} document")
            print(f"🔍 DEBUG: Generated sections keys: {list(state['generated_sections'].keys())}")

            # For now, just return the generated content directly without template formatting
            # since the LLM is already generating complete documentation
            if doc_type in state["generated_sections"]:
                state["final_document"] = state["generated_sections"][doc_type]
            else:
                # Fallback: use any available section
                if state["generated_sections"]:
                    state["final_document"] = list(state["generated_sections"].values())[0]
                else:
                    state["final_document"] = f"# {state['code_analysis'].name} Documentation\n\nNo content generated."

            print(f"🔍 DEBUG: Final document length: {len(state['final_document'])} characters")

            return state

        # Create the graph
        workflow = StateGraph(DocumentationState)

        # Add nodes
        workflow.add_node("analyze_context", analyze_context)
        workflow.add_node("generate_section", generate_section)
        workflow.add_node("compile_document", compile_document)

        # Add edges
        workflow.add_edge("analyze_context", "generate_section")
        workflow.add_edge("generate_section", "compile_document")
        workflow.add_edge("compile_document", END)

        # Set entry point
        workflow.set_entry_point("analyze_context")

        return workflow.compile()

    def _generate_api_section(self, state) -> str:
        """Generate API documentation section"""

        # Group modules by functionality
        modules = state["modules"]
        core_modules = [
            m for m in modules if not any(skip in m.path.lower() for skip in ["test", "example", "__pycache__"])
        ]

        prompt = PromptTemplate(
            input_variables=["repo_name", "core_modules", "total_modules", "key_classes", "key_functions"],
            template="""
            Generate comprehensive API documentation for {repo_name}:
            
            **Repository**: {repo_name}
            **Core Modules Analyzed**: {total_modules} modules
            **Key Classes Identified**: {key_classes}
            **Key Functions Identified**: {key_functions}
            
            **Module Details**:
            {core_modules}
            
            Create professional API documentation with:
            
            1. **API Overview**: Brief introduction to the API structure
            2. **Core Modules**: Detailed documentation for each major module
            3. **Classes**: Complete class documentation with methods and attributes
            4. **Functions**: Detailed function documentation with parameters and returns
            5. **Examples**: Practical usage examples for each major component
            6. **Error Handling**: Common errors and how to handle them
            7. **Authentication**: If applicable, authentication methods
            8. **Response Formats**: Expected response structures
            
            **Requirements**:
            - Use actual class and function names from the code analysis
            - Include parameter types and return types where available
            - Provide realistic code examples
            - Follow Python documentation standards (docstring format)
            - Include import statements in examples
            - Group related functionality together
            - Use proper markdown formatting with syntax highlighting
            
            Generate complete, developer-friendly API documentation.
            """,
        )

        # Use new LangChain syntax: prompt | llm
        chain = prompt | self.llm

        # Build detailed module information
        module_details = []
        key_classes = []
        key_functions = []

        for module in core_modules[:10]:  # Limit to top 10 modules to avoid token limits
            module_detail = f"\n## Module: {module.name}\n"
            module_detail += f"**Path**: `{module.path}`\n"

            if module.docstring:
                module_detail += f"**Description**: {module.docstring[:200]}...\n"

            if module.classes:
                module_detail += f"**Classes** ({len(module.classes)}):\n"
                for cls in module.classes[:3]:  # Top 3 classes per module
                    key_classes.append(f"{module.name}.{cls.name}")
                    module_detail += (
                        f"- `{cls.name}`: {cls.docstring[:100] if cls.docstring else 'No description'}...\n"
                    )
                    if cls.methods:
                        module_detail += f"  - Methods: {', '.join([m.name for m in cls.methods[:5]])}\n"

            if module.functions:
                module_detail += f"**Functions** ({len(module.functions)}):\n"
                for func in module.functions[:5]:  # Top 5 functions per module
                    key_functions.append(f"{module.name}.{func.name}")
                    params = ", ".join(func.parameters) if func.parameters else "no parameters"
                    return_type = f" -> {func.return_type}" if func.return_type else ""
                    module_detail += f"- `{func.name}({params}){return_type}`: {func.docstring[:100] if func.docstring else 'No description'}...\n"

            module_details.append(module_detail)

        result = chain.invoke(
            {
                "repo_name": state["code_analysis"].name,
                "total_modules": len(core_modules),
                "core_modules": "\n".join(module_details),
                "key_classes": ", ".join(key_classes[:20]),  # Limit to avoid token overflow
                "key_functions": ", ".join(key_functions[:30]),
            }
        )
        return result.content if hasattr(result, "content") else str(result)

    def _generate_readme_section(self, state) -> str:
        """Generate README documentation section"""

        # Extract key information from the repository
        repo_name = state["code_analysis"].name
        description = state["code_analysis"].description
        main_modules = [m for m in state["modules"] if "main" in m.name.lower() or "cli" in m.name.lower()]
        example_modules = [m for m in state["modules"] if "example" in m.path.lower()]
        server_modules = [m for m in state["modules"] if "server" in m.name.lower()]
        client_modules = [m for m in state["modules"] if "client" in m.name.lower()]

        prompt = PromptTemplate(
            input_variables=[
                "repo_name",
                "description",
                "language",
                "dependencies",
                "main_modules",
                "example_modules",
                "module_count",
                "key_features",
            ],
            template="""
            Generate a comprehensive README.md for the {repo_name} project based on this analysis:
            
            **Project**: {repo_name}
            **Description**: {description}
            **Language**: {language}
            **Module Count**: {module_count}
            **Key Dependencies**: {dependencies}
            
            **Main Entry Points**: {main_modules}
            **Example Modules**: {example_modules}
            **Key Features Detected**: {key_features}
            
            Create a professional README that includes:
            
            1. **Project Title & Description**: Clear, engaging description
            2. **Table of Contents**: Well-organized navigation
            3. **Installation**: Step-by-step installation instructions
            4. **Quick Start**: Simple examples to get users started immediately
            5. **Key Features**: Highlight the main capabilities based on code analysis
            6. **Usage Examples**: Practical code examples from the actual modules
            7. **API Overview**: Brief overview of main components
            8. **Development**: How to contribute and develop
            9. **License**: Standard license section
            
            **Requirements**:
            - Be specific to THIS project, not generic
            - Use actual module names and functions found in the code
            - Include realistic examples based on the detected functionality
            - Match the technical level of a serious SDK project
            - Use proper markdown formatting with code blocks
            - Include badges and professional formatting
            
            Generate complete, production-ready documentation.
            """,
        )

        # Use new LangChain syntax: prompt | llm
        chain = prompt | self.llm

        # Extract meaningful information
        main_modules_info = []
        for module in main_modules[:3]:
            if module.functions:
                main_modules_info.append(f"{module.name} (functions: {[f.name for f in module.functions[:3]]})")

        example_modules_info = []
        for module in example_modules[:5]:
            example_modules_info.append(f"{module.name} ({module.path})")

        # Detect key features from dependencies and modules
        deps = state["code_analysis"].dependencies
        key_features = []
        if any("async" in d.lower() for d in deps):
            key_features.append("Asynchronous programming support")
        if any("http" in d.lower() for d in deps):
            key_features.append("HTTP client/server capabilities")
        if any("cli" in d.lower() or "click" in d.lower() or "typer" in d.lower() for d in deps):
            key_features.append("Command-line interface")
        if any("test" in d.lower() or "pytest" in d.lower() for d in deps):
            key_features.append("Comprehensive testing framework")
        if any("pydantic" in d.lower() for d in deps):
            key_features.append("Data validation and serialization")
        if server_modules:
            key_features.append("Server implementation")
        if client_modules:
            key_features.append("Client implementation")
        if example_modules:
            key_features.append("Rich examples and tutorials")

        result = chain.invoke(
            {
                "repo_name": repo_name,
                "description": description[:200] + "..." if len(description) > 200 else description,
                "language": state["code_analysis"].language,
                "module_count": len(state["modules"]),
                "dependencies": ", ".join(deps[:10]),
                "main_modules": ", ".join(main_modules_info) if main_modules_info else "No main modules detected",
                "example_modules": ", ".join(example_modules_info) if example_modules_info else "No examples detected",
                "key_features": ", ".join(key_features) if key_features else "General Python SDK",
            }
        )
        return result.content if hasattr(result, "content") else str(result)

    def _generate_tutorial_section(self, state) -> str:
        """Generate tutorial documentation section"""
        prompt = PromptTemplate(
            input_variables=["project_context", "entry_points"],
            template="""
            Create a step-by-step tutorial for this project:
            
            Project Context: {project_context}
            
            Entry Points: {entry_points}
            
            Please include:
            1. Clear learning objectives
            2. Prerequisites and setup
            3. Step-by-step instructions with code examples
            4. Expected outputs and results
            5. Common troubleshooting tips
            6. Next steps for advanced usage
            
            Make it beginner-friendly but comprehensive.
            """,
        )

        # Use new LangChain syntax: prompt | llm
        chain = prompt | self.llm

        project_context = f"""
        Project: {state['code_analysis'].name}
        Purpose: {state['code_analysis'].description}
        Complexity: {state['code_analysis'].complexity_score:.2f}
        """

        entry_points = ", ".join(state["code_analysis"].entry_points)

        result = chain.invoke({"project_context": project_context, "entry_points": entry_points})
        return result.content if hasattr(result, "content") else str(result)

    def _generate_architecture_section(self, state) -> str:
        """Generate architecture documentation section"""
        prompt = PromptTemplate(
            input_variables=["system_info", "dependencies", "modules_structure"],
            template="""
            Document the architecture of this system:
            
            System Information: {system_info}
            
            Dependencies: {dependencies}
            
            Module Structure: {modules_structure}
            
            Please include:
            1. High-level system overview
            2. Component relationships and interactions
            3. Data flow patterns
            4. Design decisions and rationale
            5. Scalability and performance considerations
            6. Future extensibility points
            
            Focus on technical depth and clarity.
            """,
        )

        # Use new LangChain syntax: prompt | llm
        chain = prompt | self.llm

        system_info = f"""
        Name: {state['code_analysis'].name}
        Size: {state['code_analysis'].size} bytes
        Files: {state['code_analysis'].file_count}
        Average Complexity: {state['code_analysis'].complexity_score:.2f}
        """

        dependencies = ", ".join(state["code_analysis"].dependencies)
        modules_structure = self._create_module_structure_text(state["modules"])

        result = chain.invoke(
            {"system_info": system_info, "dependencies": dependencies, "modules_structure": modules_structure}
        )
        return result.content if hasattr(result, "content") else str(result)

    def _generate_generic_section(self, state) -> str:
        """Generate generic documentation section"""
        prompt = PromptTemplate(
            input_variables=["code_context"],
            template="""
            Generate comprehensive documentation for this code:
            
            {code_context}
            
            Please provide clear, well-structured documentation that explains:
            1. Purpose and functionality
            2. How to use it
            3. Important details and considerations
            4. Examples where appropriate
            """,
        )

        # Use new LangChain syntax: prompt | llm
        chain = prompt | self.llm
        context = "\n".join(state["context_chunks"])

        result = chain.invoke({"code_context": context})
        return result.content if hasattr(result, "content") else str(result)

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

    def _get_template(self, doc_type: str) -> str:
        """Get documentation template by type"""
        templates = {
            "api": DocumentationTemplates.get_api_template(),
            "readme": DocumentationTemplates.get_readme_template(),
            "tutorial": DocumentationTemplates.get_tutorial_template(),
            "architecture": DocumentationTemplates.get_architecture_template(),
        }
        return templates.get(doc_type, "{content}")


class DocumentationGenerator:
    """Main documentation generator using LLM chains"""

    def __init__(self, config: Optional[DocumentationConfig] = None):
        self.config = config or DocumentationConfig()
        self.llm_chain = LLMDocumentationChain(self.config)

        # Validate OpenAI API key
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
            print(f"🔄 Starting generation of {doc_type} documentation...")
            try:
                # Create initial state
                state = {
                    "code_analysis": repo_metadata,
                    "modules": modules,
                    "current_doc_type": doc_type,
                    "generated_sections": {},
                    "context_chunks": [],
                    "final_document": "",
                }

                print(f"🔍 DEBUG: Initial state created for {doc_type}")

                # Run the documentation generation graph
                result = self.llm_chain.graph.invoke(state)

                print(f"🔍 DEBUG: Graph execution completed for {doc_type}")
                print(f"🔍 DEBUG: Result keys: {list(result.keys())}")

                # Create GeneratedDocument
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
                    confidence_score=0.85,  # Could be calculated based on model certainty
                    word_count=len(result["final_document"].split()),
                )

                generated_docs.append(doc)
                print(f"✅ Successfully generated {doc_type} documentation ({doc.word_count} words)")

            except Exception as e:
                print(f"❌ Error generating {doc_type} documentation: {str(e)}")
                print(f"🔍 DEBUG: Full error details:")
                import traceback

                traceback.print_exc()
                continue

        print(f"🎉 Completed generation of {len(generated_docs)}/{len(doc_types)} documentation types")
        return generated_docs

    def generate_single_module_doc(self, module: ModuleInfo) -> GeneratedDocument:
        """Generate documentation for a single module"""

        prompt = PromptTemplate(
            input_variables=["module_name", "module_content", "functions", "classes"],
            template="""
            Generate comprehensive documentation for this Python module:
            
            Module: {module_name}
            Content Overview: {module_content}
            
            Functions:
            {functions}
            
            Classes:
            {classes}
            
            Please provide:
            1. Module overview and purpose
            2. Detailed function documentation with parameters and return values
            3. Class documentation with methods and attributes
            4. Usage examples
            5. Integration notes
            
            Format as clean markdown.
            """,
        )

        chain = LLMChain(llm=self.llm_chain.llm, prompt=prompt)

        # Format functions
        functions_text = ""
        for func in module.functions:
            func_text = f"- {func.name}({', '.join(func.parameters)})"
            if func.return_type:
                func_text += f" -> {func.return_type}"
            if func.docstring:
                func_text += f"\n  {func.docstring}"
            functions_text += func_text + "\n"

        # Format classes
        classes_text = ""
        for cls in module.classes:
            cls_text = f"- {cls.name}"
            if cls.inheritance:
                cls_text += f"({', '.join(cls.inheritance)})"
            if cls.docstring:
                cls_text += f"\n  {cls.docstring}"
            classes_text += cls_text + "\n"

        result = chain.run(
            module_name=module.name,
            module_content=module.docstring or "No description available",
            functions=functions_text,
            classes=classes_text,
        )

        return GeneratedDocument(
            title=f"Module: {module.name}",
            content=result,
            doc_type="module",
            metadata={"module_path": module.path},
            confidence_score=0.8,
            word_count=len(result.split()),
        )

    def generate_function_doc(self, function: FunctionInfo) -> str:
        """Generate documentation for a single function"""

        prompt = PromptTemplate(
            input_variables=["func_name", "parameters", "docstring", "complexity"],
            template="""
            Generate documentation for this function:
            
            Function: {func_name}
            Parameters: {parameters}
            Existing Docstring: {docstring}
            Complexity: {complexity}
            
            Provide:
            1. Clear description of purpose
            2. Parameter explanations with types
            3. Return value description
            4. Usage example
            5. Any important notes or warnings
            
            Be concise but complete.
            """,
        )

        chain = LLMChain(llm=self.llm_chain.llm, prompt=prompt)

        result = chain.run(
            func_name=function.name,
            parameters=", ".join(function.parameters),
            docstring=function.docstring or "No docstring available",
            complexity=function.complexity,
        )

        return result

    def enhance_existing_documentation(self, existing_doc: str, code_context: str) -> str:
        """Enhance existing documentation with additional context"""

        prompt = PromptTemplate(
            input_variables=["existing_doc", "code_context"],
            template="""
            Enhance this existing documentation with additional insights from the code:
            
            Existing Documentation:
            {existing_doc}
            
            Code Context:
            {code_context}
            
            Please:
            1. Keep all valuable existing information
            2. Add missing technical details from the code
            3. Improve clarity and structure
            4. Add practical examples if missing
            5. Ensure consistency and completeness
            
            Return the enhanced documentation.
            """,
        )

        chain = LLMChain(llm=self.llm_chain.llm, prompt=prompt)

        result = chain.run(existing_doc=existing_doc, code_context=code_context)

        return result
