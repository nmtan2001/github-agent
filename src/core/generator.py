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
from langchain_core.prompts import PromptTemplate
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
    max_tokens: Optional[int] = None  # No token limit for comprehensive documentation
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
        # Updated to use latest OpenAI syntax with no token limit for comprehensive docs
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,  # No limit for detailed documentation
            api_key=os.getenv("OPENAI_API_KEY"),  # Explicit API key handling
        )

        # Initialize embeddings for context retrieval with explicit API key
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
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

            # Debug info for section generation
            pass

            try:
                if doc_type == "api":
                    section = self._generate_api_section(state)
                elif doc_type == "readme":
                    section = self._generate_readme_section(state)
                elif doc_type == "tutorial":
                    section = self._generate_tutorial_section(state)
                elif doc_type == "architecture":
                    section = self._generate_architecture_section(state)
                elif doc_type == "comprehensive":
                    section = self._generate_comprehensive_section(state)
                else:
                    section = self._generate_generic_section(state)

                if "generated_sections" not in state:
                    state["generated_sections"] = {}
                state["generated_sections"][doc_type] = section

                # Section generated successfully
                pass

            except Exception as e:
                # Log error without emojis to avoid encoding issues
                print(f"ERROR: Error generating {doc_type} section: {str(e)}")
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
            # Compiling document sections
            pass

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

            # Document compilation complete

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
            
            Create professional API documentation with the following structure:
            
            # {repo_name} API Documentation
            
            ## API Overview
            Brief introduction to the API structure and purpose.
            
            ## Core Modules
            
            For each module, provide:
            - Module purpose and overview
            - Key classes with detailed method documentation
            - Functions with parameters, return types, and usage examples
            - Implementation details and design patterns
            
            ## Classes
            
            For each class, include:
            - Class purpose and inheritance
            - Constructor parameters
            - Public methods with full signatures
            - Usage examples with imports
            - Error handling patterns
            
            ## Functions
            
            For each function, provide:
            - Function purpose and behavior
            - Parameter descriptions with types
            - Return value documentation
            - Example usage with realistic data
            - Exception handling
            
            ## Examples
            
            Provide comprehensive examples showing:
            - Basic usage patterns
            - Advanced scenarios
            - Integration examples
            - Best practices
            
            ## Error Handling
            
            Document:
            - Common exceptions and their causes
            - Error handling strategies
            - Debugging tips
            
            ## Authentication
            
            If applicable, document:
            - Authentication methods
            - API keys and tokens
            - Security considerations
            
            ## Response Formats
            
            Document expected:
            - Response structures
            - Data formats
            - Status codes
            
            **Requirements**:
            - Use actual class and function names from the code analysis
            - Include parameter types and return types where available
            - Provide realistic, runnable code examples
            - Follow Python documentation standards (PEP 257)
            - Include proper import statements in all examples
            - Group related functionality together logically
            - Use proper markdown formatting with syntax highlighting
            - Make it comprehensive and production-ready
            - Focus on developer experience and usability
            
            Generate complete, professional, developer-friendly API documentation that serves as a comprehensive reference.
            """,
        )

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

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(
            repo_name=state["code_analysis"].name,
            total_modules=len(core_modules),
            core_modules="\n".join(module_details),
            key_classes=", ".join(key_classes[:20]),  # Limit to avoid token overflow
            key_functions=", ".join(key_functions[:30]),
        )

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive, accurate, and well-structured API documentation."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm.invoke(messages)
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
            Generate a comprehensive, professional README.md for the {repo_name} project based on this analysis:
            
            **Project**: {repo_name}
            **Description**: {description}
            **Language**: {language}
            **Module Count**: {module_count}
            **Key Dependencies**: {dependencies}
            
            **Main Entry Points**: {main_modules}
            **Example Modules**: {example_modules}
            **Key Features Detected**: {key_features}
            
            Create a professional README with this exact structure:
            
            ```markdown
            # {repo_name}
            
            <div align="center">
              <h1>{repo_name}</h1>
              <p>[Engaging project description based on analysis]</p>
            </div>
            
            ![Python](https://img.shields.io/badge/python-3.6%2B-blue)
            ![License](https://img.shields.io/badge/license-MIT-green)
            
            ## Table of Contents
            - [Installation](#installation)
            - [Quick Start](#quick-start)
            - [Key Features](#key-features)
            - [Usage Examples](#usage-examples)
            - [API Overview](#api-overview)
            - [Development](#development)
            - [License](#license)
            
            ## Installation
            
            To install {repo_name}, you can use pip:
            
            ```bash
            pip install {repo_name}
            ```
            
            ### Dependencies
            
            [List key dependencies with brief explanations]
            
            ## Quick Start
            
            Here's a simple example to get you started:
            
            ```python
            # Realistic example based on actual modules
            ```
            
            ## Key Features
            
            [List and explain key features based on code analysis]
            
            ## Usage Examples
            
            ### [Feature 1]
            ```python
            # Real code examples using actual module/function names
            ```
            
            ### [Feature 2]
            ```python
            # More examples
            ```
            
            ## API Overview
            
            ### Main Entry Points
            [Document the actual entry points found]
            
            ### Core Modules
            [Brief overview of main modules and their purposes]
            
            ## Development
            
            We welcome contributions! To get started:
            
            1. Fork the repository
            2. Create a feature branch
            3. Make your changes
            4. Run tests
            5. Submit a pull request
            
            ### Testing
            ```bash
            pytest
            ```
            
            ## License
            
            This project is licensed under the MIT License.
            ```
            
            **Critical Requirements**:
            - Be specific to THIS project, not generic
            - Use actual module names and functions found in the code analysis
            - Include realistic, runnable examples based on detected functionality
            - Match the technical level of a serious SDK project
            - Use proper markdown formatting with code blocks and syntax highlighting
            - Include professional badges and formatting
            - Make examples practical and immediately useful
            - Focus on developer experience and getting started quickly
            - Ensure all code examples are realistic and based on actual codebase
            
            Generate complete, production-ready README documentation that represents this specific project professionally.
            """,
        )

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

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(
            repo_name=repo_name,
            description=description[:200] + "..." if len(description) > 200 else description,
            language=state["code_analysis"].language,
            module_count=len(state["modules"]),
            dependencies=", ".join(deps[:10]),
            main_modules=", ".join(main_modules_info) if main_modules_info else "No main modules detected",
            example_modules=", ".join(example_modules_info) if example_modules_info else "No examples detected",
            key_features=", ".join(key_features) if key_features else "General Python SDK",
        )

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive, professional README documentation."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm.invoke(messages)
        return result.content if hasattr(result, "content") else str(result)

    def _generate_tutorial_section(self, state) -> str:
        """Generate tutorial documentation section"""
        prompt = PromptTemplate(
            input_variables=["project_context", "entry_points"],
            template="""
            Create a comprehensive, beginner-friendly tutorial for this project:
            
            Project Context: {project_context}
            
            Entry Points: {entry_points}
            
            Generate a complete tutorial with this structure:
            
            # [Project Name] Tutorial
            
            ## Learning Objectives
            By the end of this tutorial, you will:
            1. [Specific objective based on project functionality]
            2. [Another objective]
            3. [etc.]
            
            ## Prerequisites and Setup
            
            ### Prerequisites
            - [List required knowledge and tools]
            
            ### Setup Instructions
            1. **Install Dependencies**: Step-by-step installation
               ```bash
               # Specific commands for this project
               ```
            
            2. **Environment Setup**: Configuration steps
               ```bash
               # Real setup commands
               ```
            
            3. **Verification**: How to verify setup works
               ```python
               # Test code
               ```
            
            ## Step-by-Step Instructions
            
            ### Step 1: [First Major Task]
            1. [Detailed instruction]
               ```python
               # Real code example using actual modules
               ```
            
            2. [Next instruction]
               ```python
               # More code
               ```
            
            ### Step 2: [Second Major Task]
            [Continue with detailed steps using actual project structure]
            
            ### Step 3: [Third Major Task]
            [More steps with real examples]
            
            ## Expected Outputs and Results
            - [Describe what users should see]
            - [Include example outputs]
            - [Show success indicators]
            
            ## Common Troubleshooting Tips
            - **Issue 1**: [Common problem and solution]
            - **Issue 2**: [Another problem and fix]
            - **Issue 3**: [More troubleshooting]
            
            ## Next Steps for Advanced Usage
            - [Suggestion for deeper exploration]
            - [Advanced features to try]
            - [Further learning resources]
            
            **Requirements**:
            - Use actual entry points and module names from the project
            - Include realistic, runnable code examples
            - Make it truly beginner-friendly with clear explanations
            - Include proper error handling in examples
            - Focus on practical, hands-on learning
            - Use the actual project structure and functionality
            - Include real commands and file paths
            - Make examples immediately executable
            
            Generate a comprehensive tutorial that gets users productive quickly with this specific project.
            """,
        )

        project_context = f"""
        Project: {state['code_analysis'].name}
        Purpose: {state['code_analysis'].description}
        Complexity: {state['code_analysis'].complexity_score:.2f}
        """

        entry_points = ", ".join(state["code_analysis"].entry_points)

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(project_context=project_context, entry_points=entry_points)

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive, beginner-friendly tutorial documentation."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm.invoke(messages)
        return result.content if hasattr(result, "content") else str(result)

    def _generate_architecture_section(self, state) -> str:
        """Generate architecture documentation section"""
        prompt = PromptTemplate(
            input_variables=["system_info", "dependencies", "modules_structure"],
            template="""
            Create comprehensive architecture documentation for this system:
            
            System Information: {system_info}
            
            Dependencies: {dependencies}
            
            Module Structure: {modules_structure}
            
            Generate detailed architecture documentation with this structure:
            
            # System Architecture Documentation
            
            ## 1. High-Level System Overview
            
            [Provide a comprehensive overview of the system's purpose, scope, and main architectural principles. Explain what the system does and how it's structured at the highest level.]
            
            ## 2. Component Relationships and Interactions
            
            ### Modules and Their Interactions
            
            [For each major module, describe:]
            - **Module Purpose**: What this module does
            - **Key Components**: Main classes and functions
            - **Dependencies**: What it depends on
            - **Dependents**: What depends on it
            - **Interface**: How other modules interact with it
            
            ### Interaction Flow
            
            [Describe the typical flow of operations through the system:]
            1. [Step-by-step process flow]
            2. [Include actual module/function names]
            3. [Show data transformation at each stage]
            
            ## 3. Data Flow Patterns
            
            [Document how data moves through the system:]
            - **Input Sources**: Where data enters the system
            - **Processing Stages**: How data is transformed
            - **Output Destinations**: Where results go
            - **State Management**: How state is maintained
            - **Error Handling**: How errors propagate
            
            ## 4. Design Decisions and Rationale
            
            [Explain key architectural decisions:]
            - **Architectural Patterns**: Why certain patterns were chosen
            - **Technology Choices**: Rationale for dependencies
            - **Structure Decisions**: Why modules are organized this way
            - **Interface Design**: Why APIs are designed as they are
            
            ## 5. Scalability and Performance Considerations
            
            [Address scalability aspects:]
            - **Performance Bottlenecks**: Potential limitations
            - **Scaling Strategies**: How to scale the system
            - **Resource Management**: How resources are managed
            - **Optimization Opportunities**: Areas for improvement
            
            ## 6. Future Extensibility Points
            
            [Identify extension opportunities:]
            - **Plugin Architecture**: How to add new functionality
            - **API Extensions**: How to extend existing APIs
            - **Module Additions**: How to add new modules
            - **Integration Points**: How to integrate with external systems
            
            **Requirements**:
            - Use actual module names and structures from the analysis
            - Include specific technical details about implementation
            - Reference real dependencies and their purposes
            - Make recommendations based on actual system characteristics
            - Focus on practical architectural insights
            - Include diagrams descriptions where helpful
            - Address real concerns for this specific system
            - Provide actionable guidance for developers
            
            Generate professional, technically accurate architecture documentation that serves as a comprehensive guide for developers working with this system.
            """,
        )

        system_info = f"""
        Name: {state['code_analysis'].name}
        Size: {state['code_analysis'].size} bytes
        Files: {state['code_analysis'].file_count}
        Average Complexity: {state['code_analysis'].complexity_score:.2f}
        """

        dependencies = ", ".join(state["code_analysis"].dependencies)
        modules_structure = self._create_module_structure_text(state["modules"])

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(
            system_info=system_info, dependencies=dependencies, modules_structure=modules_structure
        )

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive, detailed architecture documentation."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm.invoke(messages)
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

        context = "\n".join(state["context_chunks"])

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(code_context=context)

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive, well-structured documentation."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm.invoke(messages)
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

    def _generate_comprehensive_section(self, state) -> str:
        """Generate comprehensive documentation that combines all aspects"""

        # Extract key information
        repo_name = state["code_analysis"].name
        description = state["code_analysis"].description
        language = state["code_analysis"].language
        dependencies = state["code_analysis"].dependencies
        modules = state["modules"]

        # Filter and organize modules
        core_modules = [
            m for m in modules if not any(skip in m.path.lower() for skip in ["test", "example", "__pycache__"])
        ]

        # Build comprehensive context
        module_details = []
        key_classes = []
        key_functions = []

        for module in core_modules[:15]:  # Include more modules for comprehensive doc
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

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(
            repo_name=repo_name,
            description=description[:500] + "..." if len(description) > 500 else description,
            language=language,
            dependencies=", ".join(dependencies[:15]),  # Top 15 dependencies
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
            print(f"Starting generation of {doc_type} documentation...")
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

                # Run the documentation generation graph
                result = self.llm_chain.graph.invoke(state)

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

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(
            module_name=module.name,
            module_content=module.docstring or "No description available",
            functions=functions_text,
            classes=classes_text,
        )

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive module documentation."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm_chain.llm.invoke(messages)
        result_content = result.content if hasattr(result, "content") else str(result)

        return GeneratedDocument(
            title=f"Module: {module.name}",
            content=result_content,
            doc_type="module",
            metadata={"module_path": module.path},
            word_count=len(result_content.split()),
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

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(
            func_name=function.name,
            parameters=", ".join(function.parameters),
            docstring=function.docstring or "No docstring available",
            complexity=function.complexity,
        )

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Generate comprehensive function documentation."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm_chain.llm.invoke(messages)
        return result.content if hasattr(result, "content") else str(result)

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

        # Use latest LangChain syntax with direct invocation
        formatted_prompt = prompt.format(existing_doc=existing_doc, code_context=code_context)

        messages = [
            SystemMessage(
                content="You are a technical documentation expert. Enhance existing documentation with code insights."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        result = self.llm_chain.llm.invoke(messages)
        return result.content if hasattr(result, "content") else str(result)


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

        Args:
            docs_paths: List of paths to documentation directories/files
            exclude_patterns: Patterns to exclude from loading

        Returns:
            True if documentation was loaded successfully
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
                    # Single file - load with LlamaIndex
                    chunks = self.document_reader.read_with_llamaindex(
                        str(Path(docs_path).parent), recursive=False, exclude_patterns=exclude_patterns
                    )
                else:
                    # Directory - use both loaders for comprehensive coverage
                    chunks_langchain = self.document_reader.read_documentation_directory(
                        docs_path, recursive=True, exclude_patterns=exclude_patterns
                    )

                    chunks_llamaindex = self.document_reader.read_with_llamaindex(
                        docs_path, recursive=True, exclude_patterns=exclude_patterns
                    )

                    # Combine and deduplicate
                    chunks = self._deduplicate_chunks(chunks_langchain + chunks_llamaindex)

                self.existing_docs.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {docs_path}")

            if self.existing_docs:
                # Create vector index for semantic search
                self.vector_index = self.document_reader.create_vector_index(self.existing_docs)

                # Log summary
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
            # Create a unique identifier based on content hash and source
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
        doc_types: Optional[List[str]] = None,
        context_similarity_threshold: float = 0.7,
    ) -> List[GeneratedDocument]:
        """
        Generate multiple separate documentation documents using both code analysis and existing documentation context.

        Args:
            repo_path: Path to the repository
            output_format: Output format (markdown, rst, etc.)
            doc_types: Types of documentation to generate
            context_similarity_threshold: Threshold for including similar existing content

        Returns:
            List of separate GeneratedDocument objects (like standard generator)
        """
        try:
            logger.info(f"Generating documentation with context for: {repo_path}")

            # 1. Analyze the code
            logger.info("Analyzing code structure...")
            code_analyzer = CodeAnalyzer(repo_path)
            repository_metadata, modules = code_analyzer.analyze_repository()

            # Convert to dict format expected by the rest of the method
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

            # 2. Prepare documentation context
            context = self._prepare_documentation_context(code_analysis, context_similarity_threshold)

            # 3. Generate separate documentation documents (like standard generator)
            doc_types = doc_types or ["readme", "api", "tutorial", "architecture"]  # Use standard doc types
            generated_docs = []

            for doc_type in doc_types:
                logger.info(f"Generating {doc_type} documentation with README enhancement...")

                # Generate section content with context
                section_content = self._generate_section_with_context(doc_type, context, output_format)

                # Create GeneratedDocument object (like standard generator)
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
                        "enhanced_with_embeddings": True,  # Mark as enhanced
                        "context_used": bool(context.existing_docs),
                    },
                    word_count=len(section_content.split()),
                )

                generated_docs.append(doc)
                logger.info(f"Generated {doc_type} documentation with embedding enhancement ({doc.word_count} words)")

            logger.info(f"Completed generation of {len(generated_docs)} enhanced documents with README embeddings")
            return generated_docs

        except Exception as e:
            logger.error(f"Error generating documentation with context: {e}")
            raise

    def _prepare_documentation_context(
        self, code_analysis: Dict[str, Any], similarity_threshold: float
    ) -> DocumentationContext:
        """Prepare comprehensive context for documentation generation."""

        # Find similar existing content based on code analysis
        similar_content = []
        if self.vector_index and self.existing_docs:
            # Search for content related to key project aspects
            search_queries = [
                f"installation guide for {code_analysis.get('name', 'project')}",
                f"API reference {code_analysis.get('primary_language', '')}",
                f"usage examples {code_analysis.get('name', '')}",
                "getting started tutorial",
                "contributing guidelines",
            ]

            for query in search_queries:
                results = self.document_reader.search_similar_content(query, self.vector_index, top_k=3)
                # Filter by similarity threshold
                filtered_results = [r for r in results if r.get("score", 0) >= similarity_threshold]
                similar_content.extend(filtered_results)

        # Remove duplicates
        unique_similar = []
        seen_content = set()
        for item in similar_content:
            content_hash = hash(item["content"])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_similar.append(item)

        # Get documentation summary
        doc_summary = self.document_reader.get_documentation_summary(self.existing_docs)

        return DocumentationContext(
            existing_docs=self.existing_docs,
            code_analysis=code_analysis,
            similar_content=unique_similar,
            doc_summary=doc_summary,
        )

    def _generate_section_with_context(self, doc_type: str, context: DocumentationContext, output_format: str) -> str:
        """Generate a documentation section using context."""

        # Get relevant existing content for this section
        relevant_content = self._get_relevant_content_for_section(doc_type, context)

        # Create a temporary LLMDocumentationChain to use its detailed prompts
        # (Already imported at top of file)

        config = DocumentationConfig(
            model_name=self.llm_manager.config.model,
            max_tokens=None,  # No token limit for comprehensive docs
            temperature=0.3,
        )
        llm_chain = LLMDocumentationChain(config)

        # Create a state object similar to what the LLMDocumentationChain expects
        # Convert code_analysis dict back to RepositoryMetadata-like structure
        from .analyzer import RepositoryMetadata, ModuleInfo, FunctionInfo, ClassInfo

        # Create RepositoryMetadata from dict
        repo_metadata = RepositoryMetadata(
            name=context.code_analysis.get("name", "Unknown"),
            description=context.code_analysis.get("description", ""),
            language=context.code_analysis.get("language", "Python"),
            size=context.code_analysis.get("size", 0),
            file_count=context.code_analysis.get("file_count", 0),
            dependencies=context.code_analysis.get("dependencies", []),
            entry_points=context.code_analysis.get("entry_points", []),
            test_coverage=context.code_analysis.get("test_coverage", 0.0),
            complexity_score=context.code_analysis.get("complexity_score", 0.0),
        )

        # Create ModuleInfo objects from dict
        modules = []
        for module_dict in context.code_analysis.get("modules", [])[:10]:  # Limit to avoid context overflow
            # Create functions
            functions = []
            for func_dict in module_dict.get("functions", []):
                func = FunctionInfo(
                    name=func_dict.get("name", ""),
                    file_path=func_dict.get("file_path", module_dict.get("path", "")),
                    line_start=func_dict.get("line_start", 1),
                    line_end=func_dict.get("line_end", 1),
                    docstring=func_dict.get("docstring", ""),
                    parameters=func_dict.get("parameters", []),
                    return_type=func_dict.get("return_type", ""),
                    complexity=func_dict.get("complexity", 0),
                    is_async=func_dict.get("is_async", False),
                )
                functions.append(func)

            # Create classes
            classes = []
            for class_dict in module_dict.get("classes", []):
                cls = ClassInfo(
                    name=class_dict.get("name", ""),
                    file_path=class_dict.get("file_path", module_dict.get("path", "")),
                    line_start=class_dict.get("line_start", 1),
                    line_end=class_dict.get("line_end", 1),
                    docstring=class_dict.get("docstring", ""),
                    methods=[],  # Simplified for now
                    attributes=class_dict.get("attributes", []),
                    inheritance=class_dict.get("inheritance", []),
                )
                classes.append(cls)

            # Create module
            module = ModuleInfo(
                name=module_dict.get("name", ""),
                path=module_dict.get("path", ""),
                docstring=module_dict.get("docstring", ""),
                imports=module_dict.get("imports", []),
                functions=functions,
                classes=classes,
                constants=module_dict.get("constants", []),
                language=module_dict.get("language", "python"),
                lines_of_code=module_dict.get("lines_of_code", 0),
                complexity_score=module_dict.get("complexity_score", 0.0),
            )
            modules.append(module)

        # Create state for the detailed prompt methods
        state = {
            "code_analysis": repo_metadata,
            "modules": modules,
            "current_doc_type": doc_type,
            "generated_sections": {},
            "context_chunks": [],
            "final_document": "",
        }

        # Use the detailed prompt methods from LLMDocumentationChain
        try:
            if doc_type == "api":
                return llm_chain._generate_api_section(state)
            elif doc_type == "readme":
                return llm_chain._generate_readme_section(state)
            elif doc_type == "tutorial":
                return llm_chain._generate_tutorial_section(state)
            elif doc_type == "architecture":
                return llm_chain._generate_architecture_section(state)
            elif doc_type == "comprehensive":
                return llm_chain._generate_comprehensive_section(state)
            else:
                return llm_chain._generate_generic_section(state)
        except Exception as e:
            logger.error(f"Error generating {doc_type} section: {e}")
            return f"Error generating {doc_type} section: {str(e)}"

    def _get_relevant_content_for_section(self, doc_type: str, context: DocumentationContext) -> List[Dict[str, Any]]:
        """Get existing content relevant to a specific documentation section."""

        # Define keywords for different section types
        section_keywords = {
            "overview": ["overview", "introduction", "about", "description", "summary"],
            "installation": ["install", "setup", "requirements", "dependencies", "getting started"],
            "usage": ["usage", "examples", "tutorial", "guide", "how to", "quickstart"],
            "api": ["api", "reference", "functions", "methods", "classes", "endpoints"],
            "contributing": ["contributing", "development", "contributing guidelines", "pull request"],
        }

        keywords = section_keywords.get(doc_type, [doc_type])
        relevant_content = []

        # Filter similar content by keywords
        for item in context.similar_content:
            content_lower = item["content"].lower()
            metadata = item.get("metadata", {})
            file_name = metadata.get("file_name", "").lower()

            # Check if content is relevant to this section
            if any(keyword in content_lower or keyword in file_name for keyword in keywords):
                relevant_content.append(item)

        # Sort by relevance score
        relevant_content.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Return top 3 most relevant pieces
        return relevant_content[:3]

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

        # Truncate modules to avoid context overflow
        modules_summary = ""
        if code_analysis.get("modules"):
            modules = code_analysis["modules"][:10]  # Limit to first 10 modules
            modules_summary = "\nKey Modules:\n"
            for module in modules:
                # Only include essential info about each module
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
                # Use summarizer instead of truncation for better context preservation
                if self.summarizer.should_summarize(content["content"]):
                    content_summary = self.summarizer.summarize_existing_documentation(content["content"], doc_type)
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

    def _compile_documentation(
        self, generated_docs: Dict[str, str], context: DocumentationContext, output_format: str
    ) -> str:
        """Compile all generated sections into final documentation."""

        if output_format.lower() == "markdown":
            return self._compile_markdown_documentation(generated_docs, context)
        else:
            # Default compilation
            compiled = []
            for doc_type, content in generated_docs.items():
                compiled.append(f"# {doc_type.title()}\n\n{content}\n\n")
            return "\n".join(compiled)

    def _compile_markdown_documentation(self, generated_docs: Dict[str, str], context: DocumentationContext) -> str:
        """Compile documentation in Markdown format."""

        project_name = context.code_analysis.get("name", "Project")

        sections_order = ["overview", "installation", "usage", "api", "contributing"]

        compiled = [f"# {project_name}\n"]

        # Add table of contents
        toc = ["## Table of Contents\n"]
        for section in sections_order:
            if section in generated_docs:
                toc.append(f"- [{section.title()}](#{section})")
        compiled.append("\n".join(toc) + "\n")

        # Add sections
        for section in sections_order:
            if section in generated_docs:
                compiled.append(f"## {section.title()}\n")
                compiled.append(generated_docs[section])
                compiled.append("\n")

        # Add metadata footer
        if context.doc_summary["total_files"] > 0:
            compiled.append("\n---\n")
            compiled.append("*This documentation was generated using AI with insights from existing documentation.*\n")

        return "\n".join(compiled)

    def _summarize_code_analysis(self, code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the code analysis for metadata."""
        return {
            "name": code_analysis.get("name"),
            "language": code_analysis.get("primary_language"),
            "files_analyzed": len(code_analysis.get("modules", [])),
            "dependencies_count": len(code_analysis.get("dependencies", [])),
            "has_tests": bool(code_analysis.get("test_files")),
            "complexity_score": code_analysis.get("complexity_score", 0),
        }
