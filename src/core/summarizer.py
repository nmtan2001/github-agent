"""
Content Summarization Module

This module provides intelligent summarization of documentation content
using LLM instead of simple truncation.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for content summarization"""

    model_name: str = "gpt-4o-mini"  # Use cheaper model for summarization
    temperature: float = 0.1  # Low temperature for consistent summaries
    max_tokens: Optional[int] = None  # No limit for summary length
    target_length: int = 200  # Target character length for summaries


class ContentSummarizer:
    """Intelligent content summarizer using LLM"""

    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name, temperature=self.config.temperature, max_tokens=self.config.max_tokens
        )

        # Cache to avoid re-summarizing the same content
        self._summary_cache = {}

    def should_summarize(self, content: str, threshold: int = 800) -> bool:
        """Check if content needs summarization based on length"""
        return len(content) > threshold

    def summarize_readme_content(self, readme_content: str) -> str:
        """Summarize README content while preserving key information"""
        if not self.should_summarize(readme_content):
            return readme_content

        # Check cache first
        content_hash = hash(readme_content)
        if content_hash in self._summary_cache:
            return self._summary_cache[content_hash]

        prompt = PromptTemplate(
            input_variables=["content", "target_length"],
            template="""
            Summarize the following README content while preserving the most important information:
            
            {content}
            
            Requirements:
            - Keep approximately {target_length} characters
            - Preserve project name, main purpose, and key features
            - Include installation/setup information if present
            - Maintain technical accuracy
            - Use clear, concise language
            - Focus on what developers need to know
            
            Summary:
            """,
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"content": readme_content, "target_length": self.config.target_length})

            # Cache the result
            self._summary_cache[content_hash] = summary
            logger.info(f"Summarized README content: {len(readme_content)} -> {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing README content: {e}")
            # Fallback to truncation if summarization fails
            return readme_content[: self.config.target_length] + "..."

    def summarize_module_docstring(self, docstring: str, module_name: str) -> str:
        """Summarize module docstring while preserving technical details"""
        if not self.should_summarize(docstring, threshold=300):
            return docstring

        content_hash = hash(f"{module_name}:{docstring}")
        if content_hash in self._summary_cache:
            return self._summary_cache[content_hash]

        prompt = PromptTemplate(
            input_variables=["docstring", "module_name"],
            template="""
            Summarize this module docstring for {module_name}:
            
            {docstring}
            
            Requirements:
            - Keep under 150 characters
            - Preserve the main purpose and functionality
            - Include key technical details
            - Use developer-friendly language
            
            Summary:
            """,
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"docstring": docstring, "module_name": module_name})

            self._summary_cache[content_hash] = summary
            return summary

        except Exception as e:
            logger.error(f"Error summarizing module docstring: {e}")
            return docstring[:150] + "..."

    def summarize_function_info(self, functions: List[Dict[str, Any]], limit: int = 5) -> str:
        """Summarize function information instead of truncating the list"""
        if len(functions) <= limit:
            # No need to summarize if under limit
            return self._format_function_list(functions)

        # Group functions by category/pattern
        categorized = self._categorize_functions(functions)

        if len(categorized) <= limit:
            return self._format_categorized_functions(categorized)

        # Use LLM to create intelligent summary
        functions_text = "\n".join(
            [
                f"- {func.get('name', 'unknown')}({', '.join(func.get('parameters', []))}): {func.get('docstring', 'No description')[:50]}"
                for func in functions[:20]  # Limit input to avoid token overflow
            ]
        )

        prompt = PromptTemplate(
            input_variables=["functions_text", "total_count", "limit"],
            template="""
            Summarize these functions into {limit} key categories or most important functions:
            
            Total functions: {total_count}
            
            {functions_text}
            
            Create a concise summary that highlights:
            - Main function categories (e.g., "Data processing", "API endpoints", "Utilities")
            - Most important public functions
            - Key functionality patterns
            
            Format as a brief list suitable for documentation.
            """,
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"functions_text": functions_text, "total_count": len(functions), "limit": limit})
            return summary

        except Exception as e:
            logger.error(f"Error summarizing functions: {e}")
            return self._format_function_list(functions[:limit])

    def _categorize_functions(self, functions: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Categorize functions by common patterns"""
        categories = {"private": [], "public": [], "init": [], "utils": [], "main": []}

        for func in functions:
            name = func.get("name", "")
            if name.startswith("_"):
                categories["private"].append(func)
            elif name in ["__init__", "__new__"]:
                categories["init"].append(func)
            elif any(keyword in name.lower() for keyword in ["util", "helper", "tool"]):
                categories["utils"].append(func)
            elif any(keyword in name.lower() for keyword in ["main", "run", "execute", "start"]):
                categories["main"].append(func)
            else:
                categories["public"].append(func)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _format_function_list(self, functions: List[Dict[str, Any]]) -> str:
        """Format function list for display"""
        return "\n".join(
            [f"- {func.get('name', 'unknown')}({', '.join(func.get('parameters', []))})" for func in functions]
        )

    def _format_categorized_functions(self, categorized: Dict[str, List[Dict]]) -> str:
        """Format categorized functions for display"""
        result = []
        for category, functions in categorized.items():
            if functions:
                result.append(
                    f"{category.title()} functions: {', '.join([f.get('name', 'unknown') for f in functions[:3]])}"
                )
                if len(functions) > 3:
                    result.append(f" (and {len(functions) - 3} more)")
        return "\n".join(result)

    def summarize_existing_documentation(self, doc_content: str, doc_type: str) -> str:
        """Summarize existing documentation for context"""
        if not self.should_summarize(doc_content, threshold=500):
            return doc_content

        content_hash = hash(f"{doc_type}:{doc_content}")
        if content_hash in self._summary_cache:
            return self._summary_cache[content_hash]

        prompt = PromptTemplate(
            input_variables=["content", "doc_type"],
            template="""
            Summarize this existing {doc_type} documentation to provide context for new documentation generation:
            
            {content}
            
            Requirements:
            - Extract key concepts and structure
            - Preserve important technical details
            - Keep formatting elements that show document structure
            - Focus on content that would be useful for generating similar documentation
            - Target around 300 words
            
            Summary:
            """,
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"content": doc_content, "doc_type": doc_type})

            self._summary_cache[content_hash] = summary
            logger.info(f"Summarized {doc_type} documentation: {len(doc_content)} -> {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing {doc_type} documentation: {e}")
            return doc_content[:300] + "..."

    def clear_cache(self):
        """Clear the summary cache"""
        self._summary_cache.clear()
        logger.info("Summary cache cleared")
