"""
LLM Agent for Software Code Documentation

This package provides an intelligent agent system for automatically generating,
maintaining, and comparing software documentation using Large Language Models.
"""

__version__ = "1.0.0"
__author__ = "LLM Documentation Agent Team"

from .core.agent import DocumentationAgent
from .core.analyzer import CodeAnalyzer
from .core.generator import DocumentationGenerator
from .core.comparator import DocumentationComparator

__all__ = ["DocumentationAgent", "CodeAnalyzer", "DocumentationGenerator", "DocumentationComparator"]
