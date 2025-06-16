"""
Code Analysis Module

This module provides comprehensive code analysis capabilities for extracting
metadata, structure, and semantic information from code repositories.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from git import Repo
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from pydantic import BaseModel


@dataclass
class FileInfo:
    """Information about a single file"""

    path: str
    content: str
    size: int
    language: str
    extension: str


@dataclass
class FunctionInfo:
    """Information about a function"""

    name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str]
    parameters: List[str]
    return_type: Optional[str]
    complexity: int
    is_async: bool


@dataclass
class ClassInfo:
    """Information about a class"""

    name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str]
    methods: List[FunctionInfo]
    attributes: List[str]
    inheritance: List[str]


@dataclass
class ModuleInfo:
    """Information about a module"""

    name: str
    path: str
    docstring: Optional[str]
    imports: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    constants: List[str]
    language: str = "python"
    lines_of_code: int = 0
    complexity_score: float = 0.0


@dataclass
class RepositoryMetadata:
    """Repository-level metadata"""

    name: str
    description: str
    language: str
    size: int
    file_count: int
    dependencies: List[str]
    entry_points: List[str]
    test_coverage: float
    complexity_score: float


class CodeAnalyzer:
    """Advanced code analyzer using AST and Tree-sitter"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

        # Try to initialize Git repository with error handling
        self.repo = None
        if os.path.exists(os.path.join(repo_path, ".git")):
            try:
                self.repo = Repo(repo_path)
            except Exception as e:
                # Git repository exists but is invalid/corrupted, continue without Git features
                print(f"Warning: Git repository found but invalid: {e}")
                self.repo = None

        # Initialize Tree-sitter parser
        try:
            PY_LANGUAGE = Language(tspython.language())
            self.parser = Parser()
            self.parser.set_language(PY_LANGUAGE)
        except Exception:
            # Fallback if tree-sitter initialization fails
            self.parser = None

        # File patterns to include/exclude
        self.include_patterns = [
            "*.py",
            "*.js",
            "*.ts",
            "*.java",
            "*.cpp",
            "*.c",
            "*.h",
            "*.go",
            "*.rs",
            "*.rb",
            "*.php",
            "*.cs",
            "*.kt",
            "*.scala",
        ]
        self.exclude_patterns = ["__pycache__", ".git", ".venv", "node_modules", "*.pyc", "*.pyo", "*.pyd", ".DS_Store"]

    def analyze_repository(self) -> Tuple[RepositoryMetadata, List[ModuleInfo]]:
        """Analyze entire repository and return comprehensive metadata"""
        files = self._get_source_files()
        modules = []

        for file_info in files:
            if file_info.language == "python":
                module_info = self._analyze_python_file(file_info)
                if module_info:
                    modules.append(module_info)

        metadata = self._build_repository_metadata(files, modules)
        return metadata, modules

    def _get_source_files(self) -> List[FileInfo]:
        """Get all source code files in the repository"""
        files = []

        for root, dirs, filenames in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [
                d for d in dirs if not any(re.match(pattern.replace("*", ".*"), d) for pattern in self.exclude_patterns)
            ]

            for filename in filenames:
                file_path = Path(root) / filename
                if self._should_include_file(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        files.append(
                            FileInfo(
                                path=str(file_path.relative_to(self.repo_path)),
                                content=content,
                                size=len(content),
                                language=self._detect_language(file_path),
                                extension=file_path.suffix,
                            )
                        )
                    except (UnicodeDecodeError, PermissionError):
                        continue

        return files

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included in analysis"""
        # Check file extension
        if not any(file_path.match(pattern) for pattern in self.include_patterns):
            return False

        # Check exclude patterns
        if any(re.search(pattern.replace("*", ".*"), str(file_path)) for pattern in self.exclude_patterns):
            return False

        # Skip empty files or very large files (>1MB)
        try:
            if file_path.stat().st_size == 0 or file_path.stat().st_size > 1024 * 1024:
                return False
        except (OSError, PermissionError):
            return False

        return True

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".kt": "kotlin",
            ".scala": "scala",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    def _analyze_python_file(self, file_info: FileInfo) -> Optional[ModuleInfo]:
        """Analyze a Python file using AST"""
        try:
            tree = ast.parse(file_info.content)
        except SyntaxError:
            return None

        # Extract module docstring
        docstring = ast.get_docstring(tree)

        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])

        # Extract functions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node, file_info.path)
                functions.append(func_info)

        # Extract classes
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node, file_info.path)
                classes.append(class_info)

        # Extract constants (module-level assignments)
        constants = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(target.id)

        # Calculate lines of code and complexity
        lines_of_code = len(file_info.content.split("\n"))
        complexity_score = sum(func.complexity for func in functions) / max(len(functions), 1)

        return ModuleInfo(
            name=Path(file_info.path).stem,
            path=file_info.path,
            docstring=docstring,
            imports=imports,
            functions=functions,
            classes=classes,
            constants=constants,
            language=file_info.language,
            lines_of_code=lines_of_code,
            complexity_score=complexity_score,
        )

    def _analyze_function(self, node: ast.FunctionDef, file_path: str) -> FunctionInfo:
        """Analyze a function AST node"""
        # Extract parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)

        # Extract return type annotation
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(node)

        return FunctionInfo(
            name=node.name,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            parameters=params,
            return_type=return_type,
            complexity=complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    def _analyze_class(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Analyze a class AST node"""
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, file_path)
                methods.append(method_info)

        # Extract attributes (simple assignment in __init__)
        attributes = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                attributes.append(target.attr)

        # Extract inheritance
        inheritance = []
        for base in node.bases:
            inheritance.append(ast.unparse(base))

        return ClassInfo(
            name=node.name,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            methods=methods,
            attributes=attributes,
            inheritance=inheritance,
        )

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With, ast.AsyncWith, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1

        return complexity

    def _build_repository_metadata(self, files: List[FileInfo], modules: List[ModuleInfo]) -> RepositoryMetadata:
        """Build repository-level metadata"""
        # Detect main language
        language_counts = {}
        for file_info in files:
            lang = file_info.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        main_language = max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else "unknown"

        # Calculate total size
        total_size = sum(file_info.size for file_info in files)

        # Extract dependencies (basic implementation)
        dependencies = set()
        for module in modules:
            for imp in module.imports:
                if "." in imp:
                    dependencies.add(imp.split(".")[0])
                else:
                    dependencies.add(imp)

        # Calculate average complexity
        all_functions = []
        for module in modules:
            all_functions.extend(module.functions)
            for cls in module.classes:
                all_functions.extend(cls.methods)

        avg_complexity = sum(f.complexity for f in all_functions) / len(all_functions) if all_functions else 0

        return RepositoryMetadata(
            name=self.repo_path.name,
            description=self._extract_description(),
            language=main_language,
            size=total_size,
            file_count=len(files),
            dependencies=list(dependencies),
            entry_points=self._find_entry_points(modules),
            test_coverage=0.0,  # Would need actual test runner integration
            complexity_score=avg_complexity,
        )

    def _extract_description(self) -> str:
        """Extract repository description from README or other sources by searching recursively."""
        readme_patterns = ["README.md", "readme.md", "README.rst", "README.txt", "README"]
        exclude_dirs = [".git", ".venv", "node_modules", "__pycache__", "dist", "build"]

        for pattern in readme_patterns:
            found_files = list(self.repo_path.glob(f"**/{pattern}"))

            for file_path in found_files:
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    lines = content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith("#") and not line.startswith("="):
                            print(f"Extracted description from {file_path}")
                            return line[:200] + ("..." if len(line) > 200 else "")

                    return "README found, but no suitable description line."

                except (UnicodeDecodeError, PermissionError) as e:
                    print(f"Could not read {file_path}: {e}")
                    continue

        print("No README file found to extract a description.")
        return "No description available"

    def _find_entry_points(self, modules: List[ModuleInfo]) -> List[str]:
        """Find potential entry points in the codebase"""
        entry_points = []

        for module in modules:
            # Look for main functions
            for func in module.functions:
                if func.name in ["main", "__main__", "run", "start"]:
                    entry_points.append(f"{module.path}::{func.name}")

            # Look for classes with main methods
            for cls in module.classes:
                for method in cls.methods:
                    if method.name in ["main", "run", "start"]:
                        entry_points.append(f"{module.path}::{cls.name}.{method.name}")

        return entry_points

    def get_file_dependencies(self, file_path: str) -> Dict[str, List[str]]:
        """Get dependencies for a specific file"""
        dependencies = {"imports": [], "internal_calls": [], "external_calls": []}

        try:
            full_path = self.repo_path / file_path
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Collect imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dependencies["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            dependencies["imports"].append(f"{module}.{alias.name}")

            # Collect function calls (basic implementation)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        dependencies["internal_calls"].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        dependencies["external_calls"].append(ast.unparse(node.func))

        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            pass

        return dependencies
