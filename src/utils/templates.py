"""
Template Manager for documentation generation templates and prompts.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Manages documentation templates and prompt templates.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize Template Manager.

        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir or self._get_default_templates_dir()
        self.templates = self._load_templates()

    def _get_default_templates_dir(self) -> str:
        """Get the default templates directory."""
        current_dir = Path(__file__).parent
        templates_dir = current_dir / "templates"

        # Create templates directory if it doesn't exist
        templates_dir.mkdir(exist_ok=True)

        return str(templates_dir)

    def _load_templates(self) -> Dict[str, Any]:
        """Load all templates from the templates directory."""
        templates = {}

        # Load built-in templates
        templates.update(self._get_builtin_templates())

        # Load custom templates from files
        templates_path = Path(self.templates_dir)
        if templates_path.exists():
            for template_file in templates_path.glob("*.json"):
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        file_templates = json.load(f)
                        templates.update(file_templates)
                        logger.info(f"Loaded templates from {template_file}")
                except Exception as e:
                    logger.error(f"Error loading template file {template_file}: {e}")

        return templates

    def _get_builtin_templates(self) -> Dict[str, Any]:
        """Get built-in template definitions."""
        return {
            "documentation_sections": {
                "overview": {
                    "title": "Overview",
                    "description": "Project overview and introduction",
                    "prompt_template": """
Generate a comprehensive overview section for the following project:

Project Name: {project_name}
Description: {project_description}
Primary Language: {primary_language}
Dependencies: {dependencies}

{existing_context}

Create an engaging overview that includes:
1. Brief project description
2. Key features and capabilities
3. Target audience
4. Technology stack overview

Format: {output_format}
""",
                    "required_fields": ["project_name", "project_description", "primary_language"],
                },
                "installation": {
                    "title": "Installation",
                    "description": "Installation and setup instructions",
                    "prompt_template": """
Generate detailed installation instructions for the following project:

Project Name: {project_name}
Primary Language: {primary_language}
Dependencies: {dependencies}
Package Manager: {package_manager}

{existing_context}

Create comprehensive installation instructions including:
1. Prerequisites and system requirements
2. Installation methods (package manager, source, etc.)
3. Verification steps
4. Common installation issues and solutions

Format: {output_format}
""",
                    "required_fields": ["project_name", "primary_language"],
                },
                "usage": {
                    "title": "Usage",
                    "description": "Usage examples and basic tutorials",
                    "prompt_template": """
Generate usage documentation for the following project:

Project Name: {project_name}
Main Functions: {main_functions}
Classes: {main_classes}
API Endpoints: {api_endpoints}

{existing_context}

Create practical usage documentation including:
1. Quick start guide
2. Basic usage examples
3. Common use cases
4. Code snippets and examples

Format: {output_format}
""",
                    "required_fields": ["project_name"],
                },
                "api": {
                    "title": "API Reference",
                    "description": "Detailed API documentation",
                    "prompt_template": """
Generate comprehensive API reference documentation for:

Project Name: {project_name}
Functions: {functions}
Classes: {classes}
Modules: {modules}

{existing_context}

Create detailed API documentation including:
1. Function/method signatures
2. Parameters and return values
3. Usage examples for each function
4. Error handling and exceptions

Format: {output_format}
""",
                    "required_fields": ["project_name"],
                },
                "contributing": {
                    "title": "Contributing",
                    "description": "Contributing guidelines and development setup",
                    "prompt_template": """
Generate contributing guidelines for the following project:

Project Name: {project_name}
Repository Structure: {repo_structure}
Development Dependencies: {dev_dependencies}
Testing Framework: {test_framework}

{existing_context}

Create comprehensive contributing guidelines including:
1. Development environment setup
2. Coding standards and style guide
3. Testing requirements
4. Pull request process
5. Issue reporting guidelines

Format: {output_format}
""",
                    "required_fields": ["project_name"],
                },
            },
            "prompts": {
                "code_analysis_summary": """
Analyze the following code structure and provide a summary:

Repository: {repo_path}
Primary Language: {primary_language}
Files: {file_count}
Functions: {function_count}
Classes: {class_count}

Modules:
{modules_info}

Provide a concise analysis covering:
1. Project structure and organization
2. Key components and functionality
3. Architecture patterns identified
4. Code quality observations
""",
                "documentation_improvement": """
Review the following existing documentation and suggest improvements:

Current Documentation:
{current_docs}

Code Analysis:
{code_analysis}

Provide specific suggestions for:
1. Missing sections or information
2. Clarity and readability improvements
3. Additional examples or explanations needed
4. Structural improvements
""",
                "context_integration": """
Integrate insights from existing documentation into new content:

New Content Topic: {topic}
Existing Documentation Context:
{existing_context}

Code Analysis:
{code_analysis}

Create content that:
1. Builds upon existing documentation
2. Maintains consistency with existing style
3. Fills gaps in current documentation
4. Provides fresh perspectives where appropriate
""",
            },
        }

    def get_template(self, template_type: str, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific template.

        Args:
            template_type: Type of template (e.g., 'documentation_sections', 'prompts')
            template_name: Name of the specific template

        Returns:
            Template dictionary or None if not found
        """
        return self.templates.get(template_type, {}).get(template_name)

    def render_template(self, template_type: str, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with the provided context.

        Args:
            template_type: Type of template
            template_name: Name of the template
            context: Context variables for template rendering

        Returns:
            Rendered template string
        """
        template = self.get_template(template_type, template_name)
        if not template:
            raise ValueError(f"Template not found: {template_type}.{template_name}")

        template_str = template.get("prompt_template", template.get("template", ""))
        if not template_str:
            raise ValueError(f"No template content found for: {template_type}.{template_name}")

        # Check required fields
        required_fields = template.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in context]
        if missing_fields:
            logger.warning(f"Missing required fields for template {template_name}: {missing_fields}")

        try:
            return template_str.format(**context)
        except KeyError as e:
            logger.error(f"Missing context variable for template {template_name}: {e}")
            raise

    def get_section_templates(self) -> Dict[str, Any]:
        """Get all documentation section templates."""
        return self.templates.get("documentation_sections", {})

    def get_prompt_template(self, prompt_name: str) -> Optional[str]:
        """Get a specific prompt template."""
        return self.templates.get("prompts", {}).get(prompt_name)

    def add_custom_template(self, template_type: str, template_name: str, template_data: Dict[str, Any]):
        """
        Add a custom template.

        Args:
            template_type: Type of template
            template_name: Name for the template
            template_data: Template data including prompt_template and metadata
        """
        if template_type not in self.templates:
            self.templates[template_type] = {}

        self.templates[template_type][template_name] = template_data
        logger.info(f"Added custom template: {template_type}.{template_name}")

    def save_templates(self, filename: str = "custom_templates.json"):
        """
        Save current templates to a file.

        Args:
            filename: Name of the file to save templates to
        """
        filepath = Path(self.templates_dir) / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2, ensure_ascii=False)
            logger.info(f"Templates saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving templates to {filepath}: {e}")

    def load_templates_from_file(self, filepath: str):
        """
        Load additional templates from a file.

        Args:
            filepath: Path to the template file
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                file_templates = json.load(f)
                self.templates.update(file_templates)
                logger.info(f"Loaded templates from {filepath}")
        except Exception as e:
            logger.error(f"Error loading templates from {filepath}: {e}")

    def list_available_templates(self) -> Dict[str, List[str]]:
        """
        List all available templates by type.

        Returns:
            Dictionary mapping template types to lists of template names
        """
        return {template_type: list(templates.keys()) for template_type, templates in self.templates.items()}

    def validate_template(self, template_data: Dict[str, Any]) -> List[str]:
        """
        Validate a template structure.

        Args:
            template_data: Template data to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for required template fields
        if "prompt_template" not in template_data and "template" not in template_data:
            errors.append("Template must have either 'prompt_template' or 'template' field")

        # Check for title and description (recommended)
        if "title" not in template_data:
            errors.append("Template should have a 'title' field")

        if "description" not in template_data:
            errors.append("Template should have a 'description' field")

        return errors
