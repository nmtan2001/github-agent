"""
Gradio Interface for LLM Documentation Agent
"""

import gradio as gr
import os
from pathlib import Path
import time

# Import our modules
try:
    from src.core.agent import DocumentationAgent, AgentConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    raise


class DocumentationAgentInterface:
    def __init__(self):
        self.agent = None
        self.repo_metadata = None
        self.modules = None
        self.generated_docs = None

    def initialize_agent(self, api_key, model_name, repo_path, readme, api_docs, tutorial, architecture):
        """Initialize the documentation agent"""
        if not api_key:
            return "❌ Error: Please provide OpenAI API key", "", "", ""

        # Convert to absolute path and ensure it exists
        if not os.path.isabs(repo_path):
            # If relative path, resolve it relative to the project root (parent of app dir)
            project_root = Path(__file__).parent.parent
            repo_path = str(project_root / repo_path)

        if not Path(repo_path).exists():
            return f"❌ Error: Repository path does not exist: {repo_path}", "", "", ""

        # Build doc_types list based on checkboxes
        doc_types = []
        if readme:
            doc_types.append("readme")
        if api_docs:
            doc_types.append("api")
        if tutorial:
            doc_types.append("tutorial")
        if architecture:
            doc_types.append("architecture")

        try:
            config = AgentConfig(
                repo_path=repo_path,
                openai_api_key=api_key,
                model_name=model_name,
                doc_types=doc_types,
                output_dir="gradio_output",
            )

            self.agent = DocumentationAgent(config)
            return "✅ Agent initialized successfully!", "", "", ""

        except Exception as e:
            # More detailed error information
            import traceback

            error_details = f"""
❌ Error initializing agent: {str(e)}

**Debug Info:**
- Repository path: {repo_path}
- Path exists: {Path(repo_path).exists()}
- Model: {model_name}
- Doc types: {doc_types}

**Error details:**
{traceback.format_exc()}
"""
            return error_details, "", "", ""

    def analyze_repository(self):
        """Analyze the repository"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", "", "", ""

        try:
            self.repo_metadata, self.modules = self.agent.analyze_repository()

            # Format analysis results
            metrics = f"""
## 📊 Repository Metrics
- **Files:** {self.repo_metadata.file_count}
- **Modules:** {len(self.modules)}
- **Dependencies:** {len(self.repo_metadata.dependencies)}
- **Complexity Score:** {self.repo_metadata.complexity_score:.2f}
"""

            details = f"""
## 📋 Repository Details
- **Name:** {self.repo_metadata.name}
- **Language:** {self.repo_metadata.language}
- **Description:** {self.repo_metadata.description or 'No description available'}
"""

            dependencies = ""
            if self.repo_metadata.dependencies:
                deps_list = "\n".join([f"- {dep}" for dep in self.repo_metadata.dependencies[:10]])
                dependencies = f"""
## 📦 Dependencies (showing first 10)
{deps_list}
"""

            return "✅ Analysis completed!", metrics, details, dependencies

        except Exception as e:
            return f"❌ Analysis failed: {e}", "", "", ""

    def generate_documentation(self):
        """Generate documentation"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", []

        try:
            self.generated_docs = self.agent.generate_documentation()

            # Format generated documents for display
            doc_outputs = []
            for doc in self.generated_docs:
                doc_content = f"""
## {doc.doc_type.title()} Documentation

**Word count:** {doc.word_count}  
**Confidence:** {doc.confidence_score:.2f}

---

{doc.content}
"""
                doc_outputs.append((f"{doc.doc_type}_documentation.md", doc.content))

            return "✅ Documentation generated successfully!", doc_outputs

        except Exception as e:
            return f"❌ Generation failed: {e}", []


def create_interface():
    """Create the Gradio interface"""
    agent_interface = DocumentationAgentInterface()

    with gr.Blocks(
        title="LLM Documentation Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        """,
    ) as app:

        # Header
        gr.Markdown(
            """
        # 🤖 LLM Documentation Agent
        *Intelligent documentation generation and analysis for software repositories*
        """
        )

        # Shared state for storing data
        agent_state = gr.State()

        with gr.Tabs() as tabs:

            # Configuration Tab
            with gr.Tab("⚙️ Configuration", id="config"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### API Settings")
                        api_key = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="sk-...",
                            info="Your OpenAI API key for LLM access",
                        )
                        model_name = gr.Dropdown(
                            label="Model",
                            choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "o4-mini"],
                            value="gpt-4o",
                            info="Choose the LLM model to use (GPT-4o recommended for best quality)",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Documentation Types")
                        readme_check = gr.Checkbox(label="README", value=True)
                        api_docs_check = gr.Checkbox(label="API Documentation", value=True)
                        tutorial_check = gr.Checkbox(label="Tutorial", value=True)
                        architecture_check = gr.Checkbox(label="Architecture", value=True)

                gr.Markdown("### Repository Setup")
                repo_path = gr.Textbox(
                    label="Repository Path",
                    value="python-sdk",
                    placeholder="path/to/your/repository",
                    info="Path to the repository to analyze",
                )

                init_btn = gr.Button("🚀 Initialize Agent", variant="primary", size="lg")
                init_status = gr.Markdown("")

                # Initialize agent event
                init_btn.click(
                    agent_interface.initialize_agent,
                    inputs=[
                        api_key,
                        model_name,
                        repo_path,
                        readme_check,
                        api_docs_check,
                        tutorial_check,
                        architecture_check,
                    ],
                    outputs=[
                        init_status,
                        gr.Textbox(visible=False),
                        gr.Textbox(visible=False),
                        gr.Textbox(visible=False),
                    ],
                )

            # Analysis Tab
            with gr.Tab("🔍 Analysis", id="analysis"):
                analyze_btn = gr.Button("📊 Analyze Repository", variant="primary", size="lg")
                analysis_status = gr.Markdown("")

                with gr.Row():
                    metrics_output = gr.Markdown("")
                    details_output = gr.Markdown("")

                dependencies_output = gr.Markdown("")

                # Analysis event
                analyze_btn.click(
                    agent_interface.analyze_repository,
                    inputs=[],
                    outputs=[analysis_status, metrics_output, details_output, dependencies_output],
                )

            # Generation Tab
            with gr.Tab("📝 Generate", id="generate"):
                generate_btn = gr.Button("📚 Generate Documentation", variant="primary", size="lg")
                generation_status = gr.Markdown("")

                # File downloads area
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generated Documentation")
                        downloads_area = gr.File(label="Download Generated Files", file_count="multiple", visible=False)

                # Documentation preview area
                docs_preview = gr.Markdown("", label="Documentation Preview")

                def generate_and_format(self=agent_interface):
                    status, doc_outputs = agent_interface.generate_documentation()

                    if doc_outputs:
                        # Create preview of all documents
                        preview_content = ""
                        file_outputs = []

                        for filename, content in doc_outputs:
                            preview_content += (
                                f"\n\n## 📄 {filename}\n\n{content[:500]}{'...' if len(content) > 500 else ''}\n\n---"
                            )

                            # Create temporary files for download
                            temp_path = f"temp_{filename}"
                            with open(temp_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            file_outputs.append(temp_path)

                        return status, preview_content, gr.File(value=file_outputs, visible=True)
                    else:
                        return status, "", gr.File(visible=False)

                # Generation event
                generate_btn.click(
                    generate_and_format, inputs=[], outputs=[generation_status, docs_preview, downloads_area]
                )

            # Help Tab
            with gr.Tab("❓ Help", id="help"):
                gr.Markdown(
                    """
                ## 🚀 Quick Start Guide
                
                ### 1. Configuration
                - Enter your OpenAI API key
                - Select your preferred model (GPT-4o recommended for best quality)
                - Choose which documentation types to generate
                - Specify the path to your repository
                
                ### 2. Analysis
                - Click "Analyze Repository" to scan your codebase
                - Review the metrics and repository details
                - Check the detected dependencies
                
                ### 3. Generation
                - Click "Generate Documentation" to create docs
                - Preview the generated content
                - Download individual documentation files
                
                ## 📋 Requirements
                - Valid OpenAI API key
                - Repository with Python code
                - Internet connection for LLM API calls
                
                ## 💡 Tips
                - **GPT-4o**: Best quality, fastest, excellent for complex analysis
                - **GPT-4o-mini**: Cost-effective, fast, good for simple documentation
                - **GPT-4.1-mini**: Enhanced mini model with improved capabilities
                - **o4-mini**: Lightweight model for basic documentation tasks
                - Ensure your repository has clear code structure
                - Check that all dependencies are properly installed
                """
                )

    return app


# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, debug=True)
