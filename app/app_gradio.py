"""
Gradio Interface for LLM Documentation Agent
"""

import gradio as gr
import os
from pathlib import Path
import time
import tempfile
import atexit
import shutil
import json

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
        self.comparison_results = None
        # Create a temporary directory for this app session
        self.temp_dir = tempfile.mkdtemp(prefix="gradio_docs_")
        # Register cleanup function to run when app closes
        atexit.register(self._cleanup_temp_files)
        print(f"📁 Created temporary directory: {self.temp_dir}")

    def _cleanup_temp_files(self):
        """Clean up temporary files when app closes"""
        try:
            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temp directory: {e}")

    def run_example_walkthrough(self):
        """Run an example walkthrough with a demonstration repository"""
        walkthrough_md = """
# 🚀 Example Walkthrough: Flask Repository Analysis

## Repository Summary
We'll demonstrate the agent using the Flask web framework repository:
- **Repository**: https://github.com/pallets/flask
- **Language**: Python
- **Type**: Web framework
- **Purpose**: Lightweight WSGI web application framework

## What the Agent Will Do:
1. 🔍 **Analyze** the repository structure and dependencies
2. 📝 **Generate** comprehensive documentation (README, API docs, tutorials)
3. ⚖️ **Compare** generated docs with existing ones
4. 📊 **Provide** similarity scores and recommendations

## Expected Outputs:
- Repository analysis with complexity metrics
- Generated documentation in multiple formats
- Comparison scores showing alignment with original docs
- Recommendations for improvement

*Click "Initialize Agent" below with the Flask repository URL to see this in action!*
"""
        return walkthrough_md

    def initialize_agent(self, api_key, model_name, repo_path, readme, api_docs, tutorial, architecture):
        """Initialize the documentation agent"""
        if not api_key:
            return (
                "❌ Error: Please provide OpenAI API key",
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
            )

        # Use relative path directly (like the working scripts)
        # Only convert to absolute if user provided an absolute path
        if not os.path.isabs(repo_path):
            # Keep relative path as-is for better compatibility
            pass

        if not Path(repo_path).exists() and not AgentConfig.is_github_url(repo_path):
            return (
                f"❌ Error: Repository path does not exist: {repo_path}",
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
            )

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
                include_comparison=True,  # Always enable comparison
            )

            self.agent = DocumentationAgent(config)

            return (
                "✅ Agent initialized successfully!",
                "",
                "",
                "",
                gr.update(visible=True),
                gr.update(visible=True),
            )

        except Exception as e:
            # More detailed error information
            import traceback

            error_details = f"""
❌ Error initializing agent: {str(e)}

**Debug Info:**
- Repository path: {repo_path}
- Path exists: {Path(repo_path).exists() if not AgentConfig.is_github_url(repo_path) else "GitHub URL (will be cloned)"}
- Model: {model_name}
- Doc types: {doc_types}
- Comparison enabled: Always

**Error details:**
{traceback.format_exc()}
"""
            return (
                error_details,
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def analyze_repository(self):
        """Analyze the repository"""
        if not self.agent:
            return (
                "⚠️ Warning: Please initialize the agent first",
                "",
                "",
                "",
                gr.update(visible=False),
            )

        try:
            self.repo_metadata, self.modules = self.agent.analyze_repository()

            # Format analysis results
            metrics = f"""
## 📊 Repository Metrics
- **Files:** {self.repo_metadata.file_count}
- **Modules:** {len(self.modules)}
- **Dependencies:** {len(self.repo_metadata.dependencies)}
- **Complexity Score:** {self.repo_metadata.complexity_score:.2f}
- **Repository Size:** {self.repo_metadata.size:,} bytes
"""

            details = f"""
## 📋 Repository Details
- **Name:** {self.repo_metadata.name}
- **Language:** {self.repo_metadata.language}
- **Description:** {self.repo_metadata.description or 'No description available'}
- **License:** {getattr(self.repo_metadata, 'license', 'Not specified')}
"""

            dependencies = ""
            if self.repo_metadata.dependencies:
                deps_list = "\n".join([f"- {dep}" for dep in self.repo_metadata.dependencies[:10]])
                dependencies = f"""
## 📦 Dependencies (showing first 10)
{deps_list}
"""
                if len(self.repo_metadata.dependencies) > 10:
                    dependencies += f"\n*... and {len(self.repo_metadata.dependencies) - 10} more dependencies*"

            # Module structure analysis
            module_structure = f"""
## 🗂️ Module Structure
"""
            for module in self.modules[:8]:  # Show first 8 modules
                module_structure += f"""
### {module.name}
- **Path:** `{module.path}`
- **Functions:** {len(module.functions)}
- **Classes:** {len(module.classes)}
- **Lines of Code:** {module.lines_of_code}
"""

            if len(self.modules) > 8:
                module_structure += f"\n*... and {len(self.modules) - 8} more modules*"

            return (
                "✅ Analysis completed!",
                metrics,
                details,
                dependencies + module_structure,
                gr.update(visible=True),
            )

        except Exception as e:
            return (
                f"❌ Analysis failed: {e}",
                "",
                "",
                "",
                gr.update(visible=False),
            )

    def generate_documentation(self):
        """Generate documentation"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", "", gr.update(visible=False)

        try:
            self.generated_docs = self.agent.generate_documentation()

            # Format generated documents for display
            doc_outputs = []
            preview_content = "# 📚 Generated Documentation\n\n"

            for doc in self.generated_docs:
                preview_content += f"""
## {doc.doc_type.title()} Documentation
**Word count:** {doc.word_count} | **Confidence:** {doc.confidence_score:.2f}

### Content Preview:
{doc.content[:1500]}{'...' if len(doc.content) > 1500 else ''}

---

"""
                doc_outputs.append((f"{doc.doc_type}_documentation.md", doc.content))

            # Create temporary files for download in temp directory
            file_outputs = []
            for filename, content in doc_outputs:
                temp_path = os.path.join(self.temp_dir, filename)
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                file_outputs.append(temp_path)

            return (
                "✅ Documentation generated successfully!",
                preview_content,
                gr.update(value=file_outputs, visible=True),
            )

        except Exception as e:
            return f"❌ Generation failed: {e}", "", gr.update(visible=False)

    def compare_documentation(self):
        """Compare generated documentation with existing documentation"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", gr.update(visible=False)

        if not self.generated_docs:
            return "⚠️ Warning: Please generate documentation first", gr.update(visible=False)

        try:
            self.comparison_results = self.agent.compare_with_existing()

            if not self.comparison_results:
                return (
                    "ℹ️ No existing documentation found for comparison",
                    gr.update(visible=False),
                )

            # Format comparison results
            comparison_content = "# ⚖️ Documentation Comparison Results\n\n"

            overall_scores = []
            for doc_type, result in self.comparison_results.items():
                # Calculate overall score
                overall_score = (
                    result.metrics.semantic_similarity
                    + result.metrics.content_coverage
                    + result.metrics.structural_similarity
                ) / 3
                overall_scores.append(overall_score)

                comparison_content += f"""
## {doc_type.title()} Documentation Comparison

### 📊 Similarity Metrics
- **Semantic Similarity:** {result.metrics.semantic_similarity:.3f} (How similar in meaning)
- **Content Coverage:** {result.metrics.content_coverage:.3f} (How much content is covered)
- **Structural Similarity:** {result.metrics.structural_similarity:.3f} (How similar in structure)
- **ROUGE-1 Score:** {result.metrics.rouge_scores.get('rouge1', 0):.3f} (Word overlap)
- **ROUGE-2 Score:** {result.metrics.rouge_scores.get('rouge2', 0):.3f} (Bigram overlap)
- **ROUGE-L Score:** {result.metrics.rouge_scores.get('rougeL', 0):.3f} (Longest common sequence)
- **Word Count Ratio:** {result.metrics.word_count_ratio:.3f} (Length comparison)

### 🎯 Overall Quality Score: {overall_score:.3f}

### 📋 Analysis Summary
{result.detailed_analysis.get('summary', 'No summary available')}

### 💡 Top Recommendations
"""
                for i, rec in enumerate(result.recommendations[:3], 1):
                    comparison_content += f"{i}. {rec}\n"

                if result.missing_sections:
                    comparison_content += f"""
### ❌ Missing Sections in Generated Doc
{', '.join(result.missing_sections)}
"""

                if result.additional_sections:
                    comparison_content += f"""
### ✅ Additional Sections in Generated Doc
{', '.join(result.additional_sections)}
"""

                comparison_content += "\n---\n"

            # Overall summary
            avg_score = sum(overall_scores) / len(overall_scores)
            comparison_content = (
                f"""
# 🎯 Overall Comparison Summary
**Average Quality Score:** {avg_score:.3f} / 1.000

{'🎉 Excellent alignment with existing documentation!' if avg_score > 0.8 else '👍 Good alignment with existing documentation!' if avg_score > 0.6 else '⚠️ Moderate alignment - consider improvements' if avg_score > 0.4 else '❌ Low alignment - significant improvements needed'}

---

"""
                + comparison_content
            )

            # Save comparison results to file
            comparison_file = os.path.join(self.temp_dir, "comparison_results.json")
            with open(comparison_file, "w", encoding="utf-8") as f:
                # Convert comparison results to JSON-serializable format
                serializable_results = {}
                for doc_type, result in self.comparison_results.items():
                    serializable_results[doc_type] = {
                        "metrics": {
                            "semantic_similarity": result.metrics.semantic_similarity,
                            "structural_similarity": result.metrics.structural_similarity,
                            "content_coverage": result.metrics.content_coverage,
                            "rouge_scores": result.metrics.rouge_scores,
                            "word_count_ratio": result.metrics.word_count_ratio,
                        },
                        "recommendations": result.recommendations,
                        "missing_sections": result.missing_sections,
                        "additional_sections": result.additional_sections,
                    }
                json.dump(serializable_results, f, indent=2)

            return (
                comparison_content,
                gr.update(visible=True),
            )

        except Exception as e:
            return f"❌ Comparison failed: {e}", gr.update(visible=False)

    def full_workflow(self, api_key, model_name, repo_path, readme, api_docs, tutorial, architecture):
        """Complete workflow: Initialize -> Analyze -> Generate -> Compare"""
        # Step 1: Initialize Agent
        init_status, _, _, _, analysis_visible, generation_visible = self.initialize_agent(
            api_key, model_name, repo_path, readme, api_docs, tutorial, architecture
        )

        if "❌" in init_status:
            return (
                init_status,
                "⏸️ Workflow stopped due to initialization error",
                "⏸️ Workflow stopped due to initialization error",
                "",
                "",
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )

        # Step 2: Analyze Repository
        analysis_status = "🔄 Analyzing repository..."
        try:
            analysis_result = self.analyze_repository()
            analysis_status, metrics, details, dependencies, results_visible = analysis_result

            if "❌" in analysis_status:
                return (
                    init_status,
                    analysis_status,
                    "⏸️ Workflow stopped due to analysis error",
                    metrics,
                    details,
                    dependencies,
                    results_visible,
                    "",
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                )
        except Exception as e:
            analysis_status = f"❌ Analysis failed: {e}"
            return (
                init_status,
                analysis_status,
                "⏸️ Workflow stopped due to analysis error",
                "",
                "",
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )

        # Step 3: Generate Documentation
        generation_status = "🔄 Generating documentation..."
        try:
            generation_result = self.generate_documentation()
            generation_status, docs_preview, downloads = generation_result

            if "❌" in generation_status:
                return (
                    init_status,
                    analysis_status,
                    generation_status,
                    "⏸️ Workflow stopped due to generation error",
                    metrics,
                    details,
                    dependencies,
                    gr.update(visible=True),
                    docs_preview,
                    downloads,
                    "",
                    gr.update(visible=False),
                )
        except Exception as e:
            generation_status = f"❌ Generation failed: {e}"
            return (
                init_status,
                analysis_status,
                generation_status,
                metrics,
                details,
                dependencies,
                gr.update(visible=True),
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )

        # Step 4: Compare Documentation
        comparison_status = "🔄 Comparing with existing documentation..."
        comparison_content = ""
        comparison_visible = gr.update(visible=False)

        try:
            comparison_result = self.compare_documentation()
            comparison_content, comparison_visible = comparison_result
            if "❌" in comparison_content or "⚠️" in comparison_content:
                comparison_status = comparison_content.split("\n")[0]  # First line as status
            else:
                comparison_status = "✅ Comparison completed successfully!"
        except Exception as e:
            comparison_status = f"❌ Comparison failed: {e}"
            comparison_content = f"Error during comparison: {str(e)}"
            comparison_visible = gr.update(visible=True)

        return (
            init_status,
            analysis_status,
            generation_status,
            comparison_status,
            metrics,
            details,
            dependencies,
            gr.update(visible=True),
            docs_preview,
            downloads,
            comparison_content,
            comparison_visible,
        )

    def run_progressive_workflow(self, api_key, model_name, repo_path, readme, api_docs, tutorial, architecture):
        """Progressive workflow that yields results after each step"""

        # Step 1: Initialize Agent
        yield (
            "🔄 Initializing agent...",
            "",
            "",
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            "",
            gr.update(visible=False),
        )

        try:
            init_result = self.initialize_agent(
                api_key, model_name, repo_path, readme, api_docs, tutorial, architecture
            )
            init_status = init_result[0]

            if "❌" in init_status:
                yield (
                    init_status,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                )
                return
        except Exception as e:
            yield (
                f"❌ Initialization failed: {e}",
                "",
                "",
                "",
                "",
                "",
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )
            return

        # Step 2: Analyze Repository
        yield (
            init_status,
            "🔄 Analyzing repository structure...",
            "",
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            "",
            gr.update(visible=False),
        )

        try:
            analysis_result = self.analyze_repository()
            analysis_status, metrics, details, dependencies, results_visible = analysis_result

            if "❌" in analysis_status:
                yield (
                    init_status,
                    analysis_status,
                    "",
                    "",
                    metrics,
                    details,
                    dependencies,
                    results_visible,
                    "",
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                )
                return

            # Show analysis results immediately
            yield (
                init_status,
                analysis_status,
                "",
                "",
                metrics,
                details,
                dependencies,
                results_visible,
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )

        except Exception as e:
            analysis_status = f"❌ Analysis failed: {e}"
            yield (
                init_status,
                analysis_status,
                "",
                "",
                "",
                "",
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )
            return

        # Step 3: Generate Documentation
        yield (
            init_status,
            analysis_status,
            "🔄 Generating documentation...",
            "",
            metrics,
            details,
            dependencies,
            gr.update(visible=True),
            "",
            gr.update(visible=False),
            "",
            gr.update(visible=False),
        )

        try:
            generation_result = self.generate_documentation()
            generation_status, docs_preview, downloads = generation_result

            if "❌" in generation_status:
                yield (
                    init_status,
                    analysis_status,
                    generation_status,
                    "",
                    metrics,
                    details,
                    dependencies,
                    gr.update(visible=True),
                    docs_preview,
                    downloads,
                    "",
                    gr.update(visible=False),
                )
                return

            # Show generation results immediately
            yield (
                init_status,
                analysis_status,
                generation_status,
                "",
                metrics,
                details,
                dependencies,
                gr.update(visible=True),
                docs_preview,
                downloads,
                "",
                gr.update(visible=False),
            )

        except Exception as e:
            generation_status = f"❌ Generation failed: {e}"
            yield (
                init_status,
                analysis_status,
                generation_status,
                "",
                metrics,
                details,
                dependencies,
                gr.update(visible=True),
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )
            return

        # Step 4: Compare Documentation
        yield (
            init_status,
            analysis_status,
            generation_status,
            "🔄 Comparing with existing documentation...",
            metrics,
            details,
            dependencies,
            gr.update(visible=True),
            docs_preview,
            downloads,
            "",
            gr.update(visible=False),
        )

        try:
            comparison_result = self.compare_documentation()
            comparison_content, comparison_visible = comparison_result

            if "❌" in comparison_content or "⚠️" in comparison_content:
                comparison_status = comparison_content.split("\n")[0]  # First line as status
            else:
                comparison_status = "✅ Comparison completed successfully!"

            # Show final results
            yield (
                init_status,
                analysis_status,
                generation_status,
                comparison_status,
                metrics,
                details,
                dependencies,
                gr.update(visible=True),
                docs_preview,
                downloads,
                comparison_content,
                comparison_visible,
            )

        except Exception as e:
            comparison_status = f"❌ Comparison failed: {e}"
            comparison_content = f"Error during comparison: {str(e)}"
            yield (
                init_status,
                analysis_status,
                generation_status,
                comparison_status,
                metrics,
                details,
                dependencies,
                gr.update(visible=True),
                docs_preview,
                downloads,
                comparison_content,
                gr.update(visible=True),
            )


def create_interface():
    """Create the Gradio interface with comparison features"""
    agent_interface = DocumentationAgentInterface()

    with gr.Blocks(
        title="LLM Documentation Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .section-box {
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
        }
        .step-number {
            background: #667eea;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
        }
        .comparison-metrics {
            background: #e8f4fd;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
        }
        """,
    ) as app:

        # Main Header
        with gr.Row():
            gr.HTML(
                """
                <div class="main-header">
                    <h1>🤖 LLM Documentation Agent</h1>
                    <p>Intelligent documentation generation and analysis for software repositories</p>
                </div>
            """
            )

        # Example Walkthrough Section
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-box"><h2>🚀 Example Walkthrough</h2></div>')
                walkthrough_btn = gr.Button("📖 Show Example Use Case", variant="secondary")
                walkthrough_display = gr.Markdown("")

        # Step 1: Configuration
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-box"><h2><span class="step-number">1</span>Configuration</h2></div>')

                with gr.Row():
                    with gr.Column(scale=2):
                        api_key = gr.Textbox(
                            label="🔑 OpenAI API Key",
                            type="password",
                            placeholder="sk-...",
                            info="Your OpenAI API key for LLM access",
                        )
                        repo_path = gr.Textbox(
                            label="📁 Repository Path or GitHub URL",
                            value="python-sdk",
                            placeholder="path/to/your/repository OR https://github.com/user/repo",
                            info="Local path to repository OR GitHub URL (will be cloned automatically)",
                        )

                    with gr.Column(scale=1):
                        model_name = gr.Dropdown(
                            label="🧠 Model",
                            choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini"],
                            value="gpt-4o-mini",
                            info="Choose the LLM model",
                        )

                with gr.Row():
                    gr.Markdown("**📚 Documentation Types to Generate:**")
                with gr.Row():
                    readme_check = gr.Checkbox(label="README", value=True)
                    api_docs_check = gr.Checkbox(label="API Documentation", value=True)
                    tutorial_check = gr.Checkbox(label="Tutorial", value=True)
                    architecture_check = gr.Checkbox(label="Architecture", value=True)

                init_btn = gr.Button("🚀 Start Complete Workflow", variant="primary", size="lg")
                init_status = gr.Markdown("")

        # Step 2: Analysis
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-box"><h2><span class="step-number">2</span>Repository Analysis</h2></div>')
                analysis_status = gr.Markdown("⏳ Waiting for initialization...")

        with gr.Row(visible=False) as analysis_results:
            with gr.Column(scale=1):
                metrics_output = gr.Markdown("")
            with gr.Column(scale=1):
                details_output = gr.Markdown("")
            with gr.Column(scale=1):
                dependencies_output = gr.Markdown("")

        # Step 3: Generation
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    '<div class="section-box"><h2><span class="step-number">3</span>Documentation Generation</h2></div>'
                )
                generation_status = gr.Markdown("⏳ Waiting for analysis...")

        # Step 4: Comparison
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    '<div class="section-box"><h2><span class="step-number">4</span>Documentation Comparison</h2></div>'
                )
                comparison_status = gr.Markdown("⏳ Waiting for documentation generation...")

        # Results and Download Section
        with gr.Row():
            with gr.Column():
                docs_preview = gr.Markdown("", label="Documentation Preview")
                downloads_area = gr.File(label="📥 Download Generated Files", file_count="multiple", visible=False)

        # Comparison Results Section
        with gr.Row(visible=True) as comparison_section:
            with gr.Column():
                gr.HTML('<div class="section-box"><h2>⚖️ Comparison Results & Similarity Analysis</h2></div>')
                comparison_results_display = gr.Markdown("", label="Comparison Analysis")

        # Help Section
        with gr.Row():
            with gr.Column():
                with gr.Accordion("❓ Help & Instructions", open=False):
                    gr.Markdown(
                        """
                    ## 🚀 How to Use
                    
                    **Step 1:** Configure your settings
                    - Enter your OpenAI API key
                    - Set repository path or GitHub URL
                    - Choose model and documentation types
                    
                    **Step 2:** Analyze your repository
                    - The system will automatically clone GitHub repos
                    - Review metrics and detected dependencies
                    
                    **Step 3:** Generate documentation
                    - Click "Generate Documentation" to create docs
                    - Preview and download your documentation
                    
                    ## 📁 Repository Input Options
                    - **Local path**: `./my-project` or `/path/to/repo`
                    - **GitHub HTTPS**: `https://github.com/username/repo`
                    - **GitHub SSH**: `git@github.com:username/repo.git`
                    - **Short format**: `github.com/username/repo`
                    
                    ## 💡 Model Recommendations
                    - **GPT-4o**: Best quality, excellent for complex analysis
                    - **GPT-4o-mini**: Cost-effective, good for simple docs
                    - **GPT-4.1-mini**: Enhanced capabilities
                    - **o4-mini**: Lightweight for basic documentation
                    """
                    )

        # Event handlers
        walkthrough_btn.click(agent_interface.run_example_walkthrough, outputs=[walkthrough_display])

        # Progressive workflow that updates UI after each step
        def progressive_workflow(*inputs):
            """Generator function for progressive updates"""
            for result in agent_interface.run_progressive_workflow(*inputs):
                yield result

        init_btn.click(
            progressive_workflow,
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
                analysis_status,
                generation_status,
                comparison_status,
                metrics_output,
                details_output,
                dependencies_output,
                analysis_results,
                docs_preview,
                downloads_area,
                comparison_results_display,
                comparison_section,
            ],
            show_progress="hidden",  # This hides the default progress bars
        )

    return app


# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, debug=True)
