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
from dataclasses import asdict

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
# 🚀 Example Walkthrough: Comprehensive Documentation Generation

## 🎯 How It Works
The agent creates a single, comprehensive documentation file that combines all aspects of your project:

### Flow:
`Code Analysis + README Analysis → Comprehensive Document Generation → README Comparison`

## 📋 Try These Examples:
- **Flask Framework**: `https://github.com/pallets/flask`
- **FastAPI**: `https://github.com/tiangolo/fastapi`
- **Django**: `https://github.com/django/django`

## 🔄 What the Agent Does:

### 1. 🔍 **Smart Analysis**
- Analyzes your entire codebase structure
- Identifies all modules, classes, and functions
- Extracts existing documentation and docstrings
- Finds and reads your original README

### 2. 📚 **Comprehensive Generation**
- Creates a single, unified documentation file
- Includes all essential sections in one place:
  - Overview and features
  - Installation instructions
  - Complete API reference
  - Usage examples
  - Architecture details
  - Configuration guides
  - Troubleshooting
  - Contributing guidelines

### 3. 🧠 **Context-Aware Enhancement**
- Uses your original README as context
- Maintains your documentation style
- Fills gaps in existing documentation
- Expands on brief descriptions

### 4. ⚖️ **README Comparison**
- Compares the new comprehensive doc with your original README
- Shows coverage improvements
- Identifies new sections added
- Provides similarity metrics

## ✨ Expected Output:
- **One comprehensive document** (10,000+ words typically)
- **Complete project documentation** in a single file
- **README comparison report** showing improvements
- **Professional markdown formatting** ready for use

## 📈 Benefits:
- **All-in-one**: No need to maintain multiple doc files
- **Comprehensive**: Covers every aspect of your project
- **Consistent**: Unified style throughout
- **README-aware**: Builds upon your existing documentation

*Perfect for creating complete project documentation from any codebase!*
"""
        return walkthrough_md

    def initialize_agent(self, api_key, model_name, repo_path):
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

        # Use a single comprehensive document type
        doc_types = ["comprehensive"]  # Single comprehensive document

        try:
            config = AgentConfig(
                repo_path=repo_path,
                openai_api_key=api_key,
                model_name=model_name,
                doc_types=doc_types,
                output_dir="gradio_output",
                include_comparison=True,
                use_enhanced_generator=True,  # Explicitly enable enhanced generator
                auto_discover_docs=True,  # Enable auto-discovery of existing docs
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

    def generate_documentation(self):
        """Generate documentation"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", "", gr.update(visible=False)

        try:
            if self.agent.config.use_enhanced_generator:
                print("🚀 Starting enhanced documentation generation pipeline...")

                if not hasattr(self.agent, "repository_metadata") or not self.agent.repository_metadata:
                    print("📊 Running repository analysis first...")
                    self.repo_metadata, self.modules = self.agent.analyze_repository()

                result = self.agent.run_enhanced_pipeline()
                generated_docs = result["documents"]
                metadata = result["metadata"]

                repo_info = ""
                if hasattr(self.agent, "repository_metadata") and self.agent.repository_metadata:
                    repo_info = f"""
## 📊 Repository Analysis
- **Name:** {self.agent.repository_metadata.name}
- **Language:** {self.agent.repository_metadata.language}
- **Files:** {self.agent.repository_metadata.file_count:,}
- **Size:** {self.agent.repository_metadata.size:,} bytes
- **Modules:** {len(self.modules) if hasattr(self, 'modules') else 0}
- **Dependencies:** {len(self.agent.repository_metadata.dependencies)}
"""

                preview_content = f"""# 📚 Comprehensive Documentation Generated
{repo_info}
## 📄 Generated Document Preview
"""
                if generated_docs:
                    doc = generated_docs[0]
                    preview_content += f"### {doc.title} ({doc.word_count:,} words)\n"
                    preview_content += f"{doc.content[:2000]}{'...' if len(doc.content) > 2000 else ''}\n\n"

                file_outputs = []
                doc_path = os.path.join(self.temp_dir, "readme_summarized.md")
                with open(doc_path, "w", encoding="utf-8") as f:
                    if generated_docs:
                        f.write(f"# {generated_docs[0].title}\n\n{generated_docs[0].content}")
                file_outputs.append(doc_path)

                metadata_path = os.path.join(self.temp_dir, "generation_metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump({"metadata": metadata, "documents": [asdict(d) for d in generated_docs]}, f, indent=2)
                file_outputs.append(metadata_path)

                self.result = result
                self.generated_docs = generated_docs
                self.agent.generated_docs = generated_docs

                if "metadata" not in self.result:
                    self.result["metadata"] = metadata

            else:
                # Standard generator path (simplified)
                print("🚀 Starting standard documentation generation pipeline...")
                if not hasattr(self.agent, "repository_metadata") or not self.agent.repository_metadata:
                    self.repo_metadata, self.modules = self.agent.analyze_repository()
                self.generated_docs = self.agent.generate_documentation()
                preview_content = "# 📚 Generated Documentation\n\n"
                file_outputs = []
                for doc in self.generated_docs:
                    preview_content += f"## {doc.doc_type.title()} Documentation\n{doc.content[:1500]}...\n---\n"
                    temp_path = os.path.join(self.temp_dir, f"{doc.doc_type}_doc.md")
                    with open(temp_path, "w", "utf-8") as f:
                        f.write(doc.content)
                    file_outputs.append(temp_path)

            return (
                "✅ Documentation generated successfully!",
                preview_content,
                gr.update(value=file_outputs, visible=True),
            )

        except Exception as e:
            import traceback

            error_details = f"❌ Documentation generation failed: {e}\n\nDetails:\n{traceback.format_exc()}"
            return error_details, "", gr.update(visible=False)

    def compare_documentation(self):
        """Compare generated documentation with original README"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", gr.update(visible=False)

        if not self.generated_docs:
            return "⚠️ Warning: Please generate documentation first", gr.update(visible=False)

        try:
            print("🔄 Running documentation comparison...")
            comparison_results = self.agent.compare_with_existing()
            self.agent.comparison_results = comparison_results
            print(f"✅ Comparison completed for {len(comparison_results)} document types")

            if not comparison_results:
                return """# 📊 README Comparison
## ℹ️ No Original README Found
The agent could not find an existing README.md to compare against. The generated documentation is based solely on code analysis.
""", gr.update(
                    visible=True
                )

            result = comparison_results.get("comprehensive")
            if not result:
                return "Could not find comparison results for 'comprehensive' document.", gr.update(visible=True)

            quality_scores = [
                result.metrics.semantic_similarity,
                result.metrics.bert_score,
                result.metrics.readability_score,
                result.metrics.word_count_ratio,
                result.metrics.rouge_scores.get("rouge1", 0),
                result.metrics.rouge_scores.get("rougeL", 0),
            ]
            if result.metrics.ragas_relevancy is not None:
                quality_scores.append(result.metrics.ragas_relevancy)
            if result.metrics.ragas_correctness is not None:
                quality_scores.append(result.metrics.ragas_correctness)

            if quality_scores:
                overall_score = sum(quality_scores) / len(quality_scores)
            else:
                overall_score = 0

            comparison_content = f"""
# 🎯 Overall Comparison Summary
**Average Quality Score:** {overall_score:.3f} / 1.000
{'🎉 Excellent alignment!' if overall_score > 0.8 else '👍 Good alignment!' if overall_score > 0.6 else '⚠️ Moderate alignment' if overall_score > 0.4 else '❌ Low alignment'}
---
## Comprehensive Documentation vs Original README
### 📊 Similarity Metrics
- **Semantic Similarity:** {result.metrics.semantic_similarity:.3f}
- **ROUGE-1 Score:** {result.metrics.rouge_scores.get('rouge1', 0):.3f}
- **ROUGE-L Score:** {result.metrics.rouge_scores.get('rougeL', 0):.3f}
- **Word Count Ratio:** {result.metrics.word_count_ratio:.3f}
"""
            if result.metrics.ragas_relevancy is not None:
                comparison_content += f"- **Ragas Answer Relevancy:** {result.metrics.ragas_relevancy:.3f}\n"
            if result.metrics.ragas_correctness is not None:
                comparison_content += f"- **Ragas Answer Correctness:** {result.metrics.ragas_correctness:.3f}\n"

            return comparison_content, gr.update(visible=True)

        except Exception as e:
            import traceback

            return f"❌ Comparison failed: {e}\n\nDetails:\n{traceback.format_exc()}", gr.update(visible=False)

    def full_workflow(self, api_key, model_name, repo_path):
        """Complete workflow: Initialize -> Analyze -> Generate -> Compare"""
        # Step 1: Initialize Agent
        init_status, _, _, _, analysis_visible, generation_visible = self.initialize_agent(
            api_key, model_name, repo_path
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

        # Step 3: Generate Documentation
        generation_status = "🔄 Generating documentation..."
        try:
            generation_result = self.generate_documentation()
            generation_status, docs_preview, downloads = generation_result

            if "❌" in generation_status:
                return (
                    init_status,
                    # analysis_status,
                    generation_status,
                    "⏸️ Workflow stopped due to generation error",
                    # metrics,
                    # details,
                    # dependencies,
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
                # analysis_status,
                generation_status,
                # metrics,
                # details,
                # dependencies,
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
            # analysis_status,
            generation_status,
            comparison_status,
            # metrics,
            # details,
            # dependencies,
            gr.update(visible=True),
            docs_preview,
            downloads,
            comparison_content,
            comparison_visible,
        )

    def run_progressive_workflow(self, api_key, model_name, repo_path):
        """Progressive workflow that yields results after each step"""

        # Step 1: Initialize Agent
        yield (
            "🔄 Initializing agent...",
            "",
            "",
            gr.update(visible=False),
            "",
            "",
        )

        try:
            init_result = self.initialize_agent(api_key, model_name, repo_path)
            init_status = init_result[0]

            if "❌" in init_status:
                yield (
                    init_status,
                    "",
                    "",
                    gr.update(visible=False),
                    "",
                    "",
                )
                return
        except Exception as e:
            yield (
                f"❌ Initialization failed: {e}",
                "",
                "",
                gr.update(visible=False),
                "",
                "",
            )
            return

        # Step 2: Generate Documentation (analysis happens internally)
        yield (
            init_status,
            "🔄 Generating documentation...",
            "",
            gr.update(visible=False),
            "",
            "",
        )

        try:
            generation_result = self.generate_documentation()
            generation_status, docs_preview, downloads = generation_result

            if "❌" in generation_status:
                yield (
                    init_status,
                    generation_status,
                    "",
                    downloads,
                    "",
                    "",
                )
                return

            # Show generation results
            yield (
                init_status,
                generation_status,
                docs_preview,
                downloads,
                "",
                "",
            )

        except Exception as e:
            generation_status = f"❌ Generation failed: {e}"
            yield (
                init_status,
                generation_status,
                "",
                gr.update(visible=False),
                "",
                "",
            )
            return

        # Step 3: Compare Documentation
        yield (
            init_status,
            generation_status,
            docs_preview,
            downloads,
            "🔄 Comparing with original README...",
            "",
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
                generation_status,
                docs_preview,
                downloads,
                comparison_status,
                comparison_content,
            )

        except Exception as e:
            comparison_status = f"❌ Comparison failed: {e}"
            comparison_content = f"Error during comparison: {str(e)}"
            yield (
                init_status,
                generation_status,
                docs_preview,
                downloads,
                comparison_status,
                comparison_content,
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
                with gr.Accordion("📖 Show Example Use Case", open=False):
                    gr.Markdown(agent_interface.run_example_walkthrough())

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

                # Remove document type checkboxes - we'll generate one comprehensive document
                gr.Markdown("**📚 Documentation Generation:**")
                gr.Markdown(
                    "The agent will generate a comprehensive documentation that includes all aspects: overview, API reference, tutorials, and architecture details."
                )

                init_btn = gr.Button("🚀 Start Complete Workflow", variant="primary", size="lg")
                init_status = gr.Markdown("")

        # Step 2: Generation
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    '<div class="section-box"><h2><span class="step-number">2</span>Documentation Generation</h2></div>'
                )
                generation_status = gr.Markdown("")

        # Step 3: Comparison
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-box"><h2><span class="step-number">3</span>README Comparison</h2></div>')
                comparison_status = gr.Markdown("")

        # Results and Download Section
        with gr.Row():
            with gr.Column():
                docs_preview = gr.Markdown("", label="Documentation Preview")
                downloads_area = gr.File(label="📥 Download Generated Files", file_count="multiple", visible=False)

        # Comparison Results Section (content only, no redundant header)
        with gr.Row(visible=True) as comparison_section:
            with gr.Column():
                comparison_results_display = gr.Markdown("", label="README Comparison Analysis")

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
                    - Choose your preferred model
                    
                    **Step 2:** Generate comprehensive documentation
                    - Click "Start Complete Workflow" to create a unified document
                    - Preview and download your documentation
                    
                    **Step 3:** Review README comparison
                    - View how the new documentation compares to the original README
                    - See similarity scores and coverage metrics
                    - Get recommendations for improvement
                    
                    ## 📄 What Gets Generated
                    A single comprehensive document that includes:
                    - Project overview and features
                    - Installation and setup instructions
                    - Complete API reference
                    - Usage examples and tutorials
                    - Architecture documentation
                    - Configuration guides
                    - Troubleshooting and FAQ
                    - Contributing guidelines
                    
                    ## 📁 Repository Input Options
                    - **Local path**: `./my-project` or `/path/to/repo`
                    - **GitHub HTTPS**: `https://github.com/username/repo`
                    - **GitHub SSH**: `git@github.com:username/repo.git`
                    - **Short format**: `github.com/username/repo`
                    
                    ## 💡 Model Recommendations
                    - **GPT-4o**: Good quality, good for most projects
                    - **GPT-4o-mini**: Cost-effective
                    - **GPT-4.1-mini**: Improved capabilities
                    - **o4-mini**: Reasoning model for complex tasks
                    
                    ## 📊 README Comparison
                    The system automatically compares your generated documentation with the original README to show:
                    - Content coverage and gaps filled
                    - Similarity metrics
                    - Sections added beyond the original
                    - Style consistency analysis
                    """
                    )

        # Event handlers
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
            ],
            outputs=[
                init_status,
                generation_status,
                docs_preview,
                downloads_area,
                comparison_status,
                comparison_results_display,
            ],
            show_progress="hidden",  # This hides the default progress bars
        )

    return app


# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, debug=True)
