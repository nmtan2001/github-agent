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

## 🎯 How It Works
The agent uses an intelligent pipeline that reads and incorporates existing documentation:

### Flow:
`Code Analysis + README Analysis → Context-Aware LLM Generation → Compare/Improve`

## 📋 Try These Examples:
- **Flask Framework**: `https://github.com/pallets/flask`
- **FastAPI**: `https://github.com/tiangolo/fastapi`
- **Django**: `https://github.com/django/django`

## 🔄 What the Agent Does:

### 1. 🔍 **Smart Discovery**
- Automatically finds README.md, docs/ folders, wiki files
- Supports multiple formats (.md, .rst, .txt, .adoc)
- Excludes irrelevant files (.git, node_modules, etc.)

### 2. 📚 **Context Integration**
- Reads and chunks existing documentation
- Creates semantic embeddings for similarity search
- Identifies documentation patterns and styles

### 3. 🧠 **Context-Aware Generation**
- Uses context-aware LLM prompts
- Incorporates existing documentation patterns
- Maintains style consistency
- Fills gaps while respecting existing structure

### 4. ⚖️ **Intelligent Comparison**
- Shows how much existing documentation was found
- Explains context integration used in generation
- Provides comprehensive analysis and recommendations

## ✨ Expected Outputs:
- **Context-aware documentation** that builds on existing content
- **Style-consistent** with your current documentation
- **Gap-filling** that complements rather than duplicates
- **Smart discovery** summary of existing documentation found

*Works great with any repository that has existing documentation!*
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
                include_comparison=True,
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
            # Check if using enhanced generator or standard generator
            if self.agent.config.use_enhanced_generator:
                # Generate documentation with context-aware pipeline
                print("🚀 Starting enhanced documentation generation pipeline...")
                result = self.agent.run_enhanced_pipeline()

                # Extract the generated documents (now returns List[GeneratedDocument])
                generated_docs = result["documents"]
                metadata = result["metadata"]

                # Format results for display
                preview_content = f"""# 📚 Documentation Generated with README Enhancement

## 📊 Generation Summary
- **Repository:** {metadata.get('repo_path', 'N/A')}
- **Total Documents:** {metadata.get('total_documents', 0)}
- **Enhanced with Embeddings:** {metadata.get('enhanced_with_embeddings', False)}
- **Existing Docs Used:** {metadata.get('existing_docs_loaded', False)}
- **Generation Method:** {metadata.get('generation_method', 'standard')}

## 📚 Generated Documents

"""

                # Add each document to the preview
                for doc in generated_docs:
                    preview_content += f"### {doc.doc_type.title()} Documentation ({doc.word_count} words)\n"
                    preview_content += f"{doc.content[:1000]}{'...' if len(doc.content) > 1000 else ''}\n\n---\n\n"

                preview_content += """
## ✅ Features Used:
- ✅ **Code Analysis:** Repository structure and dependencies analyzed
- ✅ **README Embeddings:** Existing documentation vectorized and used for context
- ✅ **Context Integration:** Existing documentation patterns incorporated
- ✅ **Multi-Document Generation:** Separate documents for each type
- ✅ **Enhanced Prompts:** LLM prompts enriched with README content
"""

                # Create temporary files for download
                file_outputs = []

                # Save each document as a separate file
                for doc in generated_docs:
                    doc_path = os.path.join(self.temp_dir, f"{doc.doc_type}_documentation.md")
                    with open(doc_path, "w", encoding="utf-8") as f:
                        f.write(f"# {doc.title}\n\n{doc.content}")
                    file_outputs.append(doc_path)

                # Save metadata
                metadata_path = os.path.join(self.temp_dir, "generation_metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump(
                        {
                            "metadata": metadata,
                            "documents": [
                                {
                                    "title": doc.title,
                                    "doc_type": doc.doc_type,
                                    "word_count": doc.word_count,
                                    "metadata": doc.metadata,
                                }
                                for doc in generated_docs
                            ],
                        },
                        f,
                        indent=2,
                    )
                file_outputs.append(metadata_path)

                # Store the results for comparison
                self.result = result
                self.generated_docs = generated_docs  # Store list of GeneratedDocument objects

            else:
                # Use standard pipeline that generates multiple documents
                print("🚀 Starting standard documentation generation pipeline...")

                # First analyze if not already done
                if not hasattr(self.agent, "repository_metadata") or not self.agent.repository_metadata:
                    self.repo_metadata, self.modules = self.agent.analyze_repository()

                # Generate multiple documents
                self.generated_docs = self.agent.generate_documentation()

                # Format generated documents for display
                doc_outputs = []
                preview_content = "# 📚 Generated Documentation\n\n"

                for doc in self.generated_docs:
                    preview_content += f"""
## {doc.doc_type.title()} Documentation
**Word count:** {doc.word_count}

### Content Preview:
{doc.content[:1500]}{'...' if len(doc.content) > 1500 else ''}

---

"""
                    doc_outputs.append((f"{doc.doc_type}_documentation.md", doc.content))

                # Create temporary files for download
                file_outputs = []
                for filename, content in doc_outputs:
                    temp_path = os.path.join(self.temp_dir, filename)
                    with open(temp_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    file_outputs.append(temp_path)

                # Store results for compatibility
                self.result = {
                    "documentation": preview_content,
                    "metadata": {
                        "repo_path": self.agent.config.repo_path,
                        "doc_types": [doc.doc_type for doc in self.generated_docs],
                        "documents_count": len(self.generated_docs),
                    },
                }

            return (
                "✅ Documentation generated successfully!",
                preview_content,
                gr.update(value=file_outputs, visible=True),
            )

        except Exception as e:
            import traceback

            error_details = f"❌ Documentation generation failed: {e}\n\nDetails:\n{traceback.format_exc()}"
            return error_details, "", gr.update(visible=False)

    def analyze_repository(self):
        """Analyze repository (compatibility method for workflows that expect separate analysis)"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", "", "", "", gr.update(visible=False)

        try:
            # Use the agent's analyze_repository method
            self.repo_metadata, self.modules = self.agent.analyze_repository()

            # Format analysis results for display
            metrics_content = f"""
**Repository:** {self.repo_metadata.name or 'N/A'}
**Language:** {self.repo_metadata.language}
**Files:** {self.repo_metadata.file_count:,}
**Size:** {self.repo_metadata.size:,} bytes
**Complexity:** {self.repo_metadata.complexity_score:.2f}
"""

            details_content = f"""
## 📊 Repository Analysis Results

### 🏗️ Structure Overview
- **Name:** {self.repo_metadata.name or 'N/A'}
- **Primary Language:** {self.repo_metadata.language}
- **Total Files:** {self.repo_metadata.file_count:,}
- **Repository Size:** {self.repo_metadata.size:,} bytes
- **Complexity Score:** {self.repo_metadata.complexity_score:.2f}/5.0

### 📝 Description
{self.repo_metadata.description or 'No description available'}

### 🗂️ Modules Found
{len(self.modules)} modules analyzed:
"""

            for i, module in enumerate(self.modules[:10], 1):  # Show first 10 modules
                details_content += f"""
**{i}. {module.name}**
- Path: `{module.path}`
- Functions: {len(module.functions)}
- Classes: {len(module.classes)}
- Lines of Code: {module.lines_of_code}
- Complexity: {module.complexity_score:.2f}
"""

            if len(self.modules) > 10:
                details_content += f"\n... and {len(self.modules) - 10} more modules"

            dependencies_content = f"""
## 📦 Dependencies ({len(self.repo_metadata.dependencies)})

"""
            for i, dep in enumerate(self.repo_metadata.dependencies[:20], 1):  # Show first 20 deps
                dependencies_content += f"{i}. `{dep}`\n"

            if len(self.repo_metadata.dependencies) > 20:
                dependencies_content += f"\n... and {len(self.repo_metadata.dependencies) - 20} more dependencies"

            return (
                "✅ Repository analysis completed!",
                metrics_content,
                details_content,
                dependencies_content,
                gr.update(visible=True),
            )

        except Exception as e:
            import traceback

            error_details = f"❌ Analysis failed: {e}\n\nDetails:\n{traceback.format_exc()}"
            return error_details, "", "", "", gr.update(visible=False)

    def compare_documentation(self):
        """Compare generated documentation with existing documentation"""
        if not self.agent:
            return "⚠️ Warning: Please initialize the agent first", gr.update(visible=False)

        if not hasattr(self, "result") or not self.result:
            return "⚠️ Warning: Please generate documentation first", gr.update(visible=False)

        try:
            # Try to get actual comparison results from agent
            if hasattr(self.agent, "comparison_results") and self.agent.comparison_results:
                comparison_results = self.agent.comparison_results

                comparison_content = "# 📊 Documentation Comparison Results\n\n"

                # Generate metrics summary
                for doc_type, result in comparison_results.items():
                    metrics = result.metrics
                    comparison_content += f"""## {doc_type.title()} Documentation

### 📈 Similarity Metrics
- **Semantic Similarity:** {metrics.semantic_similarity:.1%}
- **Structural Similarity:** {metrics.structural_similarity:.1%}
- **Content Coverage:** {metrics.content_coverage:.1%}
- **ROUGE-1:** {metrics.rouge_scores.get('rouge1', 0):.1%}
- **ROUGE-L:** {metrics.rouge_scores.get('rougeL', 0):.1%}
- **BERTScore:** {metrics.bert_score:.1%}
- **Word Count Ratio:** {metrics.word_count_ratio:.2f}

### 🎯 Analysis
{result.detailed_analysis.get('summary', 'No detailed analysis available')}

### 💡 Recommendations
"""
                    for rec in result.recommendations[:3]:  # Show top 3 recommendations
                        comparison_content += f"- {rec}\n"

                    if result.missing_sections:
                        comparison_content += f"\n**Missing Sections:** {', '.join(result.missing_sections)}\n"

                    if result.additional_sections:
                        comparison_content += f"**Additional Sections:** {', '.join(result.additional_sections)}\n"

                    comparison_content += "\n---\n\n"

                # Save comparison results to file
                comparison_file = os.path.join(self.temp_dir, "comparison_results.json")
                with open(comparison_file, "w", encoding="utf-8") as f:
                    import json
                    from dataclasses import asdict

                    serializable_results = {}
                    for doc_type, result in comparison_results.items():
                        serializable_results[doc_type] = asdict(result)

                    json.dump(serializable_results, f, indent=2, default=str)

                return (
                    comparison_content,
                    gr.update(visible=True),
                )

            else:
                # Fall back to context integration summary
                metadata = self.result.get("metadata", {})
                existing_docs_count = metadata.get("existing_docs_count", 0)

                if existing_docs_count == 0:
                    return (
                        "ℹ️ No existing documentation found - comparison not available",
                        gr.update(visible=False),
                    )

                comparison_content = f"""# 📊 Context Integration Summary

## 📚 Existing Documentation Analysis
- **Existing Documentation Found:** {existing_docs_count} chunks
- **Context Used in Generation:** ✅ Yes

### 🎯 Integration Features Used
- **Smart Discovery:** Automatically found and analyzed existing documentation
- **Context-Aware Generation:** LLM prompts enriched with existing content patterns
- **Style Preservation:** Maintained consistency with existing documentation style
- **Gap Filling:** Generated content fills gaps while respecting existing structure

*Note: For detailed comparison metrics, ensure the agent's comparison feature is enabled.*
"""

                return (
                    comparison_content,
                    gr.update(visible=True),
                )

        except Exception as e:
            import traceback

            error_details = f"❌ Comparison failed: {e}\n\nDetails:\n{traceback.format_exc()}"
            return error_details, gr.update(visible=False)

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
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )
            return

        # Step 2: Generate Documentation (skipping analysis)
        yield (
            init_status,
            "🔄 Generating documentation...",
            "",
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
                    generation_status,
                    "",
                    "",
                    downloads,
                    "",
                    gr.update(visible=False),
                )
                return

            # Show generation results immediately
            yield (
                init_status,
                generation_status,
                "",
                docs_preview,
                downloads,
                "",
                gr.update(visible=False),
            )

        except Exception as e:
            generation_status = f"❌ Generation failed: {e}"
            yield (
                init_status,
                generation_status,
                "",
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            )
            return

        # Step 3: Compare Documentation
        yield (
            init_status,
            generation_status,
            "🔄 Comparing with existing documentation...",
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
                generation_status,
                comparison_status,
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
                generation_status,
                comparison_status,
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
                gr.HTML(
                    '<div class="section-box"><h2><span class="step-number">3</span>Documentation Comparison</h2></div>'
                )
                comparison_status = gr.Markdown("")

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
                    
                    **Step 2:** Generate documentation
                    - Click "Start Complete Workflow" to create docs
                    - Preview and download your documentation
                    
                    **Step 3:** Review comparison results
                    - View similarity scores and quality metrics
                    - Get recommendations for improvement
                    
                    ## 📁 Repository Input Options
                    - **Local path**: `./my-project` or `/path/to/repo`
                    - **GitHub HTTPS**: `https://github.com/username/repo`
                    - **GitHub SSH**: `git@github.com:username/repo.git`
                    - **Short format**: `github.com/username/repo`
                    
                    ## 💡 Model Recommendations
                    - **GPT-4o**: Best quality, excellent for complex analysis
                    - **GPT-4o-mini**: Cost-effective, good for simple docs
                    - **GPT-4.1-mini**: Improved capabilities
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
                generation_status,
                comparison_status,
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
