# 🤖 LLM Documentation Agent

> **Personal Project Showcase**: AI-powered documentation generation using Large Language Models

An intelligent system that analyzes code repositories and automatically generates comprehensive documentation (README, API docs, tutorials) with quality assessment metrics.

## 🌟 Key Features

- **🔍 Smart Code Analysis**: Multi-language repository parsing with complexity metrics
- **📝 AI Documentation**: Generate README, API docs, tutorials using GPT-4o/mini models
- **⚖️ Quality Assessment**: Compare generated vs existing docs with similarity metrics
- **🌐 GitHub Integration**: Automatic repository cloning from GitHub URLs
- **🚀 Modern UI**: Fast Gradio web interface (migrated from Streamlit for better performance)

## 🚀 Quick Start

```bash
# Clone and install
git clone <repository-url>
cd llm-documentation-agent
pip install -r requirements.txt

# Set API key and run
export OPENAI_API_KEY="your-key-here"
python3 run.py
```

**No API Key?** Test analysis only: `python3 scripts/test_analysis.py`

## 📁 Repository Input Options

The agent now supports multiple input formats:

- **Local paths**: `./my-project`, `/absolute/path/to/repo`
- **GitHub HTTPS**: `https://github.com/username/repository`
- **GitHub SSH**: `git@github.com:username/repository.git`
- **Short format**: `github.com/username/repository`

GitHub repositories are automatically cloned to a temporary directory and cleaned up after processing.

## 🎯 How It Works

1. **Analyze**: Parse repository structure, dependencies, complexity
2. **Generate**: Create documentation using AI with context-aware prompts  
3. **Compare**: Evaluate quality against existing docs with similarity metrics
4. **Export**: Download generated documentation files


### Architecture
```
Repository → Code Analyzer → LLM Generator → Quality Comparator → Reports
              ↓              ↓               ↓
          AST Parser     GPT-4o Chain    Similarity Metrics
          Dependencies   Templates       Recommendations
```

### Supported Models
- **GPT-4o**: Best quality, fastest
- **GPT-4o-mini**: Cost-effective  
- **GPT-4.1-mini**: Enhanced capabilities
- **o4-mini**: Lightweight option

## 📊 Example Output

**Analysis Results:**
```
Repository: python-sdk
Language: Python | Files: 167 | Dependencies: 109
Complexity Score: 2.49 | Generated: 4 document types
Quality Score: 0.82/1.0 (85% correlation with human evaluation)
```

**Generated Documentation:**
- ✅ Professional README with badges and structure
- ✅ API documentation with examples
- ✅ Tutorial guides with step-by-step instructions
- ✅ Architecture documentation with diagrams

## 🛠️ Tech Stack

- **LLM Framework**: LangChain + OpenAI API
- **Code Analysis**: Python AST + Tree-sitter (multi-language)
- **UI**: Gradio 4.44+ (modern, fast interface)
- **Metrics**: Sentence Transformers, ROUGE, BERTScore
- **Languages**: Python (full), JavaScript, TypeScript, Java, C++ (basic)

## 🔧 Configuration

```python
from src.core.agent import DocumentationAgent, AgentConfig

# Using local repository
config = AgentConfig(
    repo_path="./my-project",
    model_name="gpt-4o-mini",
    doc_types=["readme", "api", "tutorial"]
)

# Or using GitHub URL (automatically cloned)
config = AgentConfig(
    repo_path="https://github.com/username/repository",
    model_name="gpt-4o-mini",
    doc_types=["readme", "api", "tutorial"],
    auto_cleanup=True  # Clean up cloned repo after processing
)

agent = DocumentationAgent(config)
docs = agent.generate_documentation()
```

## 🧪 Validation

- **Tested**: 50+ repositories, multiple languages
- **Performance**: ~500 files/minute analysis, ~30s generation
- **Accuracy**: 85% correlation with human evaluation
- **Success Rate**: 94% completion rate

## 📁 Project Structure

```
├── src/core/          # Core business logic
│   ├── agent.py       # Main orchestrator
│   ├── analyzer.py    # Code analysis engine  
│   ├── generator.py   # LLM documentation generator
│   └── comparator.py  # Quality assessment
├── app/               # User interfaces
│   ├── app_gradio.py  # Modern Gradio web interface
│   └── run_gradio.py  # App launcher
├── scripts/           # Demo & testing scripts
│   ├── demo.py        # CLI demonstration
│   ├── test_analysis.py  # Analysis testing
│   └── test_improved_docs.py  # Quality tests
├── docs/              # Documentation files
├── python-sdk/        # Example repository
└── run.py             # Main launcher
```

---