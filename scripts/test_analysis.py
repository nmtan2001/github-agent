#!/usr/bin/env python3
"""
Test script for repository analysis functionality

This script tests the code analysis features without requiring OpenAI API key.
"""

import os
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🔄 Loading analyzer module...")
try:
    from src.core.analyzer import CodeAnalyzer

    print("✅ Analyzer module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import analyzer: {e}")
    sys.exit(1)


def test_repository_analysis():
    """Test repository analysis functionality"""
    print("=" * 60)
    print("🔍 TESTING REPOSITORY ANALYSIS")
    print("=" * 60)

    # Check if we have a Python repository to analyze (use current directory)
    repo_path = "."
    print(f"📁 Analyzing repository: {repo_path}")
    print(f"🔄 Current working directory: {os.getcwd()}")

    try:
        # Initialize analyzer
        print("🔄 Initializing CodeAnalyzer...")
        analyzer = CodeAnalyzer(repo_path)
        print("✅ Code analyzer initialized")

        # Check if Git repo
        if analyzer.repo:
            print("✅ Git repository detected")
        else:
            print("⚠️  No Git repository found (analysis will continue)")

        # Analyze repository
        print("🔍 Starting repository analysis...")
        print("   📂 Scanning for source files...")
        start_time = time.time()

        repo_metadata, modules = analyzer.analyze_repository()

        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        print()

        # Display results
        print("📊 ANALYSIS RESULTS:")
        print(f"  Repository Name: {repo_metadata.name or 'N/A'}")
        print(f"  Language: {repo_metadata.language}")
        print(f"  Files: {repo_metadata.file_count}")
        print(f"  Modules: {len(modules)}")
        print(f"  Dependencies: {len(repo_metadata.dependencies)}")
        print(f"  Complexity Score: {repo_metadata.complexity_score:.2f}")
        print(f"  Size: {repo_metadata.size:,} bytes")

        if repo_metadata.description:
            print(f"  Description: {repo_metadata.description[:100]}...")

        print("\n📦 DEPENDENCIES FOUND:")
        for dep in repo_metadata.dependencies[:10]:
            print(f"    • {dep}")
        if len(repo_metadata.dependencies) > 10:
            print(f"    ... and {len(repo_metadata.dependencies) - 10} more")

        print("\n🗂️ MODULES ANALYZED:")
        for i, module in enumerate(modules[:5], 1):
            print(f"    {i}. {module.name} ({module.path})")
            print(f"       Functions: {len(module.functions)}, Classes: {len(module.classes)}")
            if module.docstring:
                preview = module.docstring[:80].replace("\n", " ")
                print(f"       Doc: {preview}...")

        if len(modules) > 5:
            print(f"    ... and {len(modules) - 5} more modules")

        # Test specific module details
        if modules:
            print(f"\n🔍 DETAILED VIEW OF FIRST MODULE: {modules[0].name}")
            module = modules[0]

            print(f"  Path: {module.path}")
            print(f"  Language: {module.language}")
            print(f"  Lines of Code: {module.lines_of_code}")
            print(f"  Complexity: {module.complexity_score:.2f}")

            if module.functions:
                print(f"\n  📋 FUNCTIONS ({len(module.functions)}):")
                for func in module.functions[:3]:
                    print(f"    • {func.name}({', '.join(func.parameters)})")
                    if func.docstring:
                        doc_preview = func.docstring[:60].replace("\n", " ")
                        print(f"      Doc: {doc_preview}...")

            if module.classes:
                print(f"\n  🏛️ CLASSES ({len(module.classes)}):")
                for cls in module.classes[:3]:
                    print(f"    • {cls.name}")
                    print(f"      Methods: {len(cls.methods)}")
                    if cls.docstring:
                        doc_preview = cls.docstring[:60].replace("\n", " ")
                        print(f"      Doc: {doc_preview}...")

        print("\n✅ Repository analysis test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print("🔍 Debug information:")
        print(f"   - Python version: {sys.version}")
        print(f"   - Working directory: {os.getcwd()}")
        print(f"   - Repository path: {repo_path}")

        import traceback

        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🧪 LLM Documentation Agent - Analysis Test")
    print("This script tests the repository analysis functionality.")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"📁 Working directory: {os.getcwd()}")
    print()

    success = test_repository_analysis()

    if success:
        print("\n🎉 All tests passed!")
        print("The analysis functionality is working correctly.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY to test full documentation generation")
        print("2. Run 'python3 scripts/demo.py' for the complete demo")
        print("3. Try 'python3 run.py' for the web interface")
    else:
        print("\n❌ Tests failed!")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()
