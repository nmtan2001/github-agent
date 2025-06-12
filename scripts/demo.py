#!/usr/bin/env python3
"""
Demo script for LLM Documentation Agent

This script demonstrates the capabilities of the documentation agent
by analyzing the python-sdk repository and generating documentation.
"""

import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent import DocumentationAgent, AgentConfig


def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🤖 LLM DOCUMENTATION AGENT DEMO")
    print("=" * 80)
    print("This demo will analyze the python-sdk repository and generate")
    print("comprehensive documentation using GPT-4o-mini.")
    print("=" * 80)
    print()


def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*20} {title} {'='*20}")


def demo_repository_analysis():
    """Demonstrate repository analysis"""
    print_section("REPOSITORY ANALYSIS")

    # Repository path - can be local path or GitHub URL
    repo_path = "python-sdk"  # Use relative path like the working test script

    # Alternative examples using GitHub URLs (comment/uncomment to test):
    # repo_path = "https://github.com/octocat/Hello-World"
    # repo_path = "https://github.com/pallets/flask.git"
    # repo_path = "git@github.com:user/repository.git"

    # Check if it's a GitHub URL or local path
    from src.core.agent import AgentConfig

    if AgentConfig.is_github_url(repo_path):
        print(f"🌐 Using GitHub repository: {repo_path}")
        print("Repository will be cloned automatically...")
    else:
        if not Path(repo_path).exists():
            print(f"❌ Repository not found: {repo_path}")
            print("Please ensure the python-sdk directory is in the current location.")
            print("Or try using a GitHub URL like: https://github.com/octocat/Hello-World")
            return None, None
        print(f"✅ Found local repository: {repo_path}")

    # Configure agent
    config = AgentConfig(
        repo_path=repo_path,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        temperature=0,
        doc_types=["readme", "api", "tutorial", "architecture"],
        include_comparison=True,
        output_dir="demo_output",
    )

    # Initialize agent
    print("🔧 Initializing documentation agent...")
    agent = DocumentationAgent(config)

    # Analyze repository
    print("🔍 Analyzing repository structure...")
    start_time = time.time()

    try:
        repo_metadata, modules = agent.analyze_repository()
        analysis_time = time.time() - start_time

        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        print()

        # Display analysis results
        print("📊 ANALYSIS RESULTS:")
        print(f"  Repository Name: {repo_metadata.name}")
        print(f"  Language: {repo_metadata.language}")
        print(f"  Files: {repo_metadata.file_count}")
        print(f"  Modules: {len(modules)}")
        print(f"  Dependencies: {len(repo_metadata.dependencies)}")
        print(f"  Complexity Score: {repo_metadata.complexity_score:.2f}")
        print(f"  Size: {repo_metadata.size:,} bytes")

        if repo_metadata.description:
            print(f"  Description: {repo_metadata.description[:100]}...")

        print("\n📦 TOP DEPENDENCIES:")
        for dep in repo_metadata.dependencies[:10]:
            print(f"    • {dep}")

        print("\n🗂️ MODULE OVERVIEW:")
        for module in modules[:5]:  # Show first 5 modules
            print(f"    • {module.name} ({module.path})")
            print(f"      Functions: {len(module.functions)}, Classes: {len(module.classes)}")

        if len(modules) > 5:
            print(f"    ... and {len(modules) - 5} more modules")

        return agent, repo_metadata

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        return None, None


def demo_documentation_generation(agent: DocumentationAgent):
    """Demonstrate documentation generation"""
    print_section("DOCUMENTATION GENERATION")

    print("📝 Generating documentation using GPT-4o-mini...")
    print("This may take a few minutes depending on the repository size...")

    start_time = time.time()

    try:
        generated_docs = agent.generate_documentation()
        generation_time = time.time() - start_time

        print(f"✅ Documentation generated in {generation_time:.2f} seconds")
        print(f"📄 Generated {len(generated_docs)} documents")
        print()

        # Display generated documents summary
        print("📋 GENERATED DOCUMENTS:")
        for doc in generated_docs:
            print(f"  📖 {doc.doc_type.title()}")
            print(f"     Words: {doc.word_count}")
            print(f"     Confidence: {doc.confidence_score:.2f}")
            print(f"     Title: {doc.title}")

            # Show preview of content
            preview = doc.content[:200].replace("\n", " ")
            print(f"     Preview: {preview}...")
            print()

        return generated_docs

    except Exception as e:
        print(f"❌ Documentation generation failed: {str(e)}")
        return None


def demo_documentation_comparison(agent: DocumentationAgent):
    """Demonstrate documentation comparison"""
    print_section("DOCUMENTATION COMPARISON")

    print("⚖️ Comparing generated documentation with existing documentation...")

    try:
        comparison_results = agent.compare_with_existing()

        if comparison_results:
            print(f"✅ Comparison completed for {len(comparison_results)} document types")
            print()

            # Display comparison results
            print("📊 COMPARISON RESULTS:")

            overall_scores = []
            for doc_type, result in comparison_results.items():
                overall_score = (
                    result.metrics.semantic_similarity
                    + result.metrics.content_coverage
                    + result.metrics.structural_similarity
                ) / 3
                overall_scores.append(overall_score)

                print(f"  📄 {doc_type.title()}:")
                print(f"     Semantic Similarity: {result.metrics.semantic_similarity:.3f}")
                print(f"     Content Coverage: {result.metrics.content_coverage:.3f}")
                print(f"     Structural Similarity: {result.metrics.structural_similarity:.3f}")
                print(f"     Overall Score: {overall_score:.3f}")

                if result.recommendations:
                    print(f"     Top Recommendation: {result.recommendations[0]}")
                print()

            avg_score = sum(overall_scores) / len(overall_scores)
            print(f"🎯 AVERAGE QUALITY SCORE: {avg_score:.3f}")

        else:
            print("⚠️ No existing documentation found for comparison")

    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")


def demo_final_report(agent: DocumentationAgent):
    """Demonstrate final report generation"""
    print_section("FINAL REPORT")

    print("📊 Generating comprehensive report...")

    try:
        report = agent.generate_report()

        print("✅ Report generated successfully!")
        print()

        # Display report summary
        print("📋 EXECUTIVE SUMMARY:")
        print(f"  Repository: {report.repository_metadata.name}")
        print(f"  Language: {report.repository_metadata.language}")
        print(f"  Modules Analyzed: {report.modules_analyzed}")
        print(f"  Documents Generated: {len(report.generated_documents)}")
        print(f"  Success Rate: {report.success_rate:.1%}")
        print(f"  Execution Time: {report.execution_time:.2f} seconds")
        print()

        if report.recommendations:
            print("💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
            if len(report.recommendations) > 3:
                print(f"     ... and {len(report.recommendations) - 3} more recommendations")

        print()
        print("📁 All outputs saved to: demo_output/")
        print("   • Generated documentation files")
        print("   • Analysis results (JSON)")
        print("   • Comparison reports")
        print("   • Comprehensive final report")

    except Exception as e:
        print(f"❌ Report generation failed: {str(e)}")


def main():
    """Main demo function"""
    print_banner()

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        print("\nExample .env file:")
        print("OPENAI_API_KEY=sk-your-api-key-here")
        return

    print("✅ OpenAI API key found")

    # Step 1: Repository Analysis
    agent, repo_metadata = demo_repository_analysis()
    if not agent:
        return

    # Step 2: Documentation Generation
    generated_docs = demo_documentation_generation(agent)
    if not generated_docs:
        return

    # Step 3: Documentation Comparison
    demo_documentation_comparison(agent)

    # Step 4: Final Report
    demo_final_report(agent)

    print_section("DEMO COMPLETE")
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Review the generated documentation in demo_output/generated/")
    print("2. Check the comparison report in demo_output/comparison_report.md")
    print("3. Read the final summary in demo_output/report_summary.md")
    print("4. Try the Gradio interface: python3 run.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
