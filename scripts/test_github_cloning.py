#!/usr/bin/env python3
"""
Test script for GitHub URL cloning functionality

This script demonstrates the new automatic cloning feature that allows
users to provide GitHub URLs instead of local repository paths.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent import DocumentationAgent, AgentConfig


def test_github_url_detection():
    """Test GitHub URL detection functionality"""
    print("🧪 Testing GitHub URL Detection")
    print("=" * 50)

    test_urls = [
        "https://github.com/octocat/Hello-World",
        "https://github.com/octocat/Hello-World.git",
        "git@github.com:octocat/Hello-World.git",
        "github.com/octocat/Hello-World",
        "local/path/to/repo",
        "/absolute/path/to/repo",
        "invalid-url",
    ]

    for url in test_urls:
        is_github = AgentConfig.is_github_url(url)
        normalized = AgentConfig.normalize_github_url(url) if is_github else "N/A"
        print(f"URL: {url}")
        print(f"  Is GitHub URL: {is_github}")
        print(f"  Normalized: {normalized}")
        print()


def test_small_repo_cloning():
    """Test cloning a small public repository"""
    print("🔄 Testing Repository Cloning")
    print("=" * 50)

    # Use a small, public repository for testing
    test_repo_url = "https://github.com/octocat/Hello-World"

    try:
        print(f"Testing with repository: {test_repo_url}")

        # Configure agent with GitHub URL
        config = AgentConfig(
            repo_path=test_repo_url,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o-mini",
            doc_types=["readme"],  # Only generate README to keep it quick
            output_dir="test_output",
            auto_cleanup=True,
        )

        print("✅ Configuration created successfully")
        print(f"Auto-cleanup enabled: {config.auto_cleanup}")

        # Initialize agent (this will trigger cloning)
        print("🔧 Initializing agent (this will clone the repository)...")
        agent = DocumentationAgent(config)

        print("✅ Agent initialized successfully")
        print(f"Cloned repository path: {agent._cloned_repo_path}")

        # Verify the cloned repository exists
        if agent._cloned_repo_path and Path(agent._cloned_repo_path).exists():
            print("✅ Repository cloned successfully")
            print(f"Repository contents:")
            for item in Path(agent._cloned_repo_path).iterdir():
                print(f"  - {item.name}")
        else:
            print("❌ Repository cloning failed")
            return

        # Test basic analysis
        print("\n🔍 Testing repository analysis...")
        repo_metadata, modules = agent.analyze_repository()

        print(f"✅ Analysis completed:")
        print(f"  Repository name: {repo_metadata.name}")
        print(f"  Language: {repo_metadata.language}")
        print(f"  Files: {repo_metadata.file_count}")
        print(f"  Modules: {len(modules)}")

        # Test cleanup
        print("\n🧹 Testing cleanup...")
        agent._cleanup_cloned_repo()

        if not Path(agent._cloned_repo_path).exists():
            print("✅ Repository cleaned up successfully")
        else:
            print("⚠️ Repository cleanup may have failed")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()


def test_ssh_url_format():
    """Test SSH URL format handling"""
    print("🔑 Testing SSH URL Format")
    print("=" * 50)

    ssh_url = "git@github.com:octocat/Hello-World.git"

    print(f"Original SSH URL: {ssh_url}")
    print(f"Is GitHub URL: {AgentConfig.is_github_url(ssh_url)}")
    print(f"Normalized: {AgentConfig.normalize_github_url(ssh_url)}")


def main():
    """Run all tests"""
    print("🚀 GitHub URL Cloning Test Suite")
    print("=" * 80)
    print()

    # Test 1: URL Detection
    test_github_url_detection()

    # Test 2: SSH URL handling
    test_ssh_url_format()

    # Test 3: Actual cloning (requires internet)
    if os.getenv("OPENAI_API_KEY"):
        print("🌐 Testing actual repository cloning (requires internet)...")
        test_small_repo_cloning()
    else:
        print("⚠️ Skipping cloning test - no OPENAI_API_KEY found")
        print("Set OPENAI_API_KEY environment variable to test cloning")

    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    main()
