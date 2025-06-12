"""
DocumentReader for ingesting existing documentation during generation process.
Combines LangChain's DirectoryLoader and LlamaIndex's SimpleDirectoryReader
for comprehensive document loading and processing.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document as LlamaDocument
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata."""

    content: str
    source: str
    file_type: str
    chunk_index: int
    metadata: Dict[str, Any]


class DocumentReader:
    """
    Enhanced document reader that loads and processes existing documentation
    files to provide context for documentation generation.
    """

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, supported_extensions: Optional[List[str]] = None
    ):
        """
        Initialize DocumentReader.

        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            supported_extensions: List of file extensions to process
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions or [".md", ".txt", ".rst", ".adoc", ".asciidoc"]

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
        )

        # Initialize node parser for LlamaIndex
        self.node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def read_documentation_directory(
        self, directory_path: str, recursive: bool = True, exclude_patterns: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """
        Read documentation files from a directory using LangChain's DirectoryLoader.

        Args:
            directory_path: Path to documentation directory
            recursive: Whether to search recursively
            exclude_patterns: Patterns to exclude (e.g., ['*.git/*', '*.node_modules/*'])

        Returns:
            List of processed document chunks
        """
        if not os.path.exists(directory_path):
            logger.warning(f"Directory not found: {directory_path}")
            return []

        try:
            # Create glob pattern for supported extensions
            glob_patterns = []
            for ext in self.supported_extensions:
                if recursive:
                    glob_patterns.append(f"**/*{ext}")
                else:
                    glob_patterns.append(f"*{ext}")

            all_chunks = []

            for pattern in glob_patterns:
                try:
                    loader = DirectoryLoader(
                        directory_path,
                        glob=pattern,
                        loader_cls=TextLoader,
                        loader_kwargs={"autodetect_encoding": True},
                        recursive=recursive,
                        show_progress=True,
                        use_multithreading=True,
                        silent_errors=True,
                        exclude=exclude_patterns or [],
                    )

                    documents = loader.load()
                    logger.info(f"Loaded {len(documents)} documents with pattern {pattern}")

                    # Process documents into chunks
                    for doc in documents:
                        chunks = self._process_langchain_document(doc)
                        all_chunks.extend(chunks)

                except Exception as e:
                    logger.error(f"Error loading documents with pattern {pattern}: {e}")
                    continue

            logger.info(f"Total chunks created: {len(all_chunks)}")
            return all_chunks

        except Exception as e:
            logger.error(f"Error reading documentation directory: {e}")
            return []

    def read_with_llamaindex(
        self, directory_path: str, recursive: bool = True, exclude_patterns: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """
        Read documentation files using LlamaIndex's SimpleDirectoryReader.

        Args:
            directory_path: Path to documentation directory
            recursive: Whether to search recursively
            exclude_patterns: Patterns to exclude

        Returns:
            List of processed document chunks
        """
        if not os.path.exists(directory_path):
            logger.warning(f"Directory not found: {directory_path}")
            return []

        try:
            # Set up file extractor mapping
            file_extractor = {}

            reader = SimpleDirectoryReader(
                input_dir=directory_path,
                exclude=exclude_patterns or [],
                recursive=recursive,
                required_exts=self.supported_extensions,
                filename_as_id=True,
                raise_on_error=False,
            )

            documents = reader.load_data(show_progress=True)
            logger.info(f"LlamaIndex loaded {len(documents)} documents")

            # Process documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = self._process_llamaindex_document(doc)
                all_chunks.extend(chunks)

            logger.info(f"Total chunks created: {len(all_chunks)}")
            return all_chunks

        except Exception as e:
            logger.error(f"Error reading documentation with LlamaIndex: {e}")
            return []

    def _process_langchain_document(self, document: LangchainDocument) -> List[DocumentChunk]:
        """Process a LangChain document into chunks."""
        try:
            # Extract metadata
            source = document.metadata.get("source", "unknown")
            file_path = Path(source)
            file_type = file_path.suffix.lower().lstrip(".")

            # Split document into chunks
            text_chunks = self.text_splitter.split_text(document.page_content)

            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():  # Skip empty chunks
                    chunk = DocumentChunk(
                        content=chunk_text.strip(),
                        source=source,
                        file_type=file_type,
                        chunk_index=i,
                        metadata={
                            **document.metadata,
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            "file_name": file_path.name,
                            "file_type": file_type,
                        },
                    )
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error processing LangChain document: {e}")
            return []

    def _process_llamaindex_document(self, document: LlamaDocument) -> List[DocumentChunk]:
        """Process a LlamaIndex document into chunks."""
        try:
            # Create nodes from document
            nodes = self.node_parser.get_nodes_from_documents([document])

            chunks = []
            for i, node in enumerate(nodes):
                source = node.metadata.get("file_path", "unknown")
                file_path = Path(source)
                file_type = file_path.suffix.lower().lstrip(".")

                chunk = DocumentChunk(
                    content=node.text.strip(),
                    source=source,
                    file_type=file_type,
                    chunk_index=i,
                    metadata={
                        **node.metadata,
                        "chunk_index": i,
                        "total_chunks": len(nodes),
                        "file_name": file_path.name,
                        "file_type": file_type,
                        "node_id": node.id_,
                    },
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error processing LlamaIndex document: {e}")
            return []

    def create_vector_index(self, chunks: List[DocumentChunk]) -> Optional[VectorStoreIndex]:
        """
        Create a LlamaIndex VectorStoreIndex from document chunks for semantic search.

        Args:
            chunks: List of document chunks

        Returns:
            VectorStoreIndex or None if creation fails
        """
        try:
            # Convert chunks to LlamaIndex documents
            documents = []
            for chunk in chunks:
                doc = LlamaDocument(text=chunk.content, metadata=chunk.metadata)
                documents.append(doc)

            # Create vector index
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
            logger.info(f"Created vector index with {len(documents)} documents")

            return index

        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            return None

    def search_similar_content(self, query: str, index: VectorStoreIndex, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content in the vector index.

        Args:
            query: Search query
            index: Vector index to search
            top_k: Number of top results to return

        Returns:
            List of similar documents with scores
        """
        try:
            # Create query engine
            query_engine = index.as_query_engine(
                similarity_top_k=top_k, response_mode="no_text"  # Just return nodes, no synthesis
            )

            # Perform search
            response = query_engine.query(query)

            # Extract results
            results = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    results.append(
                        {"content": node.text, "metadata": node.metadata, "score": getattr(node, "score", 0.0)}
                    )

            return results

        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []

    def get_documentation_summary(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Generate a summary of the loaded documentation.

        Args:
            chunks: List of document chunks

        Returns:
            Summary statistics and metadata
        """
        if not chunks:
            return {"total_files": 0, "total_chunks": 0, "file_types": {}}

        file_types = {}
        sources = set()
        total_content_length = 0

        for chunk in chunks:
            sources.add(chunk.source)
            file_type = chunk.file_type
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_content_length += len(chunk.content)

        return {
            "total_files": len(sources),
            "total_chunks": len(chunks),
            "file_types": file_types,
            "average_chunk_size": total_content_length // len(chunks) if chunks else 0,
            "sources": list(sources),
        }
