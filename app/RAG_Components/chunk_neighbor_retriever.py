"""
Chunk neighbor retrieval functionality.
Handles retrieving neighboring chunks from the vector store for better context.
"""

from typing import Dict, List
import chromadb


class ChunkNeighborRetriever:
    """
    Handles retrieval of neighboring chunks for better context in RAG responses.
    Works with ChromaDB to find chunks that are adjacent in the original document.
    """
    
    def __init__(self, collection=None):
        """Initialize with a ChromaDB collection"""
        self.collection = collection
    
    def set_collection(self, collection):
        """Set the ChromaDB collection to work with"""
        self.collection = collection
    
    def retrieve_neighbor_chunks_for_a_chunk(self, chunk: str, chunk_type: str = 'academic', chunk_score: float = 0, neighbor_count: int = 2) -> List[Dict]:
        """
        Retrieve neighboring chunks for a single given chunk to provide better context
        
        Args:
            chunk: Single chunk text to find neighbors for
            chunk_type: 'academic' or 'non-academic' 
            chunk_score: Score for non-academic chunks (ignored for academic)
            neighbor_count: Number of neighbors to retrieve on each side of target chunk
        
        Returns:
            List of chunk dictionaries with text, metadata, and scores in document order
            Includes the original chunk + its neighbors
        """
        try:
            if not chunk or not self.collection:
                return []
            
            print(f"\n=== NEIGHBOR CHUNK RETRIEVAL ===")
            print(f"Processing single {chunk_type} chunk")
            print(f"Neighbor count: {neighbor_count} on each side")
            print(f"Passed chunk (length: {len(chunk)}):")
            print(f"'{chunk[:200]}...' " + ("(truncated)" if len(chunk) > 200 else ""))
            
            # Get all documents from collection with metadata
            all_results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            if not all_results['documents']:
                return []
            
            all_documents = all_results['documents']
            all_metadatas = all_results['metadatas']
            
            # Find the target chunk in the collection
            target_metadata = None
            target_index = None
            doc_name = None
            
            for doc, metadata in zip(all_documents, all_metadatas):
                if doc == chunk:  # Found the target chunk
                    target_metadata = metadata
                    target_index = metadata.get('chunk_index', 0)
                    doc_name = metadata.get('document_name', 'Unknown')
                    break
            
            if target_metadata is None:
                print(f"❌ Target chunk not found in collection")
                print(f"Collection has {len(all_documents)} documents")
                print(f"Target chunk length: {len(chunk)}")
                print(f"Searching for exact matches...")
                
                # Try partial matching as fallback
                for i, (doc, metadata) in enumerate(zip(all_documents[:5], all_metadatas[:5])):
                    print(f"  Doc {i}: {len(doc)} chars, starts with: '{doc[:50]}...'")
                    if chunk[:100] in doc or doc[:100] in chunk:
                        print(f"  ✓ Found partial match with doc {i}")
                        target_metadata = metadata
                        target_index = metadata.get('chunk_index', 0)
                        doc_name = metadata.get('document_name', 'Unknown')
                        break
                
                if target_metadata is None:
                    print(f"❌ No matches found - returning empty list")
                    return []
            
            print(f"Found target chunk in document: {doc_name}, index: {target_index}")
            
            # Group all chunks from the same document by chunk_index
            document_chunks = {}
            for doc, metadata in zip(all_documents, all_metadatas):
                if metadata.get('document_name') == doc_name:
                    chunk_index = metadata.get('chunk_index', 0)
                    document_chunks[chunk_index] = {
                        'text': doc,
                        'metadata': metadata,
                        'chunk_index': chunk_index
                    }
            
            # Calculate neighbor range
            max_index = max(document_chunks.keys()) if document_chunks else 0
            start_index = max(0, target_index - neighbor_count)
            end_index = min(max_index, target_index + neighbor_count)
            
            print(f"Retrieving chunks from index {start_index} to {end_index} (max: {max_index})")
            
            # Collect chunks in document order
            result_chunks = []
            for chunk_idx in range(start_index, end_index + 1):
                if chunk_idx in document_chunks:
                    chunk_data = document_chunks[chunk_idx]
                    chunk_text = chunk_data['text']
                    
                    # Create result chunk with appropriate scoring
                    result_chunk = {
                        'text': chunk_text,
                        'metadata': chunk_data['metadata'],
                        'chunk_index': chunk_idx,
                        'document_name': doc_name,
                        'is_original_chunk': chunk_text == chunk  # True only for the target chunk
                    }
                    
                    # Add scoring based on chunk type
                    if chunk_type.lower() == 'non-academic':
                        if chunk_text == chunk:
                            # Original chunk gets the provided score
                            result_chunk['total_score'] = chunk_score
                            result_chunk['hierarchical_score'] = max(0, chunk_score - 15)  # Assume freshness ~15
                            result_chunk['freshness_score'] = min(15, chunk_score)
                        else:
                            # Neighbors get slightly reduced score
                            result_chunk['total_score'] = chunk_score * 0.9
                            result_chunk['hierarchical_score'] = max(0, chunk_score - 15) * 0.9
                            result_chunk['freshness_score'] = min(15, chunk_score) * 0.9
                    else:
                        # Academic chunks don't have hierarchical scores
                        result_chunk['similarity_based'] = True
                    
                    result_chunks.append(result_chunk)
            
            print(f"Retrieved {len(result_chunks)} chunks (including neighbors)")
            print(f"Chunk indices: {[chunk['chunk_index'] for chunk in result_chunks]}")
            
            print(f"\n=== NEIGHBOR CHUNKS RETRIEVED ===")
            for i, result_chunk in enumerate(result_chunks):
                chunk_preview = result_chunk['text'][:100] + ("..." if len(result_chunk['text']) > 100 else "")
                is_original = "ORIGINAL" if result_chunk['is_original_chunk'] else "NEIGHBOR"
                print(f"  {i+1}. [{is_original}] Index {result_chunk['chunk_index']}: '{chunk_preview}'")
            
            return result_chunks
            
        except Exception as e:
            print(f"Error in retrieve_neighbor_chunks_for_a_chunk: {e}")
            import traceback
            traceback.print_exc()
            return []


# Standalone function for direct import
def retrieve_neighbor_chunks_for_a_chunk(collection, chunk: str, chunk_type: str = 'academic', chunk_score: float = 0, neighbor_count: int = 2) -> List[Dict]:
    """
    Standalone function version for direct import.
    
    Args:
        collection: ChromaDB collection instance
        chunk: Single chunk text to find neighbors for
        chunk_type: 'academic' or 'non-academic' 
        chunk_score: Score for non-academic chunks (ignored for academic)
        neighbor_count: Number of neighbors to retrieve on each side of target chunk
    
    Returns:
        List of chunk dictionaries with text, metadata, and scores in document order
    """
    retriever = ChunkNeighborRetriever(collection)
    return retriever.retrieve_neighbor_chunks_for_a_chunk(chunk, chunk_type, chunk_score, neighbor_count)