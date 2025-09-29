from typing import Dict, List, Tuple
from app.models.chat import StudentQueryRequest

# External dependencies - using relative imports to avoid path issues
try:
    from .Ontology_ranking import apply_hierarchical_ranking
    from .Meta_data_ruleset import MetadataRuleset
    from .chunk_neighbor_retriever import ChunkNeighborRetriever
except ImportError:
    # Fallback to absolute imports
    from app.RAG_Components.Ontology_ranking import apply_hierarchical_ranking
    from app.RAG_Components.Meta_data_ruleset import MetadataRuleset
    from app.RAG_Components.chunk_neighbor_retriever import ChunkNeighborRetriever

def apply_rule_based_filters(
    results: Dict,
    studentMetadata: StudentQueryRequest,
    collection=None  # Add collection parameter for neighbor retrieval
) -> Tuple[List[str], List[str]]:
    """
    Apply strict rule-based filtering with hierarchical ranking for non-academic documents
    Returns: (academic_chunks_list, non_academic_chunks_list)
    """
    if not results['documents'] or len(results['documents']) == 0:
        return [], []
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    distances = results['distances'][0] if results['distances'] else []
    
    # Initialize components
    metadata_ruleset = MetadataRuleset()
    neighbor_retriever = ChunkNeighborRetriever(collection)
    
    # Separate chunks by type during filtering
    academic_chunks = []
    non_academic_chunks = []
    
    for doc, metadata, distance in zip(documents, metadatas, distances):
        if metadata_ruleset.passes_all_rules(metadata, studentMetadata):
            similarity_score = 1 - distance  # Convert distance to similarity
            chunk_data = {
                'text': doc,
                'metadata': metadata,
                'similarity_score': similarity_score
            }
            
            if metadata.get('type', '').lower() == 'academic':
                academic_chunks.append(chunk_data)
            else:
                non_academic_chunks.append(chunk_data)
    
    print(f"Chunks after rule filtering - Academic: {len(academic_chunks)}, Non-Academic: {len(non_academic_chunks)}")
    
    # Apply hierarchical ranking to non-academic chunks
    if non_academic_chunks:
        ranked_non_academic = apply_hierarchical_ranking(non_academic_chunks, studentMetadata)
    else:
        ranked_non_academic = []
    
    # Process academic chunks with neighbors (sorted by similarity)
    final_academic_list = []
    academic_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
    for chunk in academic_chunks:
        print(f"\n--- Processing Academic Chunk ---")
        print(f"Original chunk preview: {chunk['text'][:100]}...")
        
        chunk_with_neighbors = neighbor_retriever.retrieve_neighbor_chunks_for_a_chunk(
            chunk=chunk['text'],
            chunk_type="academic",
            neighbor_count=2  # Only 2 neighbors on each side
        )
        
        if chunk_with_neighbors:
            print(f"Found {len(chunk_with_neighbors)} chunks (including neighbors)")
            for neighbor_chunk in chunk_with_neighbors:
                final_academic_list.append(neighbor_chunk['text'])
        else:
            print(f"No neighbors found for academic chunk, adding original")
            final_academic_list.append(chunk['text'])
    
    # Process non-academic chunks with neighbors (already ranked)
    final_non_academic_list = []
    for chunk in ranked_non_academic:
        print(f"\n--- Processing Non-Academic Chunk ---")
        print(f"Original chunk preview: {chunk['text'][:100]}...")
        print(f"Chunk score: {chunk['total_score']}")
        
        chunk_with_neighbors = neighbor_retriever.retrieve_neighbor_chunks_for_a_chunk(
            chunk=chunk['text'],
            chunk_type="non-academic",
            chunk_score=chunk['total_score'],
            neighbor_count=2  # Only 2 neighbors on each side
        )
        
        if chunk_with_neighbors:
            print(f"Found {len(chunk_with_neighbors)} chunks (including neighbors)")
            for neighbor_chunk in chunk_with_neighbors:
                if neighbor_chunk['is_original_chunk']:
                    annotated_chunk = f"{neighbor_chunk['text']} [ORIGINAL-SCORE: {chunk['total_score']:.1f}]"
                else:
                    annotated_chunk = f"{neighbor_chunk['text']} [NEIGHBOR-SCORE: {chunk['total_score']*0.9:.1f}]"
                final_non_academic_list.append(annotated_chunk)
        else:
            print(f"No neighbors found for non-academic chunk, adding original")
            annotated_chunk = f"{chunk['text']} [SCORE: {chunk['total_score']:.1f}]"
            final_non_academic_list.append(annotated_chunk)
    
    print(f"\n=== BEFORE DEDUPLICATION ===")
    print(f"Academic chunks: {len(final_academic_list)}")
    print(f"Non-academic chunks: {len(final_non_academic_list)}")
    
    # ==================== DEDUPLICATION ====================
    
    # Deduplicate academic chunks while preserving order
    seen_academic = set()
    deduplicated_academic = []
    for chunk in final_academic_list:
        # Create a shorter key for comparison (first 100 characters)
        chunk_key = chunk[:100].strip()
        if chunk_key not in seen_academic:
            seen_academic.add(chunk_key)
            deduplicated_academic.append(chunk)
        else:
            print(f"Removed duplicate academic chunk: {chunk[:50]}...")
    
    # Deduplicate non-academic chunks while preserving order and scores
    seen_non_academic = set()
    deduplicated_non_academic = []
    for chunk in final_non_academic_list:
        # Extract text without score annotation for comparison
        if '[SCORE:' in chunk or '[ORIGINAL-SCORE:' in chunk or '[NEIGHBOR-SCORE:' in chunk:
            # Find the last occurrence of '[' to split text from score
            last_bracket = chunk.rfind('[')
            chunk_text = chunk[:last_bracket].strip() if last_bracket > 0 else chunk
        else:
            chunk_text = chunk
        
        # Create a shorter key for comparison
        chunk_key = chunk_text[:100].strip()
        if chunk_key not in seen_non_academic:
            seen_non_academic.add(chunk_key)
            deduplicated_non_academic.append(chunk)
        else:
            print(f"Removed duplicate non-academic chunk: {chunk[:50]}...")
    
    print(f"\n=== AFTER DEDUPLICATION ===")
    print(f"Academic chunks: {len(deduplicated_academic)} (removed {len(final_academic_list) - len(deduplicated_academic)} duplicates)")
    print(f"Non-academic chunks: {len(deduplicated_non_academic)} (removed {len(final_non_academic_list) - len(deduplicated_non_academic)} duplicates)")
    
    # Show final chunk previews with lengths
    for i, chunk in enumerate(deduplicated_academic[:3], 1):  # Show first 3
        print(f"Academic {i} (len:{len(chunk)}): {chunk[:100]}...")
        
    for i, chunk in enumerate(deduplicated_non_academic[:3], 1):  # Show first 3
        print(f"Non-Academic {i} (len:{len(chunk)}): {chunk[:100]}...")
    
    # Calculate total token estimate (rough: 4 chars = 1 token)
    total_academic_tokens = sum(len(chunk) // 4 for chunk in deduplicated_academic)
    total_non_academic_tokens = sum(len(chunk) // 4 for chunk in deduplicated_non_academic)
    total_tokens = total_academic_tokens + total_non_academic_tokens
    
    print(f"\n=== TOKEN USAGE ESTIMATE ===")
    print(f"Academic tokens: ~{total_academic_tokens}")
    print(f"Non-academic tokens: ~{total_non_academic_tokens}")
    print(f"Total estimated tokens: ~{total_tokens}")
    
    if total_tokens > 6000:
        print(f"⚠️  WARNING: Estimated tokens ({total_tokens}) may exceed context window!")
    
    print(f"Final results - Academic: {len(deduplicated_academic)} chunks, Non-Academic: {len(deduplicated_non_academic)} chunks")
    
    return deduplicated_academic, deduplicated_non_academic
