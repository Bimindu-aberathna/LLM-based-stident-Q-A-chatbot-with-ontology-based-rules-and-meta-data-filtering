"""
Example integration of hierarchical ranking system in chat endpoint

This shows how to use the NonAcademicChunkRanker with your existing ChromaDB system
"""

from app.abstract_factory.Database.chromadb import ChromaDB
from app.RAG_Components.chunk_ranker import NonAcademicChunkRanker
from app.models.chat import StudentQueryRequest

def enhanced_retrieval_with_ranking(query_vector, student_request, top_k=5):
    """
    Enhanced retrieval with hierarchical ranking for non-academic documents
    
    Process:
    1. Retrieve filtered chunks (academic + non-academic) 
    2. Keep academic chunks as-is (already well-filtered by course/year rules)
    3. Apply hierarchical ranking to non-academic chunks
    4. Combine and return top results
    """
    
    # Initialize components
    db = ChromaDB()
    ranker = NonAcademicChunkRanker()
    
    # Get structured retrieval data
    retrieval_data = db.retrieve_with_ranking_data(
        query_vector, student_request, top_k=15, similarity_threshold=0.3
    )
    
    academic_chunks = retrieval_data['academic_chunks']
    non_academic_chunks = retrieval_data['non_academic_chunks']
    
    print(f"\n=== ENHANCED RETRIEVAL PIPELINE ===")
    print(f"Academic chunks (pre-filtered): {len(academic_chunks)}")
    print(f"Non-academic chunks (for ranking): {len(non_academic_chunks)}")
    
    final_chunks = []
    
    # Academic chunks: Use as-is (already well-filtered by your rule system)
    for chunk in academic_chunks:
        final_chunks.append({
            'text': chunk['text'],
            'type': 'academic',
            'score': chunk['similarity_score'] * 100,  # Similarity-based score
            'source': chunk['metadata'].get('document_name', 'Unknown'),
            'priority': 'high'  # Academic content is high priority for students
        })
    
    # Non-academic chunks: Apply hierarchical ranking
    if non_academic_chunks:
        ranked_non_academic = ranker.rank_chunks(non_academic_chunks, student_request)
        
        for ranked_chunk in ranked_non_academic:
            final_chunks.append({
                'text': ranked_chunk['text'],
                'type': 'non-academic',
                'score': ranked_chunk['total_score'],
                'source': ranked_chunk['metadata'].get('document_name', 'Unknown'),
                'priority': ranked_chunk['ranking_details']['hierarchical_level'],
                'freshness_days': ranked_chunk['ranking_details']['document_age_days']
            })
    
    # Sort all chunks by score (academic similarity + non-academic hierarchical)
    final_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top chunks
    top_chunks = final_chunks[:top_k]
    
    print(f"\n=== FINAL RANKING RESULTS ===")
    for i, chunk in enumerate(top_chunks, 1):
        chunk_preview = chunk['text'][:50] + "..." if len(chunk['text']) > 50 else chunk['text']
        print(f"{i}. [{chunk['type'].upper()}] {chunk_preview}")
        print(f"   Score: {chunk['score']:.1f} | Priority: {chunk['priority']} | Source: {chunk['source'][:30]}")
        if chunk['type'] == 'non-academic':
            print(f"   Freshness: {chunk['freshness_days']} days old")
    
    # Return just the text chunks for LLM
    return [chunk['text'] for chunk in top_chunks]

# Example usage in your chat.py:
"""
# In your chat endpoint, replace:
# results = dbInstance.retrieve_similar_with_metadata(query_embedding, request, top_k=5)

# With:
results = enhanced_retrieval_with_ranking(query_embedding, request, top_k=5)
"""