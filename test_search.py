from vector_store.vector_manager import VectorManager

def test_full_pipeline():
    """Tests the full search, store, and rank pipeline."""
    # Initialize our manager. This will create a 'chroma_db' folder.
    manager = VectorManager()
    
    # 1. Define a test query and filters
    query = "Transformer architectures in NLP after 2022"
    year_filter = 2023
    
    print(f"--- Step 1: Searching arXiv for papers on '{query}' from {year_filter} ---")
    # This will fetch from arXiv and add any new papers to ChromaDB
    num_added = manager.search_and_process_papers(query=query, max_results=25, year=year_filter)
    print(f"Completed search. Added {num_added} papers.")

    # 2. Rank the papers based on the query
    print("\n--- Step 2: Ranking stored papers by relevance ---")
    # This performs a semantic search within ChromaDB
    ranked_papers = manager.rank_papers(query=query, top_n=5)
    
    # 3. Print the results to verify
    print(f"\nTop {len(ranked_papers)} relevant papers found:")
    for i, paper in enumerate(ranked_papers):
        print(f"  {i+1}. Title: {paper['title']} ({paper['published']})")
        print(f"     URL: {paper['url']}")

    assert len(ranked_papers) > 0, "Ranking should return at least one paper."
    print("\n✅ Vector Manager Test Passed!")

if __name__ == "__main__":
    test_full_pipeline()