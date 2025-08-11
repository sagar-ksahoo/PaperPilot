from vector_store.vector_manager import VectorManager
from agents.summarizer_agent import SummarizerAgent
import time

def test_summarization_pipeline():
    """Tests fetching a top paper and summarizing it."""
    print("--- Initializing Components ---")
    # Initialize the necessary components
    manager = VectorManager()
    summarizer = SummarizerAgent(model="mistral") # Ensure Ollama and 'mistral' are running

    # 1. Define a query to get a relevant paper
    query = "Transformer architectures in NLP"
    print(f"\n--- Searching and Ranking for '{query}' ---")
    
    # Check if we have papers in DB, if not, search for them
    if manager.collection.count() == 0:
        print("No papers in DB. Fetching from arXiv...")
        manager.search_and_process_papers(query=query, max_results=10)
    else:
        print("Papers found in DB.")
        
    ranked_papers = manager.rank_papers(query=query, top_n=1)

    if not ranked_papers:
        print("❌ Could not retrieve any papers for the query.")
        return

    top_paper = ranked_papers[0]
    title = top_paper.get('title', 'N/A')
    abstract = top_paper.get('summary', '')

    print(f"\n--- Summarizing Top Paper ---")
    print(f"Title: {title}")

    # 2. Summarize the abstract of the top paper
    start_time = time.time()
    summary = summarizer.summarize_paper(abstract)
    end_time = time.time()

    print("\n--- Generated Summary ---")
    print(summary)
    print(f"\n(Summary generated in {end_time - start_time:.2f} seconds)")

    assert summary is not None and "Error" not in summary
    print("\n✅ Summarizer Agent Test Passed!")

if __name__ == "__main__":
    # Ensure Ollama is running before starting the test!
    print("🚀 Starting Summarizer Agent Test...")
    print("Please ensure the Ollama application is running with the 'mistral' model available.")
    test_summarization_pipeline()