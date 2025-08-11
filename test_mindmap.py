from vector_store.vector_manager import VectorManager
from tools.mindmap_generator import MindmapGenerator

def test_mindmap_pipeline():
    """Tests fetching papers and generating a mind map."""
    print("--- Initializing Components ---")
    manager = VectorManager()
    mindmapper = MindmapGenerator(model="mistral")

    query = "Transformer architectures in computer vision"
    print(f"\n--- Getting Top Papers for '{query}' ---")
    
    # Ensure we have papers to work with
    if manager.collection.count() < 5:
        print("Fewer than 5 papers in DB. Fetching more...")
        manager.search_and_process_papers(query=query, max_results=10)

    ranked_papers = manager.rank_papers(query=query, top_n=5)

    if not ranked_papers:
        print("❌ Could not retrieve any papers.")
        return

    # Extract abstracts from the top papers
    abstracts = [paper.get('summary', '') for paper in ranked_papers]
    print(f"Found {len(abstracts)} abstracts to process.")

    print("\n--- Generating Mind Map Data with LLM ---")
    # Generate the relationship data
    relationships = mindmapper.generate_mindmap_data(abstracts)

    if not relationships:
        print("❌ LLM did not return any parsable relationships.")
        return
        
    print(f"Extracted {len(relationships)} relationships.")
    print(relationships)

    print("\n--- Creating Mind Map Image ---")
    # Create and save the image
    mindmapper.create_mindmap_image(relationships, output_path="test_mindmap.png")


if __name__ == "__main__":
    print("🚀 Starting Mind Map Generation Test...")
    print("Please ensure Ollama is running with the 'mistral' model.")
    test_mindmap_pipeline()