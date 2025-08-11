from vector_store.vector_manager import VectorManager
from agents.summarizer_agent import SummarizerAgent
from tools.mindmap_generator import MindmapGenerator
from tools.pdf_exporter import PDFExporter

def test_full_brief_generation():
    # --- 1. Initialization ---
    print("--- Initializing all components ---")
    manager = VectorManager()
    summarizer = SummarizerAgent(model="mistral")
    mindmapper = MindmapGenerator(model="mistral")
    exporter = PDFExporter(model="mistral")

    # --- 2. Search and Rank ---
    query = "Large Language Models for Code Generation"
    print(f"\n--- Searching and ranking papers for '{query}' ---")
    if manager.collection.count() < 5:
        manager.search_and_process_papers(query=query, max_results=10)
    
    top_papers = manager.rank_papers(query=query, top_n=5)
    if not top_papers:
        print("❌ Could not find any papers. Exiting.")
        return

    print(f"Found {len(top_papers)} relevant papers.")

    # --- 3. Summarize Top Papers ---
    print("\n--- Summarizing top papers ---")
    summaries = {}
    for paper in top_papers:
        title = paper.get('title')
        abstract = paper.get('summary')
        summary = summarizer.summarize_paper(abstract)
        summaries[title] = summary
        print(f"  - Summarized '{title}'")

    # --- 4. Generate Mind Map ---
    print("\n--- Generating mind map ---")
    abstracts = [p.get('summary', '') for p in top_papers]
    relationships = mindmapper.generate_mindmap_data(abstracts)
    mindmap_path = "final_mindmap.png"
    mindmapper.create_mindmap_image(relationships, output_path=mindmap_path)

    # --- 5. Export to PDF ---
    print("\n--- Exporting to PDF ---")
    pdf_path = "Final_Research_Brief.pdf"
    exporter.export_to_pdf(query, top_papers, summaries, mindmap_path, pdf_path)
    
    print(f"\n✅ Full pipeline test complete. Check '{pdf_path}'!")

if __name__ == "__main__":
    print("🚀 Starting Full Research Brief Generation Test...")
    print("This will take a few minutes as it involves multiple LLM calls.")
    print("Please ensure Ollama is running with the 'mistral' model.")
    test_full_brief_generation()