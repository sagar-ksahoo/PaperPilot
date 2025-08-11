import arxiv
import chromadb
from sentence_transformers import SentenceTransformer

class VectorManager:
    """Manages vector storage, retrieval, and ranking of research papers."""

    def __init__(self, db_path="./chroma_db"):
        """
        Initializes the VectorManager.

        Args:
            db_path (str): The path to the ChromaDB database directory.
        """
        # Use a pre-trained model for creating embeddings. 'all-MiniLM-L6-v2' is a great, lightweight choice.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Set up the persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create the collection. A collection in ChromaDB is like a table in a SQL DB.
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"} # Use cosine distance for similarity search
        )

    def search_and_process_papers(self, query: str, max_results: int = 20, year: int = None):
        """
        Searches arXiv, processes, and stores papers in the vector database.

        Args:
            query (str): The research topic to search for.
            max_results (int): The maximum number of papers to retrieve.
            year (int): Optional filter for the publication year.

        Returns:
            int: The number of new papers added to the database.
        """
        # Construct the search query for arXiv
        search_query = f'({query})'
        if year:
            # Format the date range for the specified year
            search_query += f' AND submittedDate: [{year}0101 TO {year}1231]'

        # Use the arxiv library to perform the search
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers_to_add = []
        for result in search.results():
            paper_id = result.entry_id
            
            # CRITICAL: Check if the paper is already in our database to avoid duplicates.
            if self.collection.get(ids=[paper_id])['ids']:
                print(f"Paper '{result.title}' already in DB. Skipping.")
                continue

            paper_meta = {
                'title': result.title,
                'authors': ", ".join([author.name for author in result.authors]),
                'published': result.published.strftime('%Y-%m-%d'),
                'summary': result.summary.replace("\n", " "), # Clean up newlines
                'url': result.pdf_url
            }
            
            papers_to_add.append({
                "document": paper_meta['summary'],
                "metadata": paper_meta,
                "id": paper_id
            })

        if not papers_to_add:
            print("No new papers found to add.")
            return 0

        # Add all new papers to the collection in one batch for efficiency
        self.collection.add(
            documents=[p['document'] for p in papers_to_add],
            metadatas=[p['metadata'] for p in papers_to_add],
            ids=[p['id'] for p in papers_to_add]
        )
        
        print(f"✅ Added {len(papers_to_add)} new papers to the database.")
        return len(papers_to_add)

    def rank_papers(self, query: str, top_n: int = 5) -> list:
        """
        Ranks papers in the database by semantic similarity to the query.

        Args:
            query (str): The user's research query.
            top_n (int): The number of top papers to return.

        Returns:
            list: A list of ranked paper metadata dictionaries.
        """
        # Query the collection using the user's query. ChromaDB handles the embedding.
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_n, self.collection.count()) # Ensure we don't ask for more results than exist
        )
        
        # The results are nested, so we extract the metadatas from the first query result
        return results['metadatas'][0] if results.get('metadatas') else []