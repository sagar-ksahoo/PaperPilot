from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import networkx as nx
import matplotlib.pyplot as plt
import re

class MindmapGenerator:
    """Generates a mind map from a collection of research paper texts."""

    def __init__(self, model: str = "mistral"):
        """
        Initializes the MindmapGenerator.

        Args:
            model (str): The name of the Ollama model to use.
        """
        # Prompt to extract relationships from text
        template = """
        Based on the following collection of research paper abstracts, extract the key concepts and their relationships.
        Present these relationships in a simple format, one per line, like this:
        (Concept A) -> [RELATIONSHIP] -> (Concept B)

        Focus on the most important and central themes that connect the papers.

        ABSTRACTS:
        {abstracts}

        EXTRACTED RELATIONSHIPS:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["abstracts"]
        )
        llm = Ollama(model=model)
        self.chain = prompt | llm | StrOutputParser()

    def _parse_relationships(self, text_blob: str) -> list:
        """Parses the LLM output to extract relationship tuples."""
        # Regex to find patterns like (Concept A) -> [RELATIONSHIP] -> (Concept B)
        pattern = re.compile(r'\((.*?)\)\s*->\s*\[(.*?)\]\s*->\s*\((.*?)\)')
        # Find all matches in the text
        matches = pattern.findall(text_blob)
        # Clean up whitespace for each part of the tuple
        return [(a.strip(), b.strip(), c.strip()) for a, b, c in matches]

    def generate_mindmap_data(self, abstracts: list) -> list:
        """
        Uses the LLM to generate relationship data from abstracts.

        Args:
            abstracts (list): A list of paper abstract strings.

        Returns:
            list: A list of tuples, where each tuple is a relationship.
        """
        if not abstracts:
            return []
        
        # Join all abstracts into a single string
        full_text = "\n\n".join(abstracts)
        
        # Invoke the LLM chain
        response = self.chain.invoke({"abstracts": full_text})
        
        # Parse the response to get structured data
        return self._parse_relationships(response)

    def create_mindmap_image(self, relationships: list, output_path: str = "mindmap.png"):
        """
        Creates and saves a mind map image from relationship data.

        Args:
            relationships (list): A list of relationship tuples.
            output_path (str): The path to save the output image.
        """
        if not relationships:
            print("No relationships found to generate a mind map.")
            return

        G = nx.DiGraph()

        # Add nodes and edges from the relationships
        for source, relation, target in relationships:
            G.add_node(source, label=source)
            G.add_node(target, label=target)
            G.add_edge(source, target, label=relation)

        plt.figure(figsize=(16, 12))
        
        # Use a layout that spreads nodes out
        pos = nx.spring_layout(G, k=3, iterations=50) 
        
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue')
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
        
        plt.title("Research Concepts Mind Map", size=20)
        plt.axis('off')
        plt.savefig(output_path, format="PNG", dpi=300)
        plt.close()
        print(f"✅ Mind map saved successfully to {output_path}")