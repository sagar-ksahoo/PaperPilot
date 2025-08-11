# SummarizerAgent implementation placeholder
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SummarizerAgent:
    """An agent that summarizes research paper abstracts."""

    def __init__(self, model: str = "mistral"):
        """
        Initializes the SummarizerAgent.

        Args:
            model (str): The name of the Ollama model to use (e.g., 'mistral', 'llama3').
        """
        # FIX: Store the model name as an attribute
        self.model = model
        
        # Define the prompt template for summarization
        template = """
        As an expert research assistant, your goal is to provide a clear and concise summary of the following research paper abstract.
        Focus on the key findings, methodology, and the main contribution of the paper.

        ABSTRACT:
        {abstract}

        CONCISE SUMMARY:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["abstract"]
        )

        # Initialize the Ollama LLM
        llm = Ollama(model=model)

        # Create the summarization chain by piping the components together
        self.chain = prompt | llm | StrOutputParser()

    def summarize_paper(self, abstract: str) -> str:
        """
        Generates a summary for a given paper abstract.

        Args:
            abstract (str): The abstract of the research paper.

        Returns:
            str: The generated summary.
        """
        if not abstract or not isinstance(abstract, str):
            return "Error: Invalid abstract provided for summarization."
            
        try:
            summary = self.chain.invoke({"abstract": abstract})
            return summary
        except Exception as e:
            print(f"An error occurred during summarization: {e}")
            return "Error: Could not generate summary."