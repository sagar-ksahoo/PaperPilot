import fitz  # PyMuPDF
from agents.summarizer_agent import SummarizerAgent

class PDFExporter:
    """Exports the research brief and mind map to a PDF file."""

    def __init__(self, model: str = "mistral"):
        self.summarizer = SummarizerAgent(model=model)

    def generate_executive_summary(self, paper_summaries: list) -> str:
        summaries_text = "\n\n".join(paper_summaries)
        
        from langchain_core.prompts import PromptTemplate
        from langchain_community.llms import Ollama
        from langchain_core.output_parsers import StrOutputParser

        template = """
        You are a senior research analyst. Based on the following summaries of several research papers on a single topic, please synthesize them into a single, high-level executive summary.
        Identify the overarching trends, common methodologies, and any potential research gaps or future directions mentioned. The summary should be concise and accessible to someone familiar with the field.
        INDIVIDUAL SUMMARIES:
        {summaries}
        EXECUTIVE SUMMARY:
        """
        prompt = PromptTemplate(template=template, input_variables=["summaries"])
        
        # FIX: Access the model name from the summarizer's stored attribute
        llm = Ollama(model=self.summarizer.model)
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"summaries": summaries_text})

    def export_to_pdf(self, research_topic: str, ranked_papers: list, summaries: dict, mindmap_path: str, output_path: str = "Research_Brief.pdf"):
        doc = fitz.open()
        page = doc.new_page()

        y_cursor = 50
        title_font_size = 20
        heading_font_size = 16
        body_font_size = 11
        line_height = 14
        margin = 50
        page_width = page.rect.width
        text_width = page_width - 2 * margin

        page.insert_text((margin, y_cursor), f"Research Brief: {research_topic}", fontsize=title_font_size)
        y_cursor += 40

        page.insert_text((margin, y_cursor), "Executive Summary", fontsize=heading_font_size)
        y_cursor += 25
        paper_summaries_list = [summaries[p['title']] for p in ranked_papers if p['title'] in summaries]
        executive_summary = self.generate_executive_summary(paper_summaries_list)
        text_rect = fitz.Rect(margin, y_cursor, page_width - margin, page.rect.height - 50)
        res = page.insert_textbox(text_rect, executive_summary, fontsize=body_font_size)
        y_cursor += res + 20

        if y_cursor > page.rect.height - 280:
            page = doc.new_page()
            y_cursor = 50
        page.insert_text((margin, y_cursor), "Concept Mind Map", fontsize=heading_font_size)
        y_cursor += 25
        img_rect = fitz.Rect(margin, y_cursor, page_width - margin, y_cursor + 250)
        page.insert_image(img_rect, filename=mindmap_path)
        y_cursor += 270

        if y_cursor > page.rect.height - 50:
            page = doc.new_page()
            y_cursor = 50
        page.insert_text((margin, y_cursor), "Detailed Summaries & Sources", fontsize=heading_font_size)
        y_cursor += 25

        for paper in ranked_papers:
            if y_cursor > page.rect.height - 150:
                page = doc.new_page()
                y_cursor = 50
            title = paper.get('title', 'N/A')
            url = paper.get('url', 'N/A')
            link_rect = fitz.Rect(margin, y_cursor, page_width - margin, y_cursor + line_height)
            page.insert_link({"kind": fitz.LINK_URI, "uri": url, "from": link_rect})
            page.insert_text((margin, y_cursor + 11), title, fontsize=body_font_size + 1, color=(0,0,1))
            y_cursor += line_height * 1.5
            authors = paper.get('authors', 'N/A')
            page.insert_text((margin, y_cursor), f"Authors: {authors}", fontsize=body_font_size - 1)
            y_cursor += line_height
            summary = summaries.get(title, "Summary not available.")
            text_rect = fitz.Rect(margin, y_cursor, text_width, 800)
            res = page.insert_textbox(text_rect, summary, fontsize=body_font_size)
            y_cursor += res + 20

        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        print(f"✅ Research brief saved successfully to {output_path}")