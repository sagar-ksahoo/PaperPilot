import streamlit as st
import os
from vector_store.vector_manager import VectorManager
from agents.summarizer_agent import SummarizerAgent
from tools.mindmap_generator import MindmapGenerator
from tools.pdf_exporter import PDFExporter

# --- Page Configuration ---
st.set_page_config(
    page_title="PaperPilot Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- App State Management ---
if 'assistant_initialized' not in st.session_state:
    st.session_state.vector_manager = VectorManager()
    st.session_state.summarizer = SummarizerAgent(model="mistral")
    st.session_state.mindmapper = MindmapGenerator(model="mistral")
    st.session_state.exporter = PDFExporter(model="mistral")
    st.session_state.assistant_initialized = True
    st.session_state.messages = []
    st.session_state.results = {}

# --- UI Components ---
st.title("🤖 PaperPilot: Your AI Research Assistant")
st.markdown("Enter a research topic to get a full brief with summaries, a concept map, and a downloadable PDF report.")

with st.sidebar:
    st.header("Controls")
    topic = st.text_input("Research Topic", placeholder="e.g., Quantum Computing in Finance")
    max_papers = st.slider("Number of Papers to Analyze", min_value=3, max_value=15, value=5)
    year_filter = st.number_input("Year (Optional)", min_value=2000, max_value=2025, step=1, placeholder="e.g., 2023")
    
    generate_button = st.button("Generate Research Brief", type="primary", use_container_width=True)

# --- Main Logic ---
if generate_button and topic:
    st.session_state.results = {} # Clear previous results
    
    with st.spinner("Step 1/4: Searching and ranking papers..."):
        num_added = st.session_state.vector_manager.search_and_process_papers(query=topic, max_results=max_papers*2, year=year_filter)
        top_papers = st.session_state.vector_manager.rank_papers(query=topic, top_n=max_papers)
        st.session_state.results['top_papers'] = top_papers
        st.success(f"Found and ranked {len(top_papers)} relevant papers.")

    if top_papers:
        with st.spinner("Step 2/4: Summarizing top papers... (This may take a moment)"):
            summaries = {}
            for paper in top_papers:
                title = paper.get('title')
                abstract = paper.get('summary')
                summary = st.session_state.summarizer.summarize_paper(abstract)
                summaries[title] = summary
            st.session_state.results['summaries'] = summaries
            st.success("Summaries generated.")

        with st.spinner("Step 3/4: Generating concept mind map..."):
            abstracts = [p.get('summary', '') for p in top_papers]
            relationships = st.session_state.mindmapper.generate_mindmap_data(abstracts)
            mindmap_path = "research_mindmap.png"
            st.session_state.mindmapper.create_mindmap_image(relationships, output_path=mindmap_path)
            st.session_state.results['mindmap_path'] = mindmap_path
            st.success("Mind map created.")

        with st.spinner("Step 4/4: Compiling PDF report..."):
            pdf_path = "Research_Brief.pdf"
            executive_summary = st.session_state.exporter.generate_executive_summary(list(summaries.values()))
            st.session_state.results['executive_summary'] = executive_summary
            
            # Use the existing export_to_pdf but we already have the exec summary
            # We will refactor this slightly for better efficiency in a real app
            # For now, let's just generate the PDF with what we have
            st.session_state.exporter.export_to_pdf(topic, top_papers, summaries, mindmap_path, pdf_path)
            st.session_state.results['pdf_path'] = pdf_path
            st.success("PDF report compiled.")
            
# --- Display Results ---
if st.session_state.results:
    st.divider()
    st.header("Research Brief")

    # Executive Summary
    st.subheader("Executive Summary")
    st.write(st.session_state.results.get('executive_summary', "Not available."))

    # PDF Download Button
    pdf_path = st.session_state.results.get('pdf_path')
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="Download Full PDF Report",
                data=file,
                file_name=f"{topic.replace(' ', '_')}_Brief.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    # Mind Map
    st.subheader("Concept Mind Map")
    mindmap_path = st.session_state.results.get('mindmap_path')
    if mindmap_path and os.path.exists(mindmap_path):
        st.image(mindmap_path)
    else:
        st.warning("Mind map could not be generated.")

    # Detailed Summaries
    st.subheader("Detailed Summaries")
    top_papers = st.session_state.results.get('top_papers', [])
    summaries = st.session_state.results.get('summaries', {})
    
    if not top_papers:
        st.warning("No papers were found for this topic.")
    
    for paper in top_papers:
        title = paper.get('title')
        with st.expander(f"**{title}**"):
            st.markdown(f"**Authors:** {paper.get('authors', 'N/A')}")
            st.markdown(f"**Published:** {paper.get('published', 'N/A')}")
            st.markdown(f"**[Read Paper]({paper.get('url', '#')})**")
            st.markdown("---")
            st.write(summaries.get(title, "Summary not available."))# Entry point for PaperPilot app
