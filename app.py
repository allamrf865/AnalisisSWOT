import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
from io import BytesIO

# Config Streamlit
st.set_page_config(page_title="Advanced SWOT Leadership Analysis", page_icon="🌟", layout="wide")

# Define watermark for consistency
WATERMARK = "AI by Muhammad Allam Rafi, CBOA® CDSP®"

# Load NLP Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Leadership Traits
LEADERSHIP_QUALITIES = {
    "Positive": {
        "Leadership": "Ability to lead and inspire others.",
        "Vision": "Clear and inspiring direction for the future.",
        "Integrity": "Acting consistently with honesty and strong ethics.",
        "Innovation": "Driving creativity and fostering change.",
        "Inclusivity": "Promoting diversity and creating an inclusive environment.",
        "Empathy": "Understanding others' perspectives and feelings.",
        "Communication": "Conveying ideas clearly and effectively.",
    },
    "Neutral": {
        "Adaptability": "Flexibility to adjust to new challenges.",
        "Time Management": "Prioritizing and organizing tasks efficiently.",
        "Problem-Solving": "Resolving issues effectively.",
        "Conflict Resolution": "Managing disagreements constructively.",
        "Resilience": "Bouncing back from setbacks.",
    },
    "Negative": {
        "Micromanagement": "Excessive control over tasks.",
        "Overconfidence": "Ignoring input due to arrogance.",
        "Conflict Avoidance": "Avoiding necessary confrontations.",
        "Indecisiveness": "Inability to make timely decisions.",
        "Rigidity": "Refusing to adapt to new circumstances.",
    }
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.3, "Opportunities": 1.4, "Threats": 1.2}

# Analyze Text with NLP and Provide Explanation
def analyze_text_with_explanation(text, qualities, confidence, category_weight):
    scores, explanations = {}, {}
    embeddings = model.encode([text] + list(qualities.values()), convert_to_tensor=True)
    text_embedding, trait_embeddings = embeddings[0], embeddings[1:]
    similarities = util.pytorch_cos_sim(text_embedding, trait_embeddings).squeeze().tolist()
    
    for trait, similarity in zip(qualities.keys(), similarities):
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score
        relevance = "High" if similarity > 0.75 else "Moderate" if similarity > 0.5 else "Low"
        explanations[trait] = (
            f"Input '{text}' aligns with '{trait}' ({qualities[trait]}). "
            f"Similarity: {similarity:.2f}, Confidence: {confidence}/10, Weighted Score: {weighted_score:.2f}. "
            f"Relevance: {relevance}."
        )
    return scores, explanations

# Calculate LSI
def calculate_lsi(scores):
    total_strengths = sum(scores["Strengths"].values())
    total_weaknesses = sum(scores["Weaknesses"].values())
    total_opportunities = sum(scores["Opportunities"].values())
    total_threats = sum(scores["Threats"].values())

    # Normalize values
    total = total_strengths + total_weaknesses + total_opportunities + total_threats + 1e-9
    strengths = total_strengths / total
    weaknesses = total_weaknesses / total
    opportunities = total_opportunities / total
    threats = total_threats / total

    # LSI Formula: Penalize weaknesses/threats, reward strengths/opportunities
    numerator = strengths + opportunities * 1.3
    denominator = weaknesses * 1.5 + threats * 1.2 + 1e-9
    lsi = np.log((numerator / denominator) + 1e-9)
    return lsi

# Interpret LSI
def interpret_lsi(lsi):
    if lsi > 1.5:
        return "Exceptional Leadership Potential"
    elif lsi > 0.5:
        return "Good Leadership Potential"
    elif lsi > -0.5:
        return "Moderate Leadership Potential"
    else:
        return "Needs Improvement"

# Visualization (2D Heatmap)
def generate_2d_heatmap(scores):
    df = pd.DataFrame(scores).T.fillna(0)
    fig = px.imshow(df, title="Trait Alignment Heatmap", text_auto=True, color_continuous_scale="Viridis")
    return fig

# Visualization (3D Bar Chart)
def generate_3d_bar_chart(scores):
    traits, categories, values = [], [], []
    for category, traits_scores in scores.items():
        for trait, value in traits_scores.items():
            traits.append(trait)
            categories.append(category)
            values.append(value)
    fig = px.bar_3d(x=categories, y=traits, z=values, title="3D SWOT Scores", color=values, color_continuous_scale="Viridis")
    return fig

# Generate PDF Report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "SWOT-Based Leadership Evaluation Report", align='C', ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Generated by Muhammad Allam Rafi, CBOA® CDSP®", align='C')

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)

def generate_pdf_report(swot_scores, lsi, lsi_interpretation, explanations, heatmap_path, bar3d_path):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Leadership Viability Index (LSI): {lsi:.2f}", ln=True)
    pdf.cell(0, 10, f"Interpretation: {lsi_interpretation}", ln=True)

    # Add SWOT Scores and Explanations
    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))

    # Add Heatmap
    pdf.image(heatmap_path, x=10, y=100, w=190)
    pdf.add_page()

    # Add 3D Bar Chart
    pdf.image(bar3d_path, x=10, y=100, w=190)
    pdf.output("/tmp/report.pdf")
    return "/tmp/report.pdf"

# Streamlit Interface
st.title("🌟 Advanced SWOT Leadership Analysis 🌟")

# User Inputs
swot_inputs = {cat: [(st.text_area(f"{cat} #{i+1}"), st.slider(f"{cat} #{i+1} Confidence", 1, 10, 5)) for i in range(3)] for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}

if st.button("Analyze"):
    # Analyze inputs
    swot_scores, swot_explanations = {}, {}
    for category, inputs in swot_inputs.items():
        category_scores, category_explanations = {}, {}
        qualities = (
            LEADERSHIP_QUALITIES["Positive"] if category in ["Strengths", "Opportunities"]
            else LEADERSHIP_QUALITIES["Negative"] if category == "Threats"
            else LEADERSHIP_QUALITIES["Neutral"]
        )
        for text, confidence in inputs:
            if text.strip():
                scores, explanations = analyze_text_with_explanation(text, qualities, confidence, CATEGORY_WEIGHTS[category])
                category_scores.update(scores)
                category_explanations.update(explanations)
        swot_scores[category] = category_scores
        swot_explanations[category] = category_explanations

    # Calculate LSI
    lsi = calculate_lsi(swot_scores)
    lsi_interpretation = interpret_lsi(lsi)

    # Display Results
    st.subheader(f"Leadership Viability Index (LSI): {lsi:.2f}")
    st.write(f"**Interpretation**: {lsi_interpretation}")

    # Generate Visualizations
    heatmap_fig = generate_2d_heatmap(swot_scores)
    st.plotly_chart(heatmap_fig)
    heatmap_path = "/tmp/heatmap.png"
    heatmap_fig.write_image(heatmap_path)

    bar3d_fig = generate_3d_bar_chart(swot_scores)
    st.plotly_chart(bar3d_fig)
    bar3d_path = "/tmp/bar3d.png"
    bar3d_fig.write_image(bar3d_path)

    # Generate and Download PDF
    pdf_path = generate_pdf_report(swot_scores, lsi, lsi_interpretation, swot_explanations, heatmap_path, bar3d_path)
    with open(pdf_path, "rb") as f:
        st.download_button("Download Professional PDF Report", f, "Leadership_Report.pdf", mime="application/pdf")
