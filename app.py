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

# Define watermark
WATERMARK = "AI by Muhammad Allam Rafi, CBOA® CDSP®"

# Load NLP Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define Leadership Traits
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

# Analyze Text with NLP
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

# Behavioral Analysis
def analyze_behavioral_responses(behavioral_responses):
    behavioral_scores, behavioral_explanations = {}, {}
    for question, response in behavioral_responses.items():
        qualities = {**LEADERSHIP_QUALITIES["Positive"], **LEADERSHIP_QUALITIES["Neutral"]}
        scores, explanations = analyze_text_with_explanation(response, qualities, confidence=10, category_weight=1)
        behavioral_scores[question] = scores
        behavioral_explanations[question] = explanations
    return behavioral_scores, behavioral_explanations

# Link Behavioral to SWOT Categories
def map_behavioral_to_swot(behavioral_scores):
    swot_relevance = {"Strengths": 0, "Weaknesses": 0, "Opportunities": 0, "Threats": 0}
    for question, scores in behavioral_scores.items():
        for trait, score in scores.items():
            if trait in LEADERSHIP_QUALITIES["Positive"]:
                swot_relevance["Strengths"] += score
            elif trait in LEADERSHIP_QUALITIES["Neutral"]:
                swot_relevance["Opportunities"] += score
            elif trait in LEADERSHIP_QUALITIES["Negative"]:
                swot_relevance["Threats"] += score
    return swot_relevance

# Calculate LSI
def calculate_lsi(swot_scores, behavioral_relevance):
    total_strengths = sum(swot_scores["Strengths"].values()) + behavioral_relevance["Strengths"]
    total_weaknesses = sum(swot_scores["Weaknesses"].values()) + behavioral_relevance["Weaknesses"]
    total_opportunities = sum(swot_scores["Opportunities"].values()) + behavioral_relevance["Opportunities"]
    total_threats = sum(swot_scores["Threats"].values()) + behavioral_relevance["Threats"]

    # Normalize values
    total = total_strengths + total_weaknesses + total_opportunities + total_threats + 1e-9
    strengths = total_strengths / total
    weaknesses = total_weaknesses / total
    opportunities = total_opportunities / total
    threats = total_threats / total

    # LSI Formula
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

def generate_pdf_report(swot_scores, behavioral_scores, lsi, lsi_interpretation, heatmap_path, bar3d_path):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Leadership Viability Index (LSI): {lsi:.2f}", ln=True)
    pdf.cell(0, 10, f"Interpretation: {lsi_interpretation}", ln=True)

    # Add SWOT Scores
    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))

    # Add Behavioral Scores
    for question, scores in behavioral_scores.items():
        pdf.add_section(f"Behavioral: {question}", "\n".join([f"{trait}: {value:.2f}" for trait, value in scores.items()]))

    # Add Heatmap
    pdf.image(heatmap_path, x=10, y=100, w=190)
    pdf.add_page()

    # Add 3D Bar Chart
    pdf.image(bar3d_path, x=10, y=100, w=190)
    pdf.output("/tmp/report.pdf")
    return "/tmp/report.pdf"

# Streamlit Interface
st.title("🌟 Advanced SWOT Leadership Analysis 🌟")

# Sidebar
st.sidebar.markdown(f"### **AI by Allam Rafi FKUI 2022**")
st.sidebar.markdown("""
👨‍⚕️ **About Me**  
I am a **Medical Student** with a strong passion for **Machine Learning**, **Leadership Research**, and **Healthcare AI**.  
- **Education**: Faculty of Medicine, Universitas Indonesia  
- **Research Interests**:  
  - Leadership Viability in Healthcare  
  - AI-driven solutions for medical challenges  
  - Natural Language Processing and Behavioral Analysis  
- **Skills**: Python, NLP, Data Visualization
""")
st.sidebar.image("https://via.placeholder.com/150", caption="Muhammad Allam Rafi", use_column_width=True)
st.sidebar.markdown(f"📫 **Contact**\n\n- LinkedIn: [LinkedIn](https://linkedin.com)\n- GitHub: [GitHub](https://github.com)\n- Email: allamrafi@example.com")
st.sidebar.markdown(f"---\n**{WATERMARK}**")

# Add behavioral questions
behavioral_questions = [
    "Describe how you handle stress.",
    "What motivates you to lead?",
    "Share an example of when you resolved a conflict effectively."
]
behavioral_responses = {q: st.text_area(q) for q in behavioral_questions}

# Collect SWOT inputs
swot_inputs = {cat: [(st.text_area(f"{cat} #{i+1}"), st.slider(f"{cat} #{i+1} Confidence", 1, 10, 5)) for i in range(3)] for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}

if st.button("Analyze"):
    # Analyze SWOT inputs
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

    # Analyze Behavioral Responses
    behavioral_scores, behavioral_explanations = analyze_behavioral_responses(behavioral_responses)
    behavioral_relevance = map_behavioral_to_swot(behavioral_scores)

    # Calculate LSI
    lsi = calculate_lsi(swot_scores, behavioral_relevance)
    lsi_interpretation = interpret_lsi(lsi)

    # Display Results
    st.subheader(f"Leadership Viability Index (LSI): {lsi:.2f}")
    st.write(f"**Interpretation**: {lsi_interpretation}")

    # Display Behavioral Analysis
    st.subheader("Behavioral Analysis")
    for question, explanations in behavioral_explanations.items():
        st.write(f"**{question}**")
        for trait, explanation in explanations.items():
            st.write(f"- {trait}: {explanation}")

    # Generate Visualizations
    heatmap_fig = px.imshow(pd.DataFrame(swot_scores).T.fillna(0), title="Trait Alignment Heatmap", color_continuous_scale="Viridis")
    st.plotly_chart(heatmap_fig)
    heatmap_path = "/tmp/heatmap.png"
    heatmap_fig.write_image(heatmap_path)

    bar3d_fig = px.bar_3d(x=list(swot_scores.keys()), y=list(swot_scores.values()), z=[], color=[], title="3D SWOT")
    st.plotly_chart(bar3d_fig)

    pdf_path = generate_pdf_report(swot_scores, behavioral_scores, lsi, lsi_interpretation, heatmap_path, "/tmp/bar.png")
