import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io

# Load multilingual NLP model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Leadership qualities for evaluation
LEADERSHIP_QUALITIES = {
    "Leadership": "Ability to lead and inspire others",
    "Influence": "Capability to motivate and guide people",
    "Vision": "Having a clear and inspiring direction for the future",
    "Integrity": "Consistently acting with honesty and strong ethics",
    "Confidence": "Demonstrating self-assuredness in decisions and actions",
    "Empathy": "Ability to understand and share others' feelings",
    "Team Collaboration": "Effectiveness in working with others to achieve common goals",
    "Conflict Resolution": "Managing and resolving disagreements constructively",
    "Strategic Thinking": "Ability to set long-term goals and plans",
    "Decision-Making": "Making effective choices under uncertainty or pressure",
    "Adaptability": "Flexibility to adjust to changing circumstances",
    "Time Management": "Prioritizing tasks to achieve efficiency and productivity",
    "Goal Orientation": "Focusing on achieving specific objectives",
    "Problem-Solving": "Analyzing and resolving challenges effectively",
    "Resilience": "Recovering quickly from setbacks and staying focused",
    "Crisis Management": "Leading effectively during high-stress situations"
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.2, "Opportunities": 1.4, "Threats": 1.1}
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Calculate dynamic scores and explanations
def calculate_scores_and_explanations(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}

    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        input_keywords = set(text.lower().split())
        trait_keywords = set(description.lower().split())
        overlapping_keywords = input_keywords.intersection(trait_keywords)
        missing_keywords = trait_keywords - input_keywords

        explanation = f"Your input ('{text}') was compared to the trait '{trait}' ({description}).\n"
        if similarity > 0.7:
            explanation += (
                f"Strong match with a similarity score of {similarity:.2f}. "
                f"The input contains relevant elements such as {', '.join(overlapping_keywords) if overlapping_keywords else 'key leadership aspects'}."
            )
        elif similarity > 0.4:
            explanation += (
                f"Moderate match with a similarity score of {similarity:.2f}. "
                f"It reflects some aspects of this trait, like {', '.join(overlapping_keywords) if overlapping_keywords else 'general qualities'}, "
                f"but could improve by addressing {', '.join(missing_keywords) if missing_keywords else 'specific areas'}."
            )
        else:
            explanation += (
                f"Weak match with a similarity score of {similarity:.2f}. "
                f"The input lacks alignment with key aspects like {', '.join(missing_keywords) if missing_keywords else 'important qualities'}."
            )

        explanation += f" Final score adjusted by confidence ({confidence}/10) and category weight ({category_weight})."
        explanations[trait] = explanation

    return scores, explanations

# Create colorful 2D visualizations
def create_visualizations(scores, category):
    if not scores:
        return None
    df = pd.DataFrame(list(scores.items()), columns=["Trait", "Score"]).sort_values(by="Score", ascending=False)
    fig = px.bar(
        df, x="Score", y="Trait", orientation="h", title=f"{category} Breakdown",
        color="Score", color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_title="Score", yaxis_title="Traits", template="plotly_dark")
    return fig

# Create 3D visualizations
def create_3d_visualization(scores):
    df = pd.DataFrame(scores.items(), columns=["Category", "Score"])
    df["Random Impact"] = np.random.uniform(0, 1, len(df))
    fig = go.Figure(data=[go.Scatter3d(
        x=df["Category"],
        y=df["Score"],
        z=df["Random Impact"],
        mode='markers',
        marker=dict(size=10, color=df["Score"], colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='Category',
        yaxis_title='Score',
        zaxis_title='Random Impact'
    ))
    return fig

# Leadership Viability Index (LSI)
def calculate_lsi(scores, behavioral_score):
    positive = scores.get("Strengths", 0) + scores.get("Opportunities", 0) + behavioral_score
    negative = scores.get("Weaknesses", 0) + scores.get("Threats", 0)
    return np.log((positive / (negative + 1e-9)) + 1e-9)

# PDF Generation
class AdvancedPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'SWOT Leadership Evaluation Report', align='C', ln=True)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, WATERMARK, align='R', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_section(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_paragraph(self, text):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def add_table(self, headers, data):
        self.set_font('Arial', 'B', 12)
        for header in headers:
            self.cell(40, 10, header, border=1, align='C')
        self.ln()
        self.set_font('Arial', '', 12)
        for row in data:
            for cell in row:
                self.cell(40, 10, str(cell), border=1)
            self.ln()

    def add_image(self, img_stream, title):
        self.add_section(title)
        img_stream.seek(0)
        self.image(img_stream, x=10, y=self.get_y(), w=190)
        self.ln(85)

    def add_signature(self):
        self.add_section("Authorized by")
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, "Muhammad Allam Rafi, CBOA® CDSP®", ln=True)
        self.ln(10)

def generate_pdf(lsi_score, lsi_interpretation, swot_breakdown, explanations, visualizations):
    pdf = AdvancedPDF()
    pdf.add_page()

    pdf.add_section("Leadership Viability Index")
    pdf.add_paragraph(f"Your Leadership Viability Index (LSI) score is {lsi_score:.2f}.")
    pdf.add_paragraph(lsi_interpretation)

    for category, breakdown in swot_breakdown.items():
        pdf.add_section(f"{category} Breakdown")
        for text, traits in breakdown.items():
            pdf.add_paragraph(f"Input: {text}")
            for trait, score in traits.items():
                pdf.add_paragraph(f"- {trait}: {score:.2f}")
                pdf.add_paragraph(f"  Explanation: {explanations[category][text][trait]}")

    pdf.add_signature()

    pdf_file_path = "/tmp/Leadership_Report.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# Streamlit app
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")
st.markdown(f"**Watermark:** {WATERMARK}")

# Input fields
swot_entries = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category} Inputs")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} #{i + 1}", placeholder=f"Enter a {category} aspect...")
        confidence = st.slider(f"Confidence for {category} #{i + 1}", 1, 10, 5)
        if text:
            entries.append((text, confidence))
    swot_entries[category] = entries

if st.button("Analyze"):
    swot_breakdown = {}
    scores = {}
    explanations = {}
    for category, entries in swot_entries.items():
        breakdown = {}
        category_explanations = {}
        for text, confidence in entries:
            analysis, explanation = calculate_scores_and_explanations(text, LEADERSHIP_QUALITIES, confidence, CATEGORY_WEIGHTS[category])
            breakdown[text] = analysis
            category_explanations[text] = explanation
        swot_breakdown[category] = breakdown
        explanations[category] = category_explanations
        scores[category] = sum([sum(analysis.values()) for analysis in breakdown.values()]) if breakdown else 0

    lsi_score = calculate_lsi(scores, 0)  # Example behavioral_score placeholder
    lsi_interpretation = "Exceptional Leadership Potential. Highly suited for leadership roles."

    st.metric("Leadership Viability Index (LSI)", f"{lsi_score:.2f}")

    # Generate PDF
    pdf_file = generate_pdf(lsi_score, lsi_interpretation, swot_breakdown, explanations, {})
    st.download_button(
        label="Download Full Report",
        data=open(pdf_file, "rb"),
        file_name="Leadership_Report.pdf",
        mime="application/pdf"
    )
