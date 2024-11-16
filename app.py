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

# Calculate similarity scores
def calculate_scores(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}

    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score

        # Generate dynamic explanation
        explanation = f"Your input ('{text}') was compared to the trait '{trait}': {description}.\n"
        if similarity > 0.7:
            explanation += f"The input strongly aligns with this trait (similarity: {similarity:.2f}), reflecting direct relevance. "
        elif similarity > 0.4:
            explanation += f"The input moderately aligns with this trait (similarity: {similarity:.2f}), suggesting partial relevance. "
        else:
            explanation += f"The input weakly aligns with this trait (similarity: {similarity:.2f}), indicating limited relevance. "
        explanation += f"Final score is adjusted by confidence ({confidence}/10) and category weight ({category_weight})."
        explanations[trait] = explanation

    return scores, explanations

# Create colorful 2D visualizations
def create_visualizations(scores, category):
    if not scores:
        st.warning(f"No valid scores for {category}. Please provide valid inputs.")
        return None
    df = pd.DataFrame(list(scores.items()), columns=["Trait", "Score"]).sort_values(by="Score", ascending=False)
    fig = px.bar(
        df, x="Score", y="Trait", orientation="h", title=f"{category} Breakdown",
        color="Score", color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_title="Score", yaxis_title="Traits", template="plotly_dark")
    return fig

# Create 3D scatter plot
def create_3d_visualization(scores):
    df = pd.DataFrame(scores.items(), columns=["Category", "Score"])
    df["Random Impact"] = np.random.uniform(0, 1, len(df))  # Add random impact for visualization
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
def calculate_lsi(scores, behavioral_score, inconsistencies):
    epsilon = 1e-9
    positive = scores["Strengths"] + scores["Opportunities"] + behavioral_score
    negative = scores["Weaknesses"] + scores["Threats"]
    inconsistency_penalty = len(inconsistencies) * 0.1
    return np.log((positive / (negative + inconsistency_penalty + epsilon)) + epsilon)

# Interpret LSI
def interpret_lsi(lsi_score):
    if lsi_score > 1.5:
        return "Exceptional Leadership Potential. Highly suited for leadership roles."
    elif 1.0 < lsi_score <= 1.5:
        return "Strong Leadership Potential. Suitable for leadership but with minor areas for improvement."
    elif 0.5 < lsi_score <= 1.0:
        return "Moderate Leadership Potential. Notable areas for improvement exist."
    elif 0.0 < lsi_score <= 0.5:
        return "Low Leadership Potential. Requires significant development in key areas."
    else:
        return "Poor Leadership Fit. Needs substantial improvement before pursuing leadership roles."

# Generate PDF report
class AdvancedPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'SWOT-Based Leadership Evaluation Report', ln=True, align='C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, WATERMARK, align='R', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Page ' + str(self.page_no()), align='C')

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

def generate_pdf(lsi_score, swot_breakdown, explanations, radar_chart_stream, bar_chart_stream):
    pdf = AdvancedPDF()
    pdf.add_page()

    # Header and LSI Score
    pdf.add_section("Leadership Viability Index (LSI)")
    pdf.add_paragraph(f"Your LSI score is {lsi_score:.2f}. {interpret_lsi(lsi_score)}")

    # SWOT Breakdown with Explanations
    for category, breakdown in swot_breakdown.items():
        pdf.add_section(f"{category} Breakdown")
        for text, traits in breakdown.items():
            pdf.add_paragraph(f"Input: {text}")
            for trait, score in traits.items():
                pdf.add_paragraph(f"- {trait}: {score:.2f}")
                pdf.add_paragraph(f"  Explanation: {explanations[category][text][trait]}")

    # Visualizations
    pdf.add_image(radar_chart_stream, "Radar Chart")
    pdf.add_image(bar_chart_stream, "Bar Chart")

    # Signature
    pdf.add_section("Disahkan oleh")
    pdf.add_paragraph("Muhammad Allam Rafi, CBOA® CDSP®")

    pdf_file = "/tmp/Leadership_Report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Streamlit App
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")
st.markdown(f"**Watermark:** {WATERMARK}")

# Input
swot_entries = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category} Inputs")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} #{i + 1}", placeholder=f"Enter a {category} aspect (in English or Indonesian)...")
        confidence = st.slider(f"Confidence for {category} #{i + 1}", 1, 10, 5)
        if text:
            entries.append((text, confidence))
    swot_entries[category] = entries

# Behavioral Evidence Section
st.subheader("Behavioral Evidence")
behavioral_examples = []
for i in range(2):
    example = st.text_area(f"Behavioral Example #{i + 1}", placeholder="Describe a leadership scenario...")
    if example:
        behavioral_examples.append(example)

# Analyze Button
if st.button("Analyze"):
    swot_breakdown = {}
    scores = {}
    explanations = {}
    for category, entries in swot_entries.items():
        breakdown = {}
        category_explanations = {}
        for text, confidence in entries:
            if text.strip():  # Only process valid inputs
                analysis, explanation = calculate_scores(text, LEADERSHIP_QUALITIES, confidence, CATEGORY_WEIGHTS[category])
                breakdown[text] = analysis
                category_explanations[text] = explanation
        swot_breakdown[category] = breakdown
        explanations[category] = category_explanations
        scores[category] = sum([sum(analysis.values()) for analysis in breakdown.values()]) if breakdown else 0

    behavioral_analysis = {example: len(example.split()) / 50 for example in behavioral_examples}
    lsi_score = calculate_lsi(scores, np.mean(list(behavioral_analysis.values())), [])

    st.metric("Leadership Viability Index (LSI)", f"{lsi_score:.2f}")
    st.markdown(f"**Interpretation:** {interpret_lsi(lsi_score)}")

    # Display visualizations and explanations
    for category, breakdown in swot_breakdown.items():
        st.subheader(f"{category} Breakdown")
        for text, analysis in breakdown.items():
            st.write(f"**Input**: {text}")
            for trait, score in analysis.items():
                st.write(f"- **{trait}**: {score:.2f}")
                st.write(f"  - Explanation: {explanations[category][text][trait]}")

        # Generate and display colorful visualization
        fig = create_visualizations(scores[category], category)
        if fig:
            st.plotly_chart(fig)

    # Add 3D Visualization
    st.subheader("3D Visualization of SWOT Impact")
    fig_3d = create_3d_visualization(scores)
    st.plotly_chart(fig_3d)
