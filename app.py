import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px
import datetime

# Set Streamlit page configuration (must be first command)
st.set_page_config(page_title="SWOT-Based Leadership Analysis", page_icon="🌟", layout="wide")

# Load NLP model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

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
BEHAVIORAL_QUESTIONS = [
    "Describe how you handle stressful situations (in English or Indonesian).",
    "Explain your approach to team leadership (in English or Indonesian).",
    "What motivates you to lead others? (in English or Indonesian).",
    "How do you make decisions under pressure? (in English or Indonesian).",
    "Describe a situation where you resolved a conflict (in English or Indonesian)."
]
WATERMARK = "AI by Allam Rafi FKUI 2022"

# Analyze NLP Inputs
def analyze_text_with_confidence(text, qualities, confidence, category_weight):
    scores = {}
    explanations = {}
    for trait, description in qualities.items():
        trait_embedding = model.encode(description, convert_to_tensor=True)
        text_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, trait_embedding).item()
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score
        explanations[trait] = (
            f"Input: '{text}' aligns with '{trait}' ({description}). "
            f"Similarity: {similarity:.2f}. Confidence: {confidence:.2f}. "
            f"Weighted score: {weighted_score:.2f}."
        )
    return scores, explanations

# Generate Bar Chart
def generate_bar_chart(scores, category):
    plt.figure(figsize=(10, 6))
    traits = list(scores.keys())
    values = list(scores.values())
    cmap = ListedColormap(plt.cm.tab20.colors[:len(traits)])
    colors = cmap(np.arange(len(traits)))

    plt.barh(traits, values, color=colors)
    plt.xlabel("Scores")
    plt.ylabel("Traits")
    plt.title(f"{category} Analysis", fontsize=14)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer

# Generate Radar Chart
def generate_radar_chart(scores, category):
    labels = list(scores.keys())
    values = list(scores.values()) + [list(scores.values())[0]]  # Close the radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"{category} Radar Chart", fontsize=14)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer

# Generate 3D Scatter Plot
def generate_3d_scatter(scores):
    fig = go.Figure()
    for category, data in scores.items():
        for text, traits in data.items():
            fig.add_trace(go.Scatter3d(
                x=list(traits.values()), y=list(range(len(traits))), z=list(traits.values()),
                mode='markers', marker=dict(size=8),
                name=f"{category}: {text[:10]}..."
            ))
    fig.update_layout(title="3D SWOT Impact Visualization", scene=dict(
        xaxis_title="Trait Scores", yaxis_title="Traits", zaxis_title="Categories"
    ))
    return fig

# Generate Heatmap
def generate_heatmap(scores):
    df = pd.DataFrame(scores).T
    fig = px.imshow(df, title="Trait Alignment Heatmap", color_continuous_scale="Viridis")
    return fig

# PDF Report Class
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'SWOT-Based Leadership Evaluation Report', align='C', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Page {self.page_no()} - {WATERMARK}", align='C')

    def add_section(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_paragraph(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def add_image(self, img_buffer):
        self.image(img_buffer, x=10, y=self.get_y(), w=190)
        self.ln(65)
        
# Sidebar content
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

st.sidebar.markdown("""
📫 **Contact**  
- [LinkedIn](https://linkedin.com)  
- [GitHub](https://github.com)  
- [Email](mailto:allamrafi@example.com)  
""")

st.sidebar.markdown(f"---\n**{WATERMARK}**")
# Streamlit App
st.sidebar.markdown(f"**{WATERMARK}**")
st.title("🌟 Advanced SWOT-Based Leadership Analysis 🌟")

# Behavioral Analysis
st.header("Behavioral Analysis")
behavioral_responses = {q: st.text_area(q) for q in BEHAVIORAL_QUESTIONS}

# SWOT Analysis
st.header("SWOT Analysis")
swot_inputs = {}
for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
    st.subheader(f"{category}")
    entries = []
    for i in range(3):
        text = st.text_area(f"{category} Aspect #{i+1}")
        confidence = st.slider(f"Confidence for {category} Aspect #{i+1}", 1, 10, 5)
        if text.strip():
            entries.append((text, confidence))
    swot_inputs[category] = entries

if st.button("Analyze"):
    swot_scores = {}
    swot_explanations = {}
    for category, inputs in swot_inputs.items():
        category_scores = {}
        category_explanations = {}
        for text, confidence in inputs:
            scores, explanations = analyze_text_with_confidence(
                text, LEADERSHIP_QUALITIES, confidence, CATEGORY_WEIGHTS[category]
            )
            category_scores[text] = scores
            category_explanations[text] = explanations
        swot_scores[category] = category_scores
        swot_explanations[category] = category_explanations

    # Display Results
    st.subheader("Results")
    for category, traits in swot_scores.items():
        st.subheader(f"{category} Breakdown")
        for text, scores in traits.items():
            st.write(f"Input: {text}")
            for trait, score in scores.items():
                st.write(f"{trait}: {score:.2f}")
            bar_chart = generate_bar_chart(scores, category)
            st.image(bar_chart, caption=f"{category} Bar Chart", use_column_width=True)

    # Visualizations
    st.plotly_chart(generate_3d_scatter(swot_scores))
    st.plotly_chart(generate_heatmap(swot_scores))

    # PDF Report
    pdf = PDFReport()
    pdf.add_page()
    pdf.output("/tmp/report.pdf")
    st.download_button("Download Full Report", open("/tmp/report.pdf", "rb"), file_name="Leadership_Report.pdf", mime="application/pdf")
