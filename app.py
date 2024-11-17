import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# Streamlit Page Configuration
st.set_page_config(page_title="🌟 SWOT Leadership Analysis 🌟", page_icon="🌟", layout="wide")

# Define Watermark
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

# Sidebar Information
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="Muhammad Allam Rafi", use_column_width=True)
    st.markdown("### **🌟 About Me 🌟**")
    st.markdown("""
    👨‍⚕️ **Medical Student**  
    Passionate about **Machine Learning**, **Leadership Research**, and **Healthcare AI**.  
    - 🎓 **Faculty of Medicine**, Universitas Indonesia  
    - 📊 **Research Interests**:  
      - Leadership Viability in Healthcare  
      - AI-driven solutions for medical challenges  
      - Natural Language Processing and Behavioral Analysis  
    - 🧑‍💻 **Skills**: Python, NLP, Data Visualization
    """)
    st.markdown("### **📫 Contact Me**")
    st.markdown("""
    - [LinkedIn](https://linkedin.com)  
    - [GitHub](https://github.com)  
    - [Email](mailto:allamrafi@example.com)  
    """)
    st.markdown(f"---\n🌟 **{WATERMARK}** 🌟")

# Header Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌟 SWOT Leadership Analysis 🌟</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #808080;'>Advanced Evaluation of Leadership Potential</h3>", unsafe_allow_html=True)
st.markdown("---")

# Validate Inputs
def validate_inputs(swot_inputs, behavior_inputs):
    for category, inputs in swot_inputs.items():
        for text, _ in inputs:
            if text.strip():
                return True
    for response in behavior_inputs.values():
        if response.strip():
            return True
    return False

# Generate Bar Chart
def generate_bar_chart(data, output_path):
    categories = list(data.keys())
    values = []

    for cat in categories:
        if isinstance(data[cat], dict) and data[cat]:
            avg_score = np.mean(list(data[cat].values()))
            values.append(avg_score)
        else:
            values.append(0)

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=['green', 'red', 'blue', 'orange'])
    plt.title("SWOT Analysis Summary")
    plt.xlabel("Categories")
    plt.ylabel("Scores")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Generate Heatmap
def generate_heatmap(data, output_path):
    df = pd.DataFrame(data).T.fillna(0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", cbar=True)
    plt.title("SWOT Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Generate PDF Report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "SWOT-Based Leadership Evaluation Report", align='C', ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {WATERMARK}", align='C')

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)

def generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_results, chart_paths):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Leadership Viability Index (LSI): {lsi:.2f}", ln=True)
    pdf.cell(0, 10, f"Interpretation: {lsi_interpretation}", ln=True)

    pdf.add_section("Behavioral Analysis Results", "\n".join([f"{q}: {a}" for q, a in behavior_results.items()]))

    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))

    for chart_path in chart_paths:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=50, w=190)
    pdf.output("/tmp/report.pdf")
    return "/tmp/report.pdf"

# Input Collection
swot_inputs = {cat: [(st.text_area(f"{cat} #{i+1}"), st.slider(f"{cat} #{i+1} Confidence", 1, 10, 5)) for i in range(3)] for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}
behavior_inputs = {q: st.text_area(q) for q in ["Describe how you handle stress.", "What motivates you to lead?"]}

if st.button("Analyze"):
    if not validate_inputs(swot_inputs, behavior_inputs):
        st.warning("Please provide at least one valid input for SWOT or Behavioral analysis.")
    else:
        st.success("Analysis Completed! Check Results Below.")
