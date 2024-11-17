import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fpdf import FPDF
from datetime import datetime

# Config Streamlit
st.set_page_config(page_title="SWOT Leadership Analysis", page_icon="🌟", layout="wide")

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

# Analyze Text with NLP
def analyze_text_with_explanation(text, qualities, confidence, category_weight):
    if not text.strip():
        return {}, {}
    scores, explanations = {}, {}
    embeddings = model.encode([text] + list(qualities.values()), convert_to_tensor=True)
    text_embedding, trait_embeddings = embeddings[0], embeddings[1:]
    similarities = util.pytorch_cos_sim(text_embedding, trait_embeddings).squeeze().tolist()
    
    for trait, similarity in zip(qualities.keys(), similarities):
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = weighted_score
        explanations[trait] = f"Input aligns with '{trait}'. Similarity: {similarity:.2f}, Weighted Score: {weighted_score:.2f}."
    return scores, explanations

# Validate Inputs
def validate_inputs(swot_inputs, behavior_inputs):
    # Validasi SWOT
    for category, inputs in swot_inputs.items():
        for text, _ in inputs:
            if text.strip():  # Jika ada teks valid
                return True
    # Validasi Behavior
    for response in behavior_inputs.values():
        if response.strip():
            return True
    return False  # Semua input kosong

# Generate 2D Charts
def generate_bar_chart(data, output_path):
    categories = list(data.keys())
    values = [np.mean(data[cat]) for cat in categories]

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
    plt.imshow(df, cmap="viridis", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Scores")
    plt.title("SWOT Heatmap")
    plt.xlabel("Traits")
    plt.ylabel("Categories")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Generate 3D Charts
def generate_3d_scatter(data, output_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    categories = list(data.keys())
    for category_idx, (category, traits) in enumerate(data.items()):
        xs = np.arange(len(traits))
        ys = list(traits.values())
        zs = category_idx
        ax.scatter(xs, [zs]*len(xs), ys, label=category)
    ax.set_title("3D Scatter Plot - SWOT Scores")
    ax.set_xlabel("Traits")
    ax.set_ylabel("Categories")
    ax.set_zlabel("Scores")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def generate_3d_surface(data, output_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    categories = list(data.keys())
    x = np.arange(len(categories))
    y = np.arange(len(next(iter(data.values()))))
    x, y = np.meshgrid(x, y)
    z = np.array([list(traits.values()) for traits in data.values()])
    ax.plot_surface(x, y, z, cmap="viridis", edgecolor='k')
    ax.set_title("3D Surface Plot - SWOT Scores")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Traits")
    ax.set_zlabel("Scores")
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

    # Add Behavior Results
    pdf.add_section("Behavioral Analysis Results", "\n".join([f"{q}: {a}" for q, a in behavior_results.items()]))

    # Add SWOT Scores
    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))

    # Add Charts
    for chart_path in chart_paths:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=50, w=190)
    pdf.output("/tmp/report.pdf")
    return "/tmp/report.pdf"

# Behavior Questions
behavior_questions = {
    "Q1": "Describe how you handle stress.",
    "Q2": "What motivates you to lead others?",
    "Q3": "How do you approach conflict resolution?",
    "Q4": "What is your strategy for long-term planning?",
    "Q5": "How do you inspire teamwork in challenging situations?"
}

# Collect Inputs
swot_inputs = {cat: [(st.text_area(f"{cat} #{i+1}"), st.slider(f"{cat} #{i+1} Confidence", 1, 10, 5)) for i in range(3)] for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}
behavior_inputs = {q: st.text_area(q) for q in behavior_questions.values()}

if st.button("Analyze"):
    if not validate_inputs(swot_inputs, behavior_inputs):
        st.warning("Please provide at least one valid input for SWOT or Behavioral analysis.")
    else:
        # Analyze SWOT
        swot_scores, swot_explanations = {}, {}
        for category, inputs in swot_inputs.items():
            category_scores, category_explanations = {}, {}
            qualities = (
                LEADERSHIP_QUALITIES["Positive"] if category in ["Strengths", "Opportunities"]
                else LEADERSHIP_QUALITIES["Negative"] if category == "Threats"
                else LEADERSHIP_QUALITIES["Neutral"]
            )
            for text, confidence in inputs:
                scores, explanations = analyze_text_with_explanation(text, qualities, confidence, CATEGORY_WEIGHTS[category])
                category_scores.update(scores)
                category_explanations.update(explanations)
            swot_scores[category] = category_scores

        # Analyze Behavior
        behavior_scores = {}
        for question, response in behavior_inputs.items():
            if response.strip():
                scores, _ = analyze_text_with_explanation(response, LEADERSHIP_QUALITIES["Positive"], 5, 1.0)
                behavior_scores[question] = scores

        # Calculate LSI
        total_strengths = sum(swot_scores["Strengths"].values())
        total_weaknesses = sum(swot_scores["Weaknesses"].values())
        lsi = np.log((total_strengths + 1) / (total_weaknesses + 1))
        lsi_interpretation = (
            "Exceptional Leadership Potential" if lsi > 1.5 else
            "Good Leadership Potential" if lsi > 0.5 else
            "Moderate Leadership Potential" if lsi > -0.5 else
            "Needs Improvement"
        )

        # Display LSI and Interpretation
        st.subheader(f"Leadership Viability Index (LSI): {lsi:.2f}")
        st.write(f"**Interpretation**: {lsi_interpretation}")

        # Generate and Display Charts
        bar_chart_path = "/tmp/bar_chart.png"
        heatmap_path = "/tmp/heatmap.png"
        scatter_chart_path = "/tmp/scatter_chart.png"
        surface_chart_path = "/tmp/surface_chart.png"
        generate_bar_chart(swot_scores, bar_chart_path)
        generate_heatmap(swot_scores, heatmap_path)
        generate_3d_scatter(swot_scores, scatter_chart_path)
        generate_3d_surface(swot_scores, surface_chart_path)

        st.image(bar_chart_path, caption="SWOT Bar Chart")
        st.image(heatmap_path, caption="SWOT Heatmap")

        # Display 3D Charts
        st.image(scatter_chart_path, caption="3D Scatter Plot")
        st.image(surface_chart_path, caption="3D Surface Plot")

        # Generate PDF Report
        pdf_path = generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_inputs, [bar_chart_path, heatmap_path, scatter_chart_path, surface_chart_path])
        with open(pdf_path, "rb") as f:
            st.download_button("Download Professional PDF Report", f, "Leadership_Report.pdf", mime="application/pdf")
