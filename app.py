import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fpdf import FPDF
from datetime import datetime

# Streamlit Config
st.set_page_config(page_title="Advanced SWOT Leadership Analysis", page_icon="🌟", layout="wide")

# Define Watermark
WATERMARK = "AI by Muhammad Allam Rafi, FKUI 2022"

# Load NLP Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define Leadership Traits
LEADERSHIP_QUALITIES = {
    "Positive": {
        "Leadership": "Ability to lead and inspire others (Internal Strength).",
        "Vision": "Clear and inspiring direction for the future (Internal Strength).",
        "Integrity": "Acting consistently with honesty and strong ethics (Internal Strength).",
        "Innovation": "Driving creativity and fostering change (Internal Strength).",
        "Inclusivity": "Promoting diversity and creating an inclusive environment (Internal Strength).",
        "Empathy": "Understanding others' perspectives and feelings (Internal Strength).",
        "Communication": "Conveying ideas clearly and effectively (Internal Strength).",
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
    },
    "External Opportunities": {
        "Emerging Markets": "New geographical or sectoral markets that offer growth potential.",
        "Technological Advancements": "Technological developments that can be leveraged for innovation.",
        "Regulatory Changes": "Changes in regulations that create opportunities for growth or market entry.",
        "Partnership Opportunities": "Potential collaborations with other organizations for mutual benefit.",
        "Globalization": "Access to international markets and partnerships as a result of global interconnectedness."
    },
    "External Threats": {
        "Academic Pressure": "Challenges arising from academic performance or deadlines.",
        "Extracurricular Commitments": "Overinvolvement in external activities that could detract from leadership focus.",
        "Economic Downturn": "External economic conditions affecting organizational success.",
        "Technological Disruptions": "External technological advancements that might reduce the relevance of current leadership strategies.",
        "Market Competition": "Increased competition in the job market or in a specific field."
    }
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.3, "Opportunities": 1.4, "Threats": 1.2}

# Collect Inputs
def collect_inputs():
    swot_inputs = {cat: [(st.text_area(f"{cat} #{i+1}"), st.slider(f"{cat} #{i+1} Confidence", 1, 10, 5)) for i in range(3)] for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}

    behavior_questions = {
        "Q1": "Describe how you handle stress.",
        "Q2": "What motivates you to lead others?",
        "Q3": "How do you approach conflict resolution?",
        "Q4": "What is your strategy for long-term planning?",
        "Q5": "How do you inspire teamwork in challenging situations?"
    }

    behavior_inputs = {q: st.text_area(q) for q in behavior_questions.values()}
    
    return swot_inputs, behavior_inputs

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
    for category, inputs in swot_inputs.items():
        for text, _ in inputs:
            if text.strip():  # Valid text
                return True
    for response in behavior_inputs.values():
        if response.strip():
            return True
    return False  # If all inputs are empty

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

# Main logic
if st.button("Analyze"):
    swot_inputs, behavior_inputs = collect_inputs()
    
    if not validate_inputs(swot_inputs, behavior_inputs):
        st.warning("Please provide at least one valid input for SWOT or Behavioral analysis.")
    else:
        swot_scores, swot_explanations = {}, {}

        for category, inputs in swot_inputs.items():
            category_scores, category_explanations = {}, {}

            if category == "Strengths":
                qualities = LEADERSHIP_QUALITIES["Positive"]
            elif category == "Weaknesses":
                qualities = LEADERSHIP_QUALITIES["Negative"]
            elif category == "Opportunities":
                qualities = LEADERSHIP_QUALITIES["External Opportunities"]
            elif category == "Threats":
                qualities = LEADERSHIP_QUALITIES["External Threats"]

            for text, confidence in inputs:
                scores, explanations = analyze_text_with_explanation(text, qualities, confidence, CATEGORY_WEIGHTS.get(category, 1))
                category_scores.update(scores)
                category_explanations.update(explanations)

            swot_scores[category] = category_scores
            swot_explanations[category] = category_explanations

        total_strengths = sum(swot_scores.get("Strengths", {}).values())
        total_opportunities = sum(swot_scores.get("Opportunities", {}).values())
        total_weaknesses = sum(swot_scores.get("Weaknesses", {}).values())
        total_threats = sum(swot_scores.get("Threats", {}).values())

        if any(np.isnan(value) or np.isinf(value) for value in [total_strengths, total_opportunities, total_weaknesses, total_threats]):
            st.warning("Invalid data encountered. Please review your input values.")

        lsi = total_strengths + total_opportunities - total_weaknesses - total_threats
        if lsi > 8:
            lsi_interpretation = "Excellent Leadership Potential."
        elif lsi > 5:
            lsi_interpretation = "Good Leadership Potential."
        else:
            lsi_interpretation = "Leadership Potential Needs Improvement."
        
        # Chart generation (Optional for visual feedback)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(total_strengths, total_opportunities, total_weaknesses, c='r', marker='o')
        ax.set_xlabel('Strengths')
        ax.set_ylabel('Opportunities')
        ax.set_zlabel('Weaknesses')
        plt.savefig("/tmp/leadership_chart.png")

        behavior_results = {
            "How you handle stress": behavior_inputs.get("Q1", ""),
            "What motivates you": behavior_inputs.get("Q2", ""),
            "How you approach conflict": behavior_inputs.get("Q3", ""),
            "Long-term planning strategy": behavior_inputs.get("Q4", ""),
            "Inspiring teamwork": behavior_inputs.get("Q5", "")
        }

        chart_paths = ["/tmp/leadership_chart.png"]
        pdf_report_path = generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_results, chart_paths)
        st.download_button("Download Report", pdf_report_path, file_name="Leadership_Report.pdf")
