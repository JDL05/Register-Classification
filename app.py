import streamlit as st
import pandas as pd
import os
import json
from tqdm import tqdm

# Enable tqdm for pandas (for local progress feedback; you might disable this if it adds overhead)
tqdm.pandas()

# -------------------------------------------
# Session State Initialization
# -------------------------------------------
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Home"
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "prev_threshold" not in st.session_state:
    st.session_state.prev_threshold = 0

# -------------------------------------------
# Persistent File Names
# -------------------------------------------
WEIGHTS_FILE = "positive_weights.json"
SCRORED_FILE = "companies_scored.csv"
LABELED_FILE = "labeled_results.csv"

# -------------------------------------------
# Base Positive Keywords with Weights (Base)
# -------------------------------------------
base_positive_keywords = {
    "Entwicklung": 1,
    "ecommerce": 1,
    "digital": 1,
    "digitale": 1,
    "digitalen": 1,
    "Onlinehandel": 2,
    "SaaS": 2,
    "Plattform": 1,
    "Web": 1,
    "Mobilgeräte": 1,
    "Data Science": 2,
    "Data": 1,
    "Daten": 1,
    "künstliche": 1,
    "Intelligenz": 1,
    "KI": 1,
    "AI": 1,
    "Technologie": 1,
    "Technologien": 1,
    "CO2": 1,
    "Software": 1,
    "Development": 1,
    "Forschung": 1,
    "Innovation": 2,
    "Quantentechnologie": 2,
    "Quanten": 1,
    "intelligent": 1,
    "intelligente": 1,
    "Computer": 1,
    "Vision": 1,
    "Erforschung": 1,
    "Blockchain": 2,
    "Virtual": 1,
    "virtuell": 1,
    "Realität": 1,
    "Reality": 1,
    "Softwareanwendung": 1,
    "Softwareanwendungen": 1,
    "Softwarelösungen": 1,
    "Chatbot": 1,
    "Anwendungen": 1,
    "Treibhausgas": 1,
    "Treibhausgase": 1,
    "Emissionen": 1,
    "Treibhausgasemissionen": 1,
    "datengestützt": 1,
    "datengestützte": 1,
    "datengestützten": 1,
    "Cloud": 1,
    "Analyse": 1,
    "Softwareentwicklung": 1,
    "IT-Systeme": 1,
    "Sensorik": 1,
    "medizinisch": 1,
    "Dokumentation": 1,
    "Informationstechnik": 1,
    "Informationstechnologien": 1,
    "Hardwareentwicklung": 1,
    "Algorithmen": 1,
    "Algorithmus": 1,
    "Artificial Intelligence": 1,
    "autonom": 1,
    "autonomes": 1
}

# -------------------------------------------
# No-go Keywords (fixed weight)
# -------------------------------------------
no_go_list = [
    "Wartung", "Consulting", "Beratungstätigkeiten", "jeglicher Art",
    "Elektrodienstleistungen", "Online-Kurse", "Marketingkommunikation",
    "Werbedienstleistungen", "Unternehmensberatung", "Agenturleistungen",
    "Erbringung", "Training", "Kulturorganisationen", "Schmuck", "Accessoires",
    "E-Books", "Vertriebs-Einheit", "Coachings", "Coaching"
]

# -------------------------------------------
# Problematic Names
# -------------------------------------------
problematic_names = [
    "Europe", "Consulting"
]

# -------------------------------------------
# Positive Weights Persistence Functions
# -------------------------------------------
def load_positive_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "r") as f:
            return json.load(f)
    else:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(base_positive_keywords, f)
        return base_positive_keywords.copy()

def update_positive_weights(description, learning_rate=0.1):
    weights = load_positive_weights()
    desc = description.lower()
    updated = False
    for kw in weights:
        if kw.lower() in desc:
            weights[kw] += learning_rate
            updated = True
    if updated:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f)
    return weights

# -------------------------------------------
# Caching Data-Intensive Operations
# -------------------------------------------
@st.cache_data(show_spinner=False)
def load_scored_data(scored_csv=SCRORED_FILE):
    return pd.read_csv(scored_csv)

@st.cache_data(show_spinner=False)
def load_labeled_data(labeled_csv=LABELED_FILE):
    if os.path.exists(labeled_csv):
        return pd.read_csv(labeled_csv)
    else:
        return pd.DataFrame(columns=["company_name", "zip", "description", "is_startup", "source"])

# -------------------------------------------
# Score Computation Functions
# -------------------------------------------
def compute_score(row):
    company_name = str(row["company_name"]).lower()
    description = str(row["description"]).lower()
    
    for pname in problematic_names:
        if pname.lower() in company_name:
            return -100

    weights = load_positive_weights()
    score = 0
    for kw, weight in weights.items():
        if kw.lower() in description:
            score += weight
    for word in no_go_list:
        if word.lower() in description:
            score -= 5
    return score

def process_dataframe(df):
    df["keyword_count"] = df.progress_apply(compute_score, axis=1)
    df = df.sort_values(by="keyword_count", ascending=False).reset_index(drop=True)
    return df

# -------------------------------------------
# Refresh Auto Labels Function
# -------------------------------------------
def refresh_auto_labels(labeled_csv, threshold):
    df_full = pd.read_csv(SCRORED_FILE)
    df_labeled = pd.read_csv(labeled_csv)
    auto_df = df_labeled[df_labeled["source"] == "auto"].copy()
    if auto_df.empty:
        return
    merged = auto_df.merge(df_full[["company_name", "zip", "description", "keyword_count"]],
                           on=["company_name", "zip", "description"], how="left")
    to_remove = merged[merged["keyword_count"] > threshold]
    if not to_remove.empty:
        df_labeled = df_labeled.drop(to_remove.index)
        df_labeled.to_csv(labeled_csv, index=False)

# -------------------------------------------
# Data Loading & Auto-Labeling
# -------------------------------------------
def load_data(scored_csv=SCRORED_FILE, labeled_csv=LABELED_FILE, threshold=0):
    df_full = pd.read_csv(scored_csv)
    
    if not os.path.exists(labeled_csv):
        pd.DataFrame(columns=["company_name", "zip", "description", "is_startup", "source"]).to_csv(labeled_csv, index=False)
    
    df_labeled = pd.read_csv(labeled_csv)
    if "source" not in df_labeled.columns:
        df_labeled["source"] = "manual"
        df_labeled.to_csv(labeled_csv, index=False)

    if threshold < st.session_state.prev_threshold:
        refresh_auto_labels(labeled_csv, threshold)
    
    df_labeled = pd.read_csv(labeled_csv)

    df_no = df_full[df_full["keyword_count"] <= threshold].copy()
    if not df_no.empty:
        df_no_merged = df_no.merge(df_labeled, on=["company_name", "zip", "description"], how="left", indicator=True)
        df_no_unlabeled = df_no_merged[df_no_merged["_merge"] == "left_only"].drop(columns=["_merge"])
        if not df_no_unlabeled.empty:
            df_no_unlabeled["is_startup"] = "No"
            df_no_unlabeled["source"] = "auto"
            df_no_unlabeled[["company_name", "zip", "description", "is_startup", "source"]].to_csv(
                labeled_csv, mode="a", header=False, index=False
            )
    
    df_labeled = pd.read_csv(labeled_csv)

    df_nonzero = df_full[df_full["keyword_count"] > threshold].copy()
    df_merged = df_nonzero.merge(df_labeled, on=["company_name", "zip", "description"], how="left", indicator=True)
    df_todo = df_merged[df_merged["_merge"] == "left_only"].drop(columns=["_merge"])

    total_companies = len(df_full)
    auto_labeled_no = len(df_full[df_full["keyword_count"] <= threshold])
    labeled_yes = sum(df_labeled["is_startup"] == "Yes")
    labeled_no = sum(df_labeled["is_startup"] == "No")
    left_to_label = len(df_todo)

    stats = {
        "total_companies": total_companies,
        "auto_labeled_no": auto_labeled_no,
        "labeled_yes": labeled_yes,
        "labeled_no": labeled_no,
        "left_to_label": left_to_label
    }
    return df_full, df_todo, stats

def save_label(row, label, labeled_file=LABELED_FILE):
    if label == "Yes":
        update_positive_weights(row["description"], learning_rate=0.1)
    new_data = {
        "company_name": row["company_name"],
        "zip": row["zip"],
        "description": row["description"],
        "is_startup": label,
        "source": "manual"
    }
    labeled_df = pd.DataFrame([new_data])
    labeled_df.to_csv(labeled_file, mode="a", header=False, index=False)

# -------------------------------------------
# Classification Interface
# -------------------------------------------
def next_company(row, choice):
    save_label(row, choice)
    st.session_state.current_index += 1

def classification_interface(threshold):
    st.title("Classification Interface")
    labeled_file = LABELED_FILE
    _, df_todo, stats = load_data(SCRORED_FILE, labeled_file, threshold)

    st.markdown("### Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Companies", stats["total_companies"])
    col2.metric("Auto-labeled (<= threshold)", stats["auto_labeled_no"])
    col3.metric("Labeled Yes", stats["labeled_yes"])
    col4.metric("Labeled No", stats["labeled_no"])
    st.write(f"**Left to label (score > {threshold}):** {stats['left_to_label']}")

    if stats["left_to_label"] == 0:
        st.success("All companies have been reviewed!")
        return
    if st.session_state.current_index >= len(df_todo):
        st.success("All remaining companies have been reviewed!")
        return

    current_row = df_todo.iloc[st.session_state.current_index]
    current_progress = (st.session_state.current_index + 1) / len(df_todo)
    st.progress(current_progress)
    st.write(f"Reviewing {st.session_state.current_index + 1} / {len(df_todo)}")

    st.subheader(f"**Company Name:** {current_row['company_name']}")
    st.write(f"**ZIP:** {current_row['zip']}")
    st.write(f"**Description:** {current_row['description']}")
    st.write(f"**Score:** {current_row['keyword_count']}")

    classification_choice = st.radio("Is this a Startup?", ("Yes", "No"), index=0, key="classification_choice")
    st.button("Next", on_click=next_company, args=(current_row, classification_choice))

# -------------------------------------------
# View Classified Startups
# -------------------------------------------
def view_positive_startups():
    st.title("View Classified Startups")
    labeled_file = LABELED_FILE
    if not os.path.exists(labeled_file):
        st.error("No classifications have been made yet!")
        return
    df_labeled = pd.read_csv(labeled_file)
    df_positive = df_labeled[df_labeled["is_startup"] == "Yes"]
    if df_positive.empty:
        st.info("No startups have been classified as positive yet.")
    else:
        st.subheader("List of Classified Startups")
        st.dataframe(df_positive.reset_index(drop=True))
        st.download_button(
            label="Download Classified Startups",
            data=df_positive.to_csv(index=False).encode('utf-8'),
            file_name='classified_startups.csv',
            mime='text/csv'
        )

# -------------------------------------------
# Reset Labels Button
# -------------------------------------------
def reset_labels(labeled_csv=LABELED_FILE):
    if os.path.exists(labeled_csv):
        os.remove(labeled_csv)
    st.session_state.current_index = 0
    st.success("Labeled results have been reset.")

# -------------------------------------------
# Home Screen
# -------------------------------------------
def home_screen():
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    st.title("Welcome to the Startup Classifier App")
    st.markdown("""
        **Overview:**
        
        This application helps classify companies from a public register into startups or non-startups.
        It computes a score for each company using weighted positive keywords, a no-go list, and problematic names.
        Companies with scores above a chosen threshold are flagged for manual review.
        
        **Smart Learning:**
        When you label a company as a startup ("Yes"), the weights for the positive keywords in its description are increased.
        These adjustments persist across sessions.
        
        **Modes:**
        - **Home:** This overview and instructions.
        - **Upload Data:** Upload a CSV file with your company data.
        - **Classification Interface:** Manually label companies with scores above your threshold.
        - **View Classified Startups:** See and download the list of startups you've classified.
        
        **Instructions:**
        1. Start in **Upload Data** to process your CSV file.
        2. Then switch to **Classification Interface** to label companies.
        3. Finally, use **View Classified Startups** to review your results.
    """)
    st.info("Use the sidebar to navigate between modes.")

# -------------------------------------------
# Main App Logic
# -------------------------------------------
def main():
    st.sidebar.title("Navigation")

    mode = st.sidebar.selectbox(
        "Choose the mode",
        ("Home", "Upload Data", "Classification Interface", "View Classified Startups"),
        index=0
    )
    st.session_state.app_mode = mode

    threshold = st.sidebar.number_input(
        "Classification Threshold (Only companies with score > threshold require manual labeling)",
        min_value=0, max_value=100, value=0, step=1
    )

    if st.session_state.prev_threshold != threshold:
        st.session_state.current_index = 0
        # Refresh auto labels when lowering the threshold.
        if threshold < st.session_state.prev_threshold:
            refresh_auto_labels(LABELED_FILE, threshold)
        st.session_state.prev_threshold = threshold

    if st.sidebar.button("Reset All Labels"):
        reset_labels()

    if mode == "Home":
        home_screen()
    elif mode == "Upload Data":
        st.title("Upload and Process Company Data")
        st.write("Upload a CSV file with columns: `company_name`, `zip`, `location`, `description`.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if st.button("Skip Upload and Continue to Classification"):
            st.session_state.app_mode = "Classification Interface"

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return
            st.write("File uploaded successfully. Processing your data... (Check your terminal for progress)")
            df_processed = process_dataframe(df)
            output_filename = SCRORED_FILE
            output_path = os.path.join(os.getcwd(), output_filename)
            df_processed.to_csv(output_path, index=False)
            st.success(f"Processing complete! The file has been saved as '{output_filename}' in {os.getcwd()}.")
            st.info("Now switch to the 'Classification Interface' mode from the sidebar to begin labeling.")
    elif mode == "Classification Interface":
        if not os.path.exists(SCRORED_FILE):
            st.error("The file 'companies_scored.csv' does not exist. Please upload and process a CSV file first.")
        else:
            classification_interface(threshold)
    elif mode == "View Classified Startups":
        view_positive_startups()

if __name__ == "__main__":
    main()
