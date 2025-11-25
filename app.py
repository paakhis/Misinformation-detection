# try.py — RumorRadar: Misinformation Detection & Spread Simulation

import streamlit as st
import joblib
import numpy as np
import random
import time
import networkx as nx
import os
import joblib
import gdown
import matplotlib.pyplot as plt
import nltk
import google.generativeai as genai
import json
from nltk.corpus import stopwords

nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)




# ---------------------------
# 1) Gemini API
# ---------------------------
GEMINI_API_KEY = st.secrets["gemini"]["API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# ---------------------------
# 2) Helper: Ranked Nodes
# ---------------------------
def get_top_nodes(G, count=10, mode="Hub"):
    degs = dict(G.degree())
    if mode == "Hub":
        ranked = sorted(degs, key=lambda n: degs[n], reverse=True)
    elif mode == "Bridge":
        ranked = sorted(degs, key=lambda n: degs[n])
    else:
        ranked = list(G.nodes())
        random.shuffle(ranked)
    return ranked[:count]


# ---------------------------
# 3) Load ML Model Artifacts
# ---------------------------
@st.cache_resource(ttl=3600)

@st.cache_resource
def load_model():
    files = {
        "ensemble_model.pkl": "1eccd0MMj3fW_dzRqPnFSFww_-GXGvlKJ",
        "tfidf_vectorizer.pkl": "15gZuFDfhYr9O33ijMyt_K6DAnGuAuNbG",
        "numeric_scaler.pkl": "1d9mATdGWCuPsJd3UuZkUeqya7KHxOXc2"
    }

    for filename, file_id in files.items():
        if not os.path.exists(filename):
            gdown.download(
                f"https://drive.google.com/uc?export=download&id={file_id}",
                filename,
                quiet=False
            )

    ensemble = joblib.load("ensemble_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    scaler = joblib.load("numeric_scaler.pkl")
    return ensemble, tfidf, scaler


ensemble_model, tfidf, scaler = load_model()


# ---------------------------
# 4) Gemini Fake News Check
# ---------------------------
def classify_with_gemini(text: str) -> str:
    """
    Ask Gemini to classify as Real News / Fake News / Uncertain.
    Returns one of those strings. Falls back to 'Uncertain' on any error.
    """
    prompt = f"""
Classify the following text into exactly one of:
- Real News
- Fake News
- Uncertain

Reply ONLY in JSON like:
{{"label": "Real News"}}

Text:
{text}
"""
    try:
        resp = gemini_model.generate_content(prompt)
        content = resp.text.strip()
        # Extract JSON substring
        json_str = content[content.find("{"): content.rfind("}") + 1]
        data = json.loads(json_str)
        return data.get("label", "Uncertain")
    except Exception:
        return "Uncertain"


# ---------------------------
# 5) SIR Simulation Logic
# ---------------------------
def run_sir_simulation(
    node_count,
    infection_prob,
    recovery_prob,
    start_node,
    propagation_type,
    animate_delay,
    log_area,
    plot_area
):
    """
    Updated SIR simulation with live network visualization.
    """

    # Build graph
    G = nx.barabasi_albert_graph(node_count, 2, seed=42)
    degs = dict(G.degree())

    # Fixed layout (so nodes don’t jump around)
    pos = nx.spring_layout(G, seed=42)

    # S, I, R status map
    status = {n: "S" for n in G.nodes()}
    start_node = max(0, min(start_node, node_count - 1))
    status[start_node] = "I"

    log_box = log_area.container()
    step = 0
    max_steps = 200

    while True:
        step += 1
        new_status = status.copy()

        # -------------------
        # Update Infection / Recovery
        # -------------------
        for node in G.nodes():
            if status[node] == "I":

                # Recovery
                if random.random() < recovery_prob:
                    new_status[node] = "R"
                else:
                    # Spread
                    for nei in G.neighbors(node):
                        if status[nei] != "S":
                            continue

                        mean_deg = np.mean(list(degs.values()))
                        med_deg = np.median(list(degs.values()))

                        # Spread model control
                        spread_ok = True
                        if "Hub" in propagation_type:
                            spread_ok = degs[nei] >= mean_deg
                        elif "Bridge" in propagation_type:
                            spread_ok = degs[nei] <= med_deg

                        if spread_ok and random.random() < infection_prob:
                            new_status[nei] = "I"

        status = new_status

        # Count populations
        S = sum(v == "S" for v in status.values())
        I = sum(v == "I" for v in status.values())
        R = sum(v == "R" for v in status.values())

        # -------------------------------
        #   GRAPH VISUALIZATION SECTION
        # -------------------------------
        fig, ax = plt.subplots(figsize=(7, 6))

        colors = [
            "lightblue" if status[n] == "S"
            else "red" if status[n] == "I"
            else "lightgreen"
            for n in G.nodes()
        ]

        node_sizes = [50 + degs[n] * 10 for n in G.nodes()]

        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            node_size=node_sizes,
            edgecolors="black",
            ax=ax
        )

        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

        # Show node numbers
        nx.draw_networkx_labels(
            G, pos,
            labels={n: n for n in G.nodes()},
            font_size=6,
            font_color="black",
            ax=ax
        )

        ax.set_title(f"Day {step} — S:{S}  I:{I}  R:{R}")
        ax.axis("off")

        plot_area.pyplot(fig)
        plt.close(fig)

        # -------------------
        # Log Output
        # -------------------
        log_box.write(f"### Day {step}")
        log_box.write(f"🟦 Safe (S): {S}")
        log_box.write(f"🔴 Infected (I): {I}")
        log_box.write(f"🟩 Recovered (R): {R}")
        log_box.write(f"Start Node: {start_node}")
        log_box.write(f"Model: {propagation_type}")
        log_box.write("---")

        time.sleep(animate_delay)

        # Stop when spread finishes
        if I == 0 or step >= max_steps:
            break

    log_box.success(" Spread Stopped — Simulation Complete!")


# ============================================================
# 6) UI + Navigation (Home <-> Simulator)
# ============================================================

st.set_page_config(page_title="RumorRadar", layout="wide")

# ---------------------------
# Global Custom CSS
# ---------------------------
custom_css = """
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0E0E11;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111216;
    border-right: 1px solid #2b2b2b;
}

.sidebar-title {
    color: #cfcfcf;
    font-weight: 700;
    font-size: 22px;
    margin-bottom: 10px;
}

.sidebar-option {
    color: #b3b3b3;
}

/* Header */
h1, h2, h3, h4 {
    color: #EAEAEA;
    font-weight: 700;
}

/* Normal text */
p, li, span, div {
    color: #d7d7d7 !important;
}

/* Purple accent buttons */
.stButton>button {
    background-color: #8E44AD !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
    font-size: 1rem !important;
    box-shadow: 0 0 12px rgba(142, 68, 173, 0.4);
}

.stButton>button:hover {
    background-color: #732d91 !important;
    box-shadow: 0 0 18px rgba(142, 68, 173, 0.6);
}

/* Cards */
.card {
    background-color: #15161a;
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #262626;
    margin-bottom: 15px;
    box-shadow: 0px 0px 18px rgba(0,0,0,0.25);
}

/* Text area */
textarea {
    background-color: #1a1b21 !important;
    color: #dfdfdf !important;
    border-radius: 10px !important;
    border: 1px solid #2e2e2e !important;
}

/* Floating Button */
.floating-btn {
    position: fixed;
    bottom: 35px;
    right: 35px;
    background-color: #8E44AD;
    color: white !important;
    padding: 14px 22px;
    border-radius: 14px;
    font-size: 20px;
    text-decoration: none;
    z-index: 9999;
    box-shadow: 0 4px 16px rgba(142, 68, 173, 0.5);
}
.floating-btn:hover {
    background-color: #732d91;
}

/* Radio + slider labels fix */
label {
    color: #d7d7d7 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ---- init session state ----
defaults = {
    "page": "Home",           # which view to show
    "final_label": None,
    "last_text": "",
    "running": False,
    "simulation_done": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---- Sidebar navigation ----
st.sidebar.title("Navigation")
st.session_state.page = st.sidebar.radio(
    "Go to",
    ["Home", "Simulator"],
    index=0 if st.session_state.page == "Home" else 1,
)

# ============================================================
# 6A) HOME PAGE
# ============================================================
if st.session_state.page == "Home":

    st.markdown("<h1 style='color:#EAEAEA;'>🛰 RumorRadar</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#8E44AD;'>Misinformation Detection & Spread Analysis</h3>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- What is SIR Model ---
    st.markdown("""
    <div class='card'>
    <h3 style='color:#8E44AD;'> What is the SIR Model?</h3>

    The SIR model is a scientific framework used to study how something spreads through a population.  
    Originally used for epidemics, we apply it here to misinformation.

    **It divides people into 3 states:**
    - **S — Susceptible:** not yet exposed  
    - **I — Infected:** currently spreading the misinformation  
    - **R — Recovered:** no longer spreading it  
    </div>
    """, unsafe_allow_html=True)

    # --- How It Works ---
    st.markdown("""
    <div class='card'>
    <h3 style='color:#8E44AD;'>⚙️ How Does It Work in RumorRadar?</h3>

    Our system converts a social network into a graph:
    - Users = nodes  
    - Connections = edges  
    - Spread depends on node type & probability  

    At each step:
    - Infected users try to pass misinformation to neighbors  
    - Some recover and stop spreading  
    - Spread continues until no one is infected  
    </div>
    """, unsafe_allow_html=True)

    # --- What the Simulator Does ---
    st.markdown("""
    <div class='card'>
    <h3 style='color:#8E44AD;'> What Does the Simulator Do?</h3>

    Once the news is detected as **Fake**, RumorRadar:
    - Builds a scale-free network (similar to real social media)  
    - Lets you choose how misinformation starts:  
      • **Hub** (influencers)  
      • **Bridge** (connectors)  
      • **Uniform** (random)  
    - Runs the SIR spread model  
    - Shows day-wise counts of:  
      • Safe (S)  
      • Infected (I)  
      • Recovered (R)  
    - Displays how quickly misinformation dies out or explodes  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<p style='opacity:0.8;'>Start exploring misinformation by opening the simulator.</p>", unsafe_allow_html=True)

    if st.button(" Go to Simulator"):
        st.session_state.page = "Simulator"
        st.rerun()


# ============================================================
# 6B) SIMULATOR PAGE  (YOUR ORIGINAL UI)
# ============================================================

else:
    st.markdown("<h1 style='color:#EAEAEA;'> RumorRadar — Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#bfbfbf;'>Analyze information and observe how misinformation spreads across networks.</p>", unsafe_allow_html=True)


    # we already initialized these earlier, so no setdefault now

    # ---------------- Sidebar: simulation settings (same as your code) -----------
    if not st.session_state.running:

        st.sidebar.header("Simulation Settings")

        nodes = st.sidebar.slider("Network Size", 50, 500, 100, 10)
        infection_prob = st.sidebar.slider("Infection Prob β", 0.0, 1.0, 0.6, 0.01)
        recovery_prob = st.sidebar.slider("Recovery Prob γ", 0.0, 1.0, 0.4, 0.01)
        delay = st.sidebar.slider("Delay (seconds)", 0.5, 5.0, 1.0, 0.5)

        propagation = st.sidebar.radio(
            "Spread Model",
            ["Hub (influencers)", "Bridge (connectors)", "Uniform"],
        )

        # preview network for recommended nodes
        G_prev = nx.barabasi_albert_graph(nodes, 2, seed=42)
        mode_key = (
            "Hub" if "Hub" in propagation
            else "Bridge" if "Bridge" in propagation
            else "Uniform"
        )

        select_mode = st.sidebar.radio(
            "Start Node Selection",
            ["Recommended Top 10", "All Nodes"],
        )

        if select_mode == "Recommended Top 10":
            options = get_top_nodes(G_prev, 10, mode_key)
        else:
            options = list(G_prev.nodes())

        start_node = st.sidebar.selectbox(f"Select Node ({mode_key})", options)

        # store in session
        st.session_state.update(
            nodes=nodes,
            inf=infection_prob,
            rec=recovery_prob,
            delay=delay,
            prop=propagation,
            start_node=start_node,
        )
    else:
        st.sidebar.empty()

    # ---------------- Text input + classification ----------------
st.markdown("### Enter News Text")

with st.form("text_form"):
    text = st.text_area(
        "Paste news text here...",
        st.session_state.last_text,
        height=180
    )

    classify = st.form_submit_button("Analyze")

    if classify:
        # Reset previous state
        st.session_state.running = False
        st.session_state.final_label = None
        st.session_state.simulation_done = False

        # Classify new input
        st.session_state.last_text = text
        st.session_state.final_label = classify_with_gemini(text)

        st.rerun()  # refresh UI with updated results

# --- Show result (executed AFTER rerun) ---
if st.session_state.final_label:
    indicator = {
        "Fake News": st.error,
        "Real News": st.success,
        "Uncertain": st.warning,
    }[st.session_state.final_label]
    indicator(f"Detected: **{st.session_state.final_label}**")

# ---------------- Run simulation only if Fake News ----------------
if st.session_state.final_label == "Fake News":

    st.subheader("Fake News Spread Simulation")

    if not st.session_state.running:
        if st.button("Run Simulation"):
            st.session_state.running = True
            st.session_state.simulation_done = False
            st.rerun()

    else:
        log_area = st.sidebar.empty()
        plot_area = st.empty()

        run_sir_simulation(
            st.session_state.nodes,
            st.session_state.inf,
            st.session_state.rec,
            st.session_state.start_node,
            st.session_state.prop,
            st.session_state.delay,
            log_area,
            plot_area,
        )

        st.session_state.running = False
        st.session_state.simulation_done = True
