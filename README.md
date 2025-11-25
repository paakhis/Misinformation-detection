# 🛰 RumorRadar
Misinformation Detection & Spread Simulation

RumorRadar is an intelligent web application that can detect misleading or false information and simulate how misinformation spreads across a social network.
It helps users understand digital rumor propagation using a dynamic SIR epidemic model.

Live Demo: https://misinformation-detection-ml.streamlit.app/

# Features
Feature	Description
Fake News Detection	Classifies news as Fake, Real, or Uncertain
Spread Simulation	Mimics rumor spreading over social media-like networks
Visual Graph Output	Shows Safe (S), Infected (I), and Recovered (R) nodes
Adjustable Controls	Infection/Recovery probability and propagation strategy
Smart Node Selection	Option to spread via influencers or connectors
Simple & Interactive UI	Built using Streamlit for a smooth user experience

# How It Works: 

Step 1: News Analysis

Users enter any news text, and the system classifies it into:
Fake News
Real News
Uncertain

Step 2: Misinformation Spread

If detected as Fake News:
A scale-free network similar to social platforms is generated
Rumor spreads based on user parameters
Simulation continues until spread ends

# Tech Stack
Component	Technology
UI Framework	Streamlit
ML Model	Ensemble Model (Random Forest + Others)
Text Features	TF-IDF Vectorizer
Network Model	Barabási–Albert Graph
Visualization	Matplotlib
File Handling	gdown
