"""
Application Streamlit — Prédiction de l'état d'un Vehicule
Pour savoir s´il s´agit d´un véhicule neuf ou D´occasion
Lancement en local :  streamlit run app.py
"""
!pip install joblib

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime 
import os



# Configuration de la page
st.set_page_config (
    page_title="Prédiction de l'état d'un Véhicule",
    page_icon="🚗",
    layout="wide",
    menu_items= {
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': "mailto:augusthiaw@gmail.com",
        'About': "https://streamlit.io/community"}
)

## Custom CSS pour le style de l'application
st.markdown("""
<style>

/* Main background */
.stApp {
    background:  linear-gradient(0deg,rgba(186, 182, 182, 1) 1%, rgba(252, 250, 250, 1) 100%);
    );
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(0deg,rgba(70, 70, 140, 1) 14%, rgba(255, 255, 255, 1) 100%);
}

/* Cards */
.kpi-card{
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    text-align:center;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.2);
    box-shadow:0 8px 20px rgba(0,0,0,0.3);
}

.pred-card{
    background: linear-gradient(
        135deg,
        #10b981,
        #14b8a6
    );
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
}

h1,h2,h3{
    color:#0B0B75 !important;
}



</style>
""", unsafe_allow_html=True)


DESCRIPTION = """
La prédiction simple permet de prédire l'état d'un seul véhicule
(Neuf ou Occasion) à partir des variables :

- Marque
- Année
- Transmission
- Prix
- Quartier
"""

DESCRIPTION2 = """
La prediction multiple permet de prédire l'état de plusieurs véhicules à partir d'un fichier CSV """

# Historique des prédictions
HISTORY_FILE = "prediction_history.csv"
    
# Chargement des artefacts (mis en cache : chargés une seule fois)



# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")   # encodeurs (Marque, Tramsission, Adresse)
    uniques = jb.load("uniques.joblib")     # valeurs uniques
    scaler = jb.load("scaler.joblib")       # normaliseur
    gb = jb.load("gb_model.joblib")         # modèle
    return encoders, uniques, scaler, gb


encoders, uniques, scaler, gb = load_artifacts()
clasnames = uniques[3]  # noms des classes



# Fonction de prédiction simple

def Pred_func(Marque, Annee, Transmission, Prix, Quartier):
    # Encoder l'adresse et la marque
    Marque = encoders[0].transform([Marque])[0]
    Transmission = encoders[1].transform([Transmission])[0]
    Quartier = encoders[2].transform([Quartier])[0]
    
    # Vecteur des valeurs numériques
    x_new = np.array([Marque, Annee, Transmission, Prix, Quartier])
    x_new = x_new.reshape(1, -1)  # conversion en un tableau 2D
    # Normaliser les données
    x_new = scaler.transform(x_new)
    # Prédire
    y_pred = gb.predict(x_new)
    return clasnames[y_pred[0]]



# Fonction de prédiction multiple

def Pred_func_csv(file):
    # Lire le fichier csv
    df = pd.read_csv(file)
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :].values:
        # prédiction simple
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4])
        predictions.append(y_pred)
    df["Etat"] = predictions
    return df

# Fonction pour sauvegarder l'historique des prédictions
def save_history(data):
    if not os.path.exists(HISTORY_FILE):
        data.to_csv(HISTORY_FILE,index=False)

    else:
        data.to_csv(
            HISTORY_FILE,
            mode="a",
            index=False,
            header=False
        )

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/744/744465.png",
        width=120
    )

    st.title("🚗 Véhicule")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 Single Prediction",
            "📂 Batch Prediction",
            "📜 Historical Data",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.info(
        """
        Vehicle Condition Prediction System

        ✅ Single Prediction
        
        ✅ Batch Prediction
        
        ✅ Analytics Dashboard
        
        ✅ Historical Data
        """
    )

    st.markdown("---")

    st.write(
        f"Nombre de Classes: {len(clasnames)}"
    )



#st.title("🛻 Prédiction de l'état d´un Véhicule")

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
padding:30px;
border-radius:20px;
text-align:center;
background:linear-gradient(0deg,rgba(70, 70, 140, 1) 14%, rgba(255, 255, 255, 1) 100%)
">

<h1>
🚗 Vehicle Condition Prediction Platform
</h1>

</div>
""", unsafe_allow_html=True)

st.write("")


onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs(["🚗 Single Prediction", "📂 Batch Prediction", " 🏠 Dashboard", "📊 Historical Data", "ℹ️ About"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.header("Single Vehicle Prediction")

    st.info(DESCRIPTION)

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            Marque = st.selectbox(
                "Marque",
                list(uniques[0])
            )

            Annee = st.number_input(
                "Année",
                min_value=1980,
                max_value=2030,
                value=2020
            )

            Transmission = st.selectbox(
                "Transmission",
                list(uniques[1])
            )

        with col2:

            Prix = st.number_input(
                "Prix",
                min_value=0.0,
                value=1000000.00
            )

            Quartier = st.selectbox(
                "Quartier",
                list(uniques[2])
            )

        submit = st.form_submit_button(
            "Predict",
            type="primary"
        )

    if submit:

        resultat = Pred_func(
            Marque,
            Annee,
            Transmission,
            Prix,
            Quartier
        )

        st.markdown(
            f"""
            <div class="pred-card">
                <h2>Vehicle Status</h2>
                <h1>{resultat}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        history_row = pd.DataFrame([{
            "Marque": Marque,
            "Annee": Annee,
            "Transmission": Transmission,
            "Prix": Prix,
            "Quartier": Quartier,
            "Etat": resultat,
            "Date": datetime.now(),
        }])

        save_history(history_row)

        chart_df = pd.DataFrame({
            "Variable": ["Année", "Prix"],
            "Valeur": [Annee, Prix]
        })

        fig = px.bar(
            chart_df,
            x="Variable",
            y="Valeur",
            color="Variable"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Predict vehicle status with multiple inputs")
    st.write(DESCRIPTION2)
    file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if file:

        with st.spinner(
            "Running predictions..."
        ):

            results = Pred_func_csv(file)

        st.success(
            f"{len(results)} Predictions Completed"
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        counts = (
            results["Etat"]
            .value_counts()
            .reset_index()
        )

        fig = px.pie(
            counts,
            names="Etat",
            values="count",
            title="Prediction Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        results["Date"] = datetime.now()

        save_history(results)

        st.download_button(
            "⬇ Download Results",
            results.to_csv(index=False).encode(),
            file_name="predictions.csv",
            mime="text/csv"
        )
# ----------------------------- Onglet 3 -------------------------------
with onglet3:
    st.header("Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Brands",
        len(uniques[0])
    )

    c2.metric(
        "Transmission Types",
        len(uniques[1])
    )

    c3.metric(
        "Quartiers",
        len(uniques[2])
    )

    c4.metric(
        "Classes",
        len(clasnames)
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        brand_df = pd.DataFrame({
            "Brand": list(uniques[0])
        })

        fig1 = px.histogram(
            brand_df,
            x="Brand",
            title="Available Brands"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        quartiers_df = pd.DataFrame({
            "Quartier": list(uniques[2])
        })

        fig2 = px.histogram(
            quartiers_df,
            x="Quartier",
            title="Available Quartiers"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        st.subheader(
            "Prediction Distribution"
        )

        if "Etat" in history.columns:

            counts = (
                history["Etat"]
                .value_counts()
                .reset_index()
            )

            fig3 = px.pie(
                counts,
                names="Etat",
                values="count",
                hole=0.4
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
        )
     
# ------------------------------Onglet 4-------------------------------
with onglet4:
    st.header("Historical Predictions")

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        search = st.text_input(
            "Search Brand"
        )

        if search:

            history = history[
                history["Marque"]
                .astype(str)
                .str.contains(
                    search,
                    case=False
                )
            ]

        st.dataframe(
            history,
            use_container_width=True
        )

        st.metric(
            "Total Predictions",
            len(history)
        )

        if "Etat" in history.columns:

            pred_counts = (
                history["Etat"]
                .value_counts()
                .reset_index()
            )

            fig = px.bar(
                pred_counts,
                x="Etat",
                y="count",
                color="Etat"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.download_button(
            "Download History",
            history.to_csv(index=False).encode(),
            "history.csv",
            "text/csv"
        )

    else:

        st.warning(
            "No historical data available."
        )  
        
# ----------------------------- Onglet 5 -------------------------------
with onglet5:
    st.header("About")

    st.markdown("""
    ### Developer
    **Augustin Thiaw**

    Senior ICT Associate

    ### AI Model
    Gradient Boosting Classifier

    ### Features

    ✅ Dashboard

    ✅ Single Prediction

    ✅ Batch Prediction

    ✅ Historical Predictions

    ✅ Analytics

    ✅ Download Results

    ### Contact

    📧 augusthiaw@gmail.com

    🔗 https://github.com/athiaw
    """)
    
# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Prediction App Developped By Augustin Thiaw, Powered by Gradient Boosting Model")
# add my Streamlit profile link and photo in the footer
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://avatars.githubusercontent.com/u/206091052?v=4" alt="Augustin Thiaw" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;">
        <a href="https://github.com/athiaw" target="_blank" style="color: #007bff; text-decoration: none;">Augustin Thiaw</a>
    </div>
    """,
    unsafe_allow_html=True,
)

