import streamlit as st
import joblib
import pandas as pd

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="IA Demande Predictor",
    page_icon="📦",
    layout="wide"
)

# --- 2. DESIGN NOIR ET VERT ÉMERAUDE (CSS) ---
st.markdown("""
<style>
    /* Fond noir profond */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stSidebar { background-color: #161920 !important; }
    
    /* Chiffre de résultat en vert */
    div[data-testid="stMetricValue"] { color: #00FF87; font-size: 45px; font-weight: bold; }
    
    /* FORMULAIRE : Bordure grise discrète */
    div[data-testid="stForm"] { border: 1px solid #343A40; border-radius: 15px; background-color: #161920; }
    
    /* MODIFICATION : Le Bouton Vert Émeraude */
    .stButton>button { 
        background-color: #00FF87; /* Couleur de fond Vert Émeraude */
        color: #0E1117;            /* Texte en noir pour le contraste */
        border: none;              /* Pas de bordure */
        border-radius: 8px; width: 100%; height: 50px;
        font-weight: bold; font-size: 18px;
    }
    
    /* Survol du bouton : Effet d'éclaircissement */
    .stButton>button:hover { 
        background-color: #33FFA3; /* Vert encore plus clair au survol */
        color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DES FICHIERS ---
try:
    # On ajoute 'models/' devant le nom de chaque fichier
    model_gbr = joblib.load('models/model_gbr.pkl')
    model_rf = joblib.load('models/model_rf.pkl') 
    le_prod = joblib.load('models/encoder_product.pkl')
    le_whse = joblib.load('models/encoder_warehouse.pkl')
    le_cat = joblib.load('models/encoder_category.pkl')
    st.sidebar.success("✅ Modèles chargés depuis /models")
except Exception as e:
    st.sidebar.error("⚠️ Fichiers .pkl introuvables dans le dossier 'models'")
    st.stop()
# --- 4. BARRE LATÉRALE (CONFIGURATION) ---
with st.sidebar:
    st.title("⚙️ Configuration")
    model_choice = st.radio(
        "Choisir le modèle d'IA",
        ('Gradient Boosting', 'Random Forest')
    )
    st.divider()
    st.subheader("📅 Période de Prédiction")
    # Choix des années de 2024 à 2028
    annee = st.selectbox("Choisir l'année", [2024, 2025, 2026, 2027, 2028,2029,2030,2031,2032,2033,2034,2035,2036])
    mois = st.selectbox("Choisir le mois", 
        ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"])

# --- 5. INTERFACE PRINCIPALE ---
st.title(f"📊 Prédiction de la Demande Produit")
st.markdown(f"Analyse prévisionnelle pour l'année **{annee}**")

with st.form("main_form"):
    st.subheader("📝 Paramètres de recherche")
    col1, col2 = st.columns(2)
    
    with col1:
        prod = st.selectbox("Code Produit", options=le_prod.classes_)
        whse = st.selectbox("Entrepôt (Warehouse)", options=le_whse.classes_)
        
    with col2:
        cat = st.selectbox("Catégorie de Produit", options=le_cat.classes_)
        btn = st.form_submit_button("Lancer la Prédiction")

# --- 6. CALCUL ET RÉSULTAT ---
if btn:
    # Encodage
    p_idx = le_prod.transform([prod])[0]
    w_idx = le_whse.transform([whse])[0]
    c_idx = le_cat.transform([cat])[0]
    
    input_data = pd.DataFrame([[p_idx, w_idx, c_idx]], 
                              columns=['Product_Code', 'Warehouse', 'Product_Category'])
    
    # Choix du modèle selon ton clic
    if model_choice == 'Gradient Boosting':
        res = model_gbr.predict(input_data)[0]
    else:
        res = model_rf.predict(input_data)[0]
        
    # Affichage propre
    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f"✅ Analyse terminée avec succès pour **{prod}**")
    
    # Bloc de résultat stylisé
    with st.container(border=True):
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric("Demande Estimée", f"{int(res)} unités")
        with col_res2:
            st.write(f"**Détails de la prévision :**")
            st.write(f"- Période : {mois} {annee}")
            st.write(f"- Modèle utilisé : {model_choice}")