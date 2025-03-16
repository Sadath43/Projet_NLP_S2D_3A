import streamlit as st
import os
import pandas as pd
import time
import yaml
from src.principal import RAGUEUR
import plotly.express as px

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RAGUEUR - Système RAG",
    page_icon="📚",
    layout="wide"
)

# Charger la configuration
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return None


config = load_config(config_path="config.yaml")

    
# Initialisation de l'état de session si nécessaire
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGUEUR()

if 'system_status' not in st.session_state:
    st.session_state.system_status = {
        "vectorstore_initialized": False,
        "documents_processed": 0,
        "chunks_created": 0,
        "last_update": None
    }

def update_vectorstore():
    """Mettre à jour le vectorstore et l'état du système"""
    with st.spinner("Mise à jour du vectorstore en cours..."):
        start_time = time.time()
        
        # Compter les fichiers avant la mise à jour
        doc_count = 0
        for file_name in os.listdir(st.session_state.rag_system.local_data_dir):
            if file_name.endswith(".pdf"):
                doc_count += 1
        
        # Mise à jour du vectorstore
        st.session_state.rag_system.update_vector_store()
        
        # Mise à jour de l'état du système
        st.session_state.system_status["vectorstore_initialized"] = (st.session_state.rag_system.vector_store is not None)
        st.session_state.system_status["documents_processed"] = doc_count
        
        # Estimation du nombre de chunks (si vectorstore existe)
        if st.session_state.rag_system.vector_store:
            try:
                # Cette méthode pourrait varier selon l'implémentation exacte de FAISS
                st.session_state.system_status["chunks_created"] = len(st.session_state.rag_system.vector_store.index_to_docstore_id)
            except:
                # Fallback si la méthode ci-dessus ne fonctionne pas
                st.session_state.system_status["chunks_created"] = "Non disponible"
        
        st.session_state.system_status["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        elapsed_time = time.time() - start_time
        st.success(f"Vectorstore mis à jour en {elapsed_time:.2f} secondes!")

def generate_response(question):
    """Générer une réponse à partir d'une question"""
    if not st.session_state.system_status["vectorstore_initialized"]:
        st.error("Veuillez d'abord initialiser le vectorstore!")
        return None
    
    with st.spinner("Génération de la réponse en cours..."):
        try:
            result = st.session_state.rag_system.generate_answer(question)
            return result
        except Exception as e:
            st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
            return None

def upload_pdf():
    """Gérer l'upload de fichiers PDF"""
    uploaded_file = st.file_uploader("Télécharger un fichier PDF", type="pdf")
    if uploaded_file:
        try:
            # Créer le chemin de destination
            destination = os.path.join(st.session_state.rag_system.local_data_dir, uploaded_file.name)
            
            # Sauvegarder le fichier
            with open(destination, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Fichier '{uploaded_file.name}' téléchargé avec succès!")
            return True
        except Exception as e:
            st.error(f"Erreur lors du téléchargement du fichier: {str(e)}")
            return False
    return False

# Interface principale avec onglets
tab1, tab2, tab3, tab4 = st.tabs(["État du système", "Gestion des documents", "Génération de réponses", "Configuration"])

# Onglet 1: État du système
with tab1:
    st.header("État du système RAG")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations générales")
        status_df = pd.DataFrame({
            "Métrique": [
                "Vectorstore initialisé",
                "Documents traités",
                "Chunks créés",
                "Dernière mise à jour"
            ],
            "Valeur": [
                "✅ Oui" if st.session_state.system_status["vectorstore_initialized"] else "❌ Non",
                st.session_state.system_status["documents_processed"],
                st.session_state.system_status["chunks_created"],
                st.session_state.system_status["last_update"] or "Jamais"
            ]
        })
        st.table(status_df)
        
        if st.button("Rafraîchir l'état du système", key="refresh_status"):
            update_vectorstore()
    
    with col2:
        st.subheader("Visualisation")
        if st.session_state.system_status["vectorstore_initialized"]:
            # Graphique simple pour visualiser les données
            if isinstance(st.session_state.system_status["chunks_created"], int):
                fig = px.pie(
                    names=["Documents", "Chunks"],
                    values=[
                        st.session_state.system_status["documents_processed"],
                        st.session_state.system_status["chunks_created"]
                    ],
                    title="Ratio Documents/Chunks"
                )
                st.plotly_chart(fig)
            else:
                st.info("Visualisation non disponible - Informations sur les chunks manquantes")
        else:
            st.info("Aucune donnée à visualiser. Veuillez initialiser le vectorstore.")

# Onglet 2: Gestion des documents
with tab2:
    st.header("Gestion des documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Télécharger des documents")
        if upload_pdf():
            st.info("Veuillez mettre à jour le vectorstore pour intégrer le nouveau document.")
    
    with col2:
        st.subheader("Documents existants")
        
        # Liste des documents dans le répertoire
        files = [f for f in os.listdir(st.session_state.rag_system.local_data_dir) if f.endswith(".pdf")]
        
        if files:
            file_df = pd.DataFrame({
                "Nom du fichier": files,
                "Taille (ko)": [
                    round(os.path.getsize(os.path.join(st.session_state.rag_system.local_data_dir, f)) / 1024, 2)
                    for f in files
                ]
            })
            st.dataframe(file_df)
        else:
            st.info("Aucun document PDF trouvé dans le répertoire de données.")
    
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Mettre à jour le vectorstore", key="update_vectorstore"):
            update_vectorstore()

# Onglet 3: Génération de réponses
with tab3:
    st.header("Génération de réponses")
    
    # Vérifier si le vectorstore est initialisé
    if not st.session_state.system_status["vectorstore_initialized"]:
        st.warning("Vectorstore non initialisé. Veuillez d'abord mettre à jour le vectorstore.")
        if st.button("Initialiser le vectorstore", key="init_vectorstore"):
            update_vectorstore()
    else:
        # Interface de questionnement
        question = st.text_area("Posez votre question", height=100)
        
        if st.button("Générer une réponse"):
            if question:
                result = generate_response(question)
                if result:
                    st.subheader("Réponse:")
                    st.write(result["response"])
                    
                    # Afficher les sources
                    st.subheader("Sources:")
                    for source in result["sources"]:
                        st.write(f"- Document: {source['file_name']}, Page: {source['page_number']}")
            else:
                st.error("Veuillez saisir une question.")
        

# Onglet 4: Configuration
with tab4:
    st.header("Configuration du système")
    
    if config:
        # Afficher les paramètres de configuration actuels sous forme de tableau
        config_data = []
        for key, value in config.items():
            if key != "hf_api_key":  # Ne pas afficher la clé API
                config_data.append({"Paramètre": key, "Valeur": str(value)})
        
        st.table(pd.DataFrame(config_data))
        
        # Option pour éditer manuellement la configuration
        st.subheader("Modifier la configuration")
        st.warning("Attention : La modification de la configuration nécessite un redémarrage du système pour prendre effet.")
        
        # Formulaire simplifié pour modifier les paramètres clés
        with st.form("config_form"):
            data_dir = st.text_input("Répertoire de données", value=config.get("data_dir", "data/droit_Marocain"))
            llm_model = st.text_input("Modèle LLM", value=config.get("llm_model", "mistralai/Mistral-7B-Instruct-v0.3"))
            similarity = st.slider("Seuil de similarité", min_value=0.0, max_value=1.0, value=float(config.get("similarity_threshold", 0.75)), step=0.05)
            temp = st.slider("Température", min_value=0.0, max_value=1.0, value=float(config.get("temperature", 0.3)), step=0.05)
            
            # Soumettre les modifications
            submit = st.form_submit_button("Enregistrer les modifications")
            
            if submit:
                # Mettre à jour la configuration
                config["data_dir"] = data_dir
                config["llm_model"] = llm_model
                config["similarity_threshold"] = similarity
                config["temperature"] = temp
                
                # Sauvegarder dans le fichier YAML
                try:
                    with open("config.yaml", "w") as file:
                        yaml.dump(config, file)
                    st.success("Configuration mise à jour avec succès ! Redémarrez l'application pour appliquer les changements.")
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
    else:
        st.error("Impossible de charger la configuration.")

# Pied de page
st.markdown("---")
st.caption("RAGUEUR - Système RAG sur PDF")