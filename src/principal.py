from dotenv import load_dotenv
import os
import fitz
import spacy
from sentence_transformers import util
from dataclasses import dataclass
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
import yaml
import warnings
warnings.filterwarnings("ignore")

# Charger les variables d'environnement
load_dotenv()


@dataclass
class Document:
    page_content: str
    metadata: dict


class RAGUEUR:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "..", "config.yaml")

        # Charger la configuration depuis le fichier YAML
        self.config = self._load_config(config_path)

        # Initialiser les variables à partir de la configuration
        self.local_data_dir = self.config.get(
            "data_dir", "data/droit_Marocain")
        os.makedirs(self.local_data_dir, exist_ok=True)

        # Initialiser le modèle NLP
        self.nlp = spacy.load(self.config.get("nlp_model", "en_core_web_trf"))

        # Initialiser le modèle d'embeddings
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.config.get("embeddings_model", "all-MiniLM-L6-v2")
        )

        # Initialiser le client LLM
        self.client = InferenceClient(
            self.config.get("llm_model", "meta-llama/Llama-3.3-70B-Instruct"),
            api_key=os.getenv("hf_api_key")
        )

        self.vector_store = None

    def _load_config(self, config_path):
        """Charger la configuration depuis un fichier YAML"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Résoudre les variables d'environnement dans le fichier de configuration
            config = self._resolve_env_vars(config)
            return config
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {str(e)}")
            # Configuration par défaut en cas d'échec
            return {
                "data_dir": "data/droit_Marocain",
                "similarity_threshold": 0.75,
                "embeddings_model": "all-MiniLM-L6-v2",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.3",
                "nlp_model": "en_core_web_trf",
                "temperature": 0.5,
                "repetition_penalty": 1.1,
                "max_tokens": 500,
                "do_sample": True
            }

    def _resolve_env_vars(self, config):
        """Résoudre les variables d'environnement dans la configuration"""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(i) for i in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, "")
        else:
            return config

    def extract_text_from_pdf(self, pdf_path):
        chunks = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                metadata = {
                    "source": pdf_path,
                    "file_name": os.path.basename(pdf_path),
                    "page_number": page_num + 1  # Les numéros de page commencent à 1
                }
                chunks.append({"text": text, "metadata": metadata})
        return chunks

    def semantic_chunking(self, text, metadata):
        similarity_threshold = self.config.get("similarity_threshold", 0.75)
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        embeddings = self.embeddings_model.embed_documents(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        for i in range(1, len(sentences)):
            similarity = util.cos_sim(embeddings[i], embeddings[i-1])[0][0]
            if similarity >= similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(Document(
                    page_content=" ".join(current_chunk),
                    metadata=metadata
                ))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(Document(
                page_content=" ".join(current_chunk),
                metadata=metadata
            ))

        return chunks

    def process_pdf(self, pdf_path):
        if not pdf_path:
            return None

        try:
            text_chunks = self.extract_text_from_pdf(pdf_path)
            all_chunks = []

            for chunk in text_chunks:
                text = chunk["text"]
                metadata = chunk["metadata"]
                chunks = self.semantic_chunking(text, metadata)
                all_chunks.extend(chunks)

            return all_chunks
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {pdf_path}: {str(e)}")
            return None

    def get_vector_store(self, chunks):
        """ Crée un VectorStore FAISS à partir des chunks. """
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        return FAISS.from_texts(texts=texts, embedding=self.embeddings_model, metadatas=metadatas)

    def update_vector_store(self):
        all_chunks = []
        for file_name in os.listdir(self.local_data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.local_data_dir, file_name)
                chunks = self.process_pdf(pdf_path)
                if chunks:
                    all_chunks.extend(chunks)

        if all_chunks:
            self.vector_store = self.get_vector_store(all_chunks)
            print(f"Vector store mis à jour avec {len(all_chunks)} chunks.")
