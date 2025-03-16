import json
import os
import pandas as pd
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas import evaluate
from dotenv import load_dotenv
import sys

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Définir la clé API OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Vérifier si le fichier existe
json_file = "ragas_dataset_eval.json"
if not os.path.exists(json_file):
    print(f"Erreur: Le fichier '{json_file}' n'existe pas.")
    sys.exit(1)

# Charger les données JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    print(f"Fichier chargé: {len(data)} entrées")

# Convertir les données au format attendu par Ragas
ragas_data = []
for item in data:
    ragas_item = {
        "question": item["question"],
        "ground_truths": [item["ground_truth"]],
        "contexts": [item["retrieved_passage"]],
        "answer": item["answer"],
        "reference": item["ground_truth"]
    }
    ragas_data.append(ragas_item)

# Créer un dataset Hugging Face
dataset = Dataset.from_list(ragas_data)

# Configurer les métriques
metrics = [
    faithfulness,
    answer_relevancy,
    context_recall
]

# Évaluer le dataset
print("\nÉvaluation en cours...")
results = evaluate(dataset, metrics)

# Afficher les résultats
print("\nRésultats de l'évaluation:")
print(results)

# Sauvegarder les résultats

if isinstance(results, dict):
    results_df = pd.DataFrame([results])
elif isinstance(results, pd.DataFrame):
    results_df = results
else:
    try:
        results_dict = results.__dict__
        results_df = pd.DataFrame([results_dict])
    except:
        metric_names = [type(m).__name__ for m in metrics]
        values = [getattr(results, type(m).__name__.lower(), None) for m in metrics]
        results_df = pd.DataFrame({metric: [value] for metric, value in zip(metric_names, values) if value is not None})

# Sauvegarder le DataFrame
results_df.to_csv("ragas_evaluation_results.csv", index=False)
print("Résultats sauvegardés dans 'ragas_evaluation_results.csv'")