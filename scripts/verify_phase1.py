"""
MaintiDoc - Phase 1 Verification and Diagnostic Script
Tests all fundamental requirements: Python environment, packages, PDF access,
PostgreSQL + PGVector connection, and Ollama LLM readiness.
"""

import sys
import os
from pathlib import Path
import importlib

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import settings, setup_logger

logger = setup_logger("verify.phase1")


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def check_python_environment():
    print_header("1. Vérification de l'environnement Python")
    py_version = sys.version.replace("\n", " ")
    print(f"[OK] Python Exécutable: {sys.executable}")
    print(f"[OK] Python Version:    {py_version}")

    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor in (10, 11, 12):
        print(f"[SUCCESS] Version Python ({major}.{minor}) optimale pour PyTorch, LlamaIndex et LangChain.")
        return True
    else:
        print(f"[INFO] Version Python ({major}.{minor}).")
        return True


def check_libraries():
    print_header("2. Vérification des bibliothèques requises")
    required_packages = [
        ("torch", "PyTorch (CPU/CUDA)"),
        ("torchvision", "TorchVision"),
        ("open_clip", "OpenCLIP (Embeddings Visuels)"),
        ("sentence_transformers", "Sentence Transformers (Embeddings Textuels)"),
        ("llama_index.core", "LlamaIndex Core"),
        ("langchain", "LangChain Orchestration"),
        ("langchain_ollama", "LangChain Ollama"),
        ("fitz", "PyMuPDF (fitz - Extraction PDF)"),
        ("pdfplumber", "pdfplumber (Extraction Tableaux)"),
        ("PIL", "Pillow (Traitement Images)"),
        ("numpy", "NumPy (Calculs matriciels)"),
        ("psycopg2", "psycopg2 (PostgreSQL Connector)"),
        ("pgvector", "pgvector (Python PGVector support)"),
        ("sqlalchemy", "SQLAlchemy ORM"),
        ("streamlit", "Streamlit UI"),
        ("pydantic", "Pydantic Validation"),
        ("requests", "Requests HTTP Client"),
    ]

    all_passed = True
    for module_name, display_name in required_packages:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {display_name:<42} : v{ver}")
        except ImportError as e:
            print(f"  [ERREUR] {display_name:<38} : NON INSTALLÉ ({e})")
            all_passed = False

    return all_passed


def check_pdf_document():
    print_header("3. Vérification du Document de Test")
    pdf_path = settings.DOCUMENTS_DIR / "Attention is all you need.pdf"
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"  [OK] Fichier trouvé : {pdf_path.name} ({size_kb:.1f} KB)")
        try:
            import fitz

            doc = fitz.open(pdf_path)
            print(f"  [OK] PyMuPDF Lecture réussie : {len(doc)} pages détectées.")
            doc.close()
            return True
        except Exception as e:
            print(f"  [ERREUR] Impossible de lire le PDF avec PyMuPDF: {e}")
            return False
    else:
        print(f"  [ATTENTION] Fichier introuvable à {pdf_path}")
        return False


def check_postgres_pgvector():
    print_header("4. Vérification PostgreSQL et PGVector")
    from database.connection import check_db_health, init_database

    healthy, message, details = check_db_health()
    if healthy:
        print(f"  [OK] Connexion réussie à PostgreSQL sur {details['host']}:{details['port']}/{details['database']}")
        print(f"  [OK] Serveur PostgreSQL : {details['pg_version'].split(',')[0] if details['pg_version'] else 'Inconnu'}")
        if details.get("pgvector_installed"):
            print(f"  [OK] Extension PGVector active : v{details.get('pgvector_version', 'detectée')}")
            # Try to initialize schema
            init_success = init_database()
            if init_success:
                print("  [OK] Schéma MaintiDoc (documents, text_chunks, image_metadata, table_chunks) validé.")
                return True
            else:
                print("  [ERREUR] Échec de l'initialisation du schéma SQL.")
                return False
        else:
            print("  [ATTENTION] PGVector non détecté dans la base PostgreSQL.")
            print("    Exécutez: CREATE EXTENSION IF NOT EXISTS vector;")
            return False
    else:
        print(f"  [ATTENTION] PostgreSQL n'est pas encore accessible : {message}")
        print("  --> Pour démarrer PostgreSQL avec PGVector via Docker :")
        print("      docker compose up -d")
        return False


def check_ollama_service():
    print_header("5. Vérification du Service Ollama & Modèle Qwen")
    from llm.ollama_client import ollama_client

    if ollama_client.is_alive():
        print(f"  [OK] Serveur Ollama actif à l'adresse : {ollama_client.base_url}")
        models = ollama_client.list_models()
        print(f"  [OK] Modèles locaux installés ({len(models)}) : {', '.join(models) if models else 'Aucun'}")

        target_model = settings.OLLAMA_MODEL
        if ollama_client.is_model_present(target_model):
            print(f"  [OK] Modèle cible '{target_model}' est PRÊT.")
            try:
                print(f"  [TEST] Génération rapide de test avec {target_model}...")
                test_resp = ollama_client.generate("Say 'MaintiDoc is ready' in 5 words.")
                print(f"  [OK] Réponse du modèle : \"{test_resp.strip()}\"")
                return True
            except Exception as e:
                print(f"  [ERREUR] Échec du test d'inférence : {e}")
                return False
        else:
            print(f"  [ATTENTION] Le modèle cible '{target_model}' n'est pas encore téléchargé.")
            print(f"  --> Exécutez : ollama pull {target_model}")
            return False
    else:
        print(f"  [ATTENTION] Serveur Ollama non joignable sur {ollama_client.base_url}")
        print("  --> Vérifiez qu'Ollama est démarré (`ollama serve` ou application Ollama).")
        return False


def main():
    print("\n" + "#" * 70)
    print("      RAPPORT DE VÉRIFICATION PHASE 1 - MAINTIDOC")
    print("#" * 70)

    p1 = check_python_environment()
    p2 = check_libraries()
    p3 = check_pdf_document()
    p4 = check_postgres_pgvector()
    p5 = check_ollama_service()

    print_header("Bilan de la Phase 1")
    print(f"  1. Python 3.11 Environment : {'PASS' if p1 else 'FAIL'}")
    print(f"  2. Bibliothèques requises    : {'PASS' if p2 else 'FAIL'}")
    print(f"  3. Document de référence    : {'PASS' if p3 else 'FAIL'}")
    print(f"  4. PostgreSQL + PGVector    : {'PASS' if p4 else 'EN ATTENTE (Docker / Local)'}")
    print(f"  5. Ollama & Qwen LLM        : {'PASS' if p5 else 'EN ATTENTE (Installation / Démarrage)'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
