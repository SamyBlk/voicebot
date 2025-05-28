import re
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
import spacy
from spacy.cli import download

# Forcer le modèle léger et empêcher l'installation du modèle lourd
try:
    spacy.load("en_core_web_sm")
except OSError:
    print("[INFO] SpaCy model not found, downloading 'en_core_web_sm'...")
    download("en_core_web_sm")

# Configuration unique de l'anonymiseur
anonymizer = PresidioReversibleAnonymizer(
    analyzed_fields=[
        "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS",
        "CREDIT_CARD", "IP_ADDRESS", "MEDICAL_LICENSE",
        "US_PASSPORT", "US_SSN"
    ]
)

def anonymize_text(text: str) -> str:
    """Anonymise le texte donné (reversible)."""
    return anonymizer.anonymize(text)

def deanonymize_text(text: str) -> str:
    """Restitue les entités originales du texte anonymisé."""
    return anonymizer.deanonymize(text)
