import re
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

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
