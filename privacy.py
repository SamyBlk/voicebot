from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from faker import Faker

class ReversiblePIIAnonymizer:
    def __init__(self):
        self.analyzer = AnalyzerEngine(
            nlp_engine_name="spacy",
            models={"en": "en_core_web_sm"}
        )
        self.anonymizer = AnonymizerEngine()
        self.faker = Faker()
        self.reverse_map = {}

    def anonymize(self, text):
        results = self.analyzer.analyze(text=text, language='en')
        anonymized_text = text
        for res in sorted(results, key=lambda x: x.start, reverse=True):
            entity = text[res.start:res.end]
            fake_val = self._fake_value(res.entity_type)
            self.reverse_map[fake_val] = entity
            anonymized_text = anonymized_text[:res.start] + fake_val + anonymized_text[res.end:]
        print(f"[ANONYMIZED] {anonymized_text}")  # Affiche le texte anonymisé
        return anonymized_text

    def deanonymize(self, text):
        for fake_val, real_val in self.reverse_map.items():
            text = text.replace(fake_val, real_val)
        return text

    def _fake_value(self, entity_type):
        if entity_type == "PERSON":
            return self.faker.name()
        elif entity_type == "PHONE_NUMBER":
            return self.faker.phone_number()
        elif entity_type == "EMAIL_ADDRESS":
            return self.faker.email()
        elif entity_type == "CREDIT_CARD":
            return self.faker.credit_card_number()
        elif entity_type == "IP_ADDRESS":
            return self.faker.ipv4()
        elif entity_type == "US_SSN":
            return self.faker.ssn()
        elif entity_type == "US_PASSPORT":
            return "P" + self.faker.bothify(text='#########')
        elif entity_type == "MEDICAL_LICENSE":
            return self.faker.bothify(text='??######')
        else:
            return self.faker.word()