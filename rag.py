import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from functools import lru_cache
import pandas as pd
import re

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def preprocess_csv(input_path, output_path):
    """
    Prétraite le CSV : nettoie, supprime les doublons, normalise les textes.
    """
    df = pd.read_csv(input_path)
    # Nettoyage basique : strip, suppression espaces multiples, normalisation
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace('\u200b', '')  # caractères invisibles éventuels
        return text

    df['Question'] = df['Question'].apply(clean_text)
    df['Answer'] = df['Answer'].apply(clean_text)
    # Suppression des doublons
    df = df.drop_duplicates(subset=['Question', 'Answer'])
    # Optionnel : suppression des lignes vides
    df = df[(df['Question'] != "") & (df['Answer'] != "")]
    df.to_csv(output_path, index=False)

def initialize_rag():
    """
    Initialise le système RAG en utilisant Weaviate Cloud (client v4).
    """
    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Prétraitement du CSV
    file_path = './data/QandA.csv'
    preprocessed_path = './data/QandA_preprocessed.csv'
    preprocess_csv(file_path, preprocessed_path)

    # Charger les données CSV prétraitées
    loader = CSVLoader(file_path=preprocessed_path)
    docs = loader.load_and_split()

    # Connexion à Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    # Créer la collection "Document2" si elle n'existe pas
    if "Document" not in client.collections.list_all():
        client.collections.create(
            name="Document",
            properties=[Property(name="text", data_type=DataType.TEXT)],
            vectorizer_config=Configure.Vectorizer.none()
        )

    # Créer le vector store avec text_key obligatoire
    vector_store = WeaviateVectorStore(
        client=client,
        index_name="Document",
        embedding=embeddings,
        text_key="text"
    )

    # Ajouter les documents si nécessaire
    if vector_store.similarity_search("test") == []:
        vector_store.add_documents(documents=docs)

    # Créer le retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Définir le prompt
    system_prompt = (
        "You are an assistant specialized in answering questions about Apple products. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Construire la chaîne RAG
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain, client

# Initialisation
rag_chain, weaviate_client = initialize_rag()

@lru_cache(maxsize=128)
def query_rag(question):
    """
    Interroge le système RAG pour une question.
    """
    answer = rag_chain.invoke({"input": question}, config={"callbacks": []})
    return answer["answer"]

