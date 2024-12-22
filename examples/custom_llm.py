"""Example usage of GraphRAG with custom LLM and Embedding services compatible with the OpenAI API."""

from typing import List

import instructor
import os

from dotenv import load_dotenv

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

load_dotenv()

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

import google.auth
import google.auth.transport.requests

DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]

ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

creds, project = google.auth.default()
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

PROJECT_ID = os.getenv('PROJECT_ID') 
REGION = (
    'asia-northeast3'  # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
)
base_url = f'https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi'
emdding_model = 'text-multilingual-embedding-002'
embedding_base_url = f'https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{emdding_model}:predict'

grag = GraphRAG(
    working_dir="./book_example",
    n_checkpoints=2,  # Number of checkpoints to keep
    domain=DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES,
    config=GraphRAG.Config(
        llm_service=OpenAILLMService(
            model="google/gemini-1.5-flash-002", base_url=base_url, api_key=creds.token,
        ),
        embedding_service=OpenAIEmbeddingService(
            model=emdding_model,
            base_url=embedding_base_url,
            api_key=creds.token,
            embedding_dim=768,  # the output embedding dim of the chosen model
        ),
    ),
)

with open("./book.txt") as f:
    grag.insert(f.read())

print(grag.query("Who is Fred?").response)