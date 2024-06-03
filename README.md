## dependencies required

``
pip install -r requirements.txt
``

python accelerate

``
pip install accelerate
``

sentence Transformers

``
pip install -U sentence-transformers
``

vicuna embeddings

minilm v6 sentence transformer
[resp](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

vector store

[chroma db](https://docs.trychroma.com/getting-started)

``
pip install chromadb
``

[streamlit](https://docs.streamlit.io/get-started/installation)

``
pip install streamlit
``

parquet python

``
pip install parquet
``

llm pipeline hugging face

inference langchain

## closed-source alternatives
openai embedding
pinecone

## logic
langchain
embeddings
vector store
chains by langchain
llm / embeddings

## knowledge point
vectore db vs sql db
inbuilt algorithms
semantic search
levenstein jacquard

## reasons
lower dimensional space
inbuilt default algorithms
cosine similarity

## directory should contain
db docs lamini-t4-738m pycache .venv license requirements.txt .gitignore

[llm](https://huggingface.co/MBZUAI/LaMini-T5-738M)

## Get Started

``
code .
``

``
conda activate deeplearning
``

### requirements.txt
langchain
streamlit	UI for citation
transformers	
requests	
torch
einops
accelerate	large models
bitsandbytes	
pdfminer.six pi pdf
bs4		
sentence_transformers	
chromadb

constants.py
app.py
ingest.py

## ingest.py

what's going on in ingest.py

import things

define directory

main function

recursive text splitter

create embeddings store in Chroma

``
python ingest.py
``

## app.py

``
streamlit run app.py
``

 validate response
 
 client requirements
 
 summarization and retrieval
