import streamlit as st
import os
import base64
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings

# Suppress the warning for streamlit_chat
warnings.filterwarnings("ignore", message="Your warning message here")

from streamlit_chat import message


st.set_page_config(layout="wide")

@st.cache_resource
def data_ingestion():
    texts = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
                for doc in text_splitter.split_documents(documents):
                    # Convert Document object to string
                    text = str(doc)
                    texts.append(text)
    return texts

def get_most_relevant_document(query, texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    most_similar_doc_index = similarity_scores.argmax()
    return texts[most_similar_doc_index]

def process_answer(instruction, texts, model, tokenizer):
    query = instruction['query']

    # Iterate through each text in the PDF
    all_answers = []
    for text in texts:
        # Tokenize the text and the query
        inputs = tokenizer(query, text, return_tensors="pt", max_length=512, truncation=True)

        # Perform question answering
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the answer
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Find the most probable answer
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        all_answers.append(answer)

    # Return the most relevant answer
    return max(all_answers, key=len)

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://www.sbjit.edu.in/'>SBJITMR with ❤️ </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('PDF is being processed...'):
                texts = data_ingestion()
                # Load Mistral 7B model and tokenizer
                model_name = "TheBloke/Mistral-7B-QA-v0.1"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            st.success('PDF is processed successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            if user_input:
                answer = process_answer({'query': user_input}, texts, model, tokenizer)
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            if st.session_state["generated"]:
                display_conversation(st.session_state)


if __name__ == "__main__":
    main()
