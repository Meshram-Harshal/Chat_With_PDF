import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pandasai.llm.local_llm import LocalLLM
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st

# Loading environment variables from .env file
load_dotenv()

# Function to initialize conversation chain with GROQ language model
groq_api_key = os.environ['GROQ_API_KEY']

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-70b-8192",
    temperature=0.2
)

# Function to process PDF files
def process_pdfs(files):
    texts = []
    metadatas = []
    for file in files:
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Create metadata for each chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    return chain

# Function to chat with CSV data
def chat_with_csv(df, query):
    # Initialize LocalLLM with Meta Llama 3 model
    llm = LocalLLM(
        api_base="http://localhost:11434/v1",
        model="llama3")
    # Initialize SmartDataframe with DataFrame and LLM configuration
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    # Chat with the DataFrame using the provided query
    result = pandas_ai.chat(query)
    return result

# Set layout configuration for the Streamlit page
st.set_page_config(layout='wide')
# Set title for the Streamlit application
st.title("Multi-file ChatApp powered by LLM")

# Select file type for upload
file_type = st.sidebar.radio("Select file type", ('PDF', 'CSV'))

if file_type == 'PDF':
    input_files = st.sidebar.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)

    if input_files:
        # Process PDFs
        @st.cache_resource
        def cached_process_pdfs(files):
            return process_pdfs(files)
        
        chain = cached_process_pdfs(input_files)
        st.success(f"Processing {len(input_files)} PDF files done. You can now ask questions!")
        st.session_state.chain = chain

        if 'chain' in st.session_state:
            user_query = st.text_input("Ask a question about the PDFs:")
            if user_query:
                chain = st.session_state.chain
                res = chain.invoke(user_query)
                answer = res["answer"]
                source_documents = res["source_documents"]

                text_elements = []  # Initialize list to store text elements

                # Process source documents if available
                if source_documents:
                    for source_idx, source_doc in enumerate(source_documents):
                        source_name = f"source_{source_idx}"
                        # Create the text element referenced in the message
                        text_elements.append(
                            source_doc.page_content
                        )
                    source_names = [source_name for source_name in text_elements]

                    # Add source references to the answer
                    if source_names:
                        answer += f"\nSources: {', '.join(source_names)}"
                    else:
                        answer += "\nNo sources found"

                st.write(answer)
                for element in text_elements:
                    st.write(element)

else:
    input_files = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

    if input_files:
        selected_file = st.selectbox("Select a CSV file", [file.name for file in input_files])
        selected_index = [file.name for file in input_files].index(selected_file)

        # Load and display the selected CSV file
        st.info("CSV uploaded successfully")
        data = pd.read_csv(input_files[selected_index])
        st.dataframe(data.head(3), use_container_width=True)

        # Enter the query for analysis
        st.info("Chat Below")
        input_text = st.text_area("Enter the query")

        # Perform analysis
        if input_text:
            if st.button("Chat with CSV"):
                st.info("Your Query: " + input_text)
                result = chat_with_csv(data, input_text)
                st.success(result)
