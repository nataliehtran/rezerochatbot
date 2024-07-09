import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#langchain is a framework 
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "" #openAI key

#Upload PDF file
st.header("My First Chatbot")
with st.sidebar: 
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract the text
if file is not None: 
    pdf_reader = PdfReader(file)
    text = "" 
    for page in pdf_reader.pages: 
        text += page.extract_text()
    #reads and extracts the text into the text variable
    #text will have all the pages of the text 
    #st.write(text) write is print using streamlit

    #Break it into chunks --> Lang Chain has text splitter 
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n", 
        #where do you want me to break the text
        chunk_size = 1500, 
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(text) 
    #st.write(chunks)

    #Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #Create vector store --> FAISS created by Facebook
    vector_store = FAISS.from_texts(chunks, embeddings) 
    #A_123 A is the chunk 123 is the embedding
    # generating embeddings using OpenAI, initializing FAISS database
    #storing chunks and embeddings

    #Get user question
    user_question = st.text_input("Type your question here")

    #Do similarity search 
    if user_question: 
        match = vector_store.similarity_search(user_question)
        #st.write(match)
        #Output results 
        #chain ---> take question -> relevant documents 
        #pass it to LLM, generate the output 

        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            #0,1,2 lower the value, asking LLM to keep it speciftic 
            max_tokens=1000,
            model_name = "gpt-4o"
        )
        #chain ---> take question -> relevant documents 
        chain = load_qa_chain(llm, chain_type = "stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
