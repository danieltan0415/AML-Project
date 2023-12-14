import streamlit as st
from pathlib import Path
from langchain.retrievers import ParentDocumentRetriever
from langchain.document_loaders import TextLoader, PyMuPDFLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain, ConversationChain
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.evaluation import load_evaluator
from langchain.schema import Document
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time
from langchain.memory import ConversationBufferMemory


answer=False
st.session_state["OPENAI_API_KEY"]='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'
if "qa" not in st.session_state:
    st.session_state.qa=None
if "cost1" not in st.session_state:
    st.session_state["cost1"]=[]
with st.sidebar:
    st.subheader(' :blue[Upload your document here bro!] :sunglasses:')
    loaders = [
            PyMuPDFLoader('byd_info.pdf'),
            PyMuPDFLoader('BYD-ATTO-Malaysia.pdf'),
            PyMuPDFLoader('dolphin-pricelist.pdf'),
            PyMuPDFLoader('dolphin-specs.pdf'),
            
        ]
    with st.form(key="Form :", clear_on_submit = True):
        
        File = st.file_uploader(label = "Upload file", type=["pdf"],accept_multiple_files=True)
        Submit = st.form_submit_button(label='Upload')

    if Submit and File:
        st.markdown("**The file is sucessfully Uploaded.**")
        for file in File:

        # Save uploaded file to 'F:/tmp' folder.
            save_folder = r'C:\Users\fei\Desktop\study\Year3 Semester1\AIT301\final'
            save_path = Path(save_folder, file.name)
            with open(save_path, mode='wb') as w:
                w.write(file.getvalue())

            if save_path.exists():
                st.success(f'File {file.name} is successfully saved!')
        

            loaders.append(PyMuPDFLoader(file.name))
    x=st.button('Embedding Documents')
    model_name=["ft:gpt-3.5-turbo-1106:personal::8T7wYoTJ","gpt-4-1106-preview","gpt-4","gpt-3.5-turbo-1106"]
    option = st.selectbox(
            "Select your model",
            model_name,
            label_visibility="visible",
            disabled=False,
        )
    temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.0,step=0.01)
    llm=ChatOpenAI(temperature=temperature,openai_api_key="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",model_name=option)
    if x:
        docs = []
        for l in loaders:
            docs.extend(l.load())
        print(len(docs))
        with st.spinner("Analysising the PDF..."):
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            vectorstore = Chroma(collection_name="split_parents", embedding_function=OpenAIEmbeddings(openai_api_key="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"))
            store = InMemoryStore()

            big_chunks_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                
            )
            big_chunks_retriever.add_documents(docs)

        general_system_template = """ 
               
                

                 <ctx>
                {context}
                </ctx>
                ------
                <hs>
                {history}
                </hs>
                ------
                """
        template = """
            Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:\
            
            You are a assistant chatbot based on pdf.\
            if user greet to you, you give him back greeting and must ask him to ask your question\
            You answer the question only base on pdf, dont give me any answer from your dataset\
            The answer do not over 100 words, write as short answer as possible to user\
            ------
            <ctx>
            {context}
            </ctx>
            ------
            <hs>
            {history}
            </hs>
            ------
            {question}
            Answer:
            """
        qa_prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )
            
        
        
        qa = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=big_chunks_retriever,
                                        chain_type_kwargs={
                                            "verbose": True,
                                            "prompt": qa_prompt,
                                            "memory": ConversationBufferMemory(
                                                memory_key="history",
                                                input_key="question"),
                                        }
                                 )
        st.session_state.qa=qa
        st.write("Finish Embedding!!!")
        
if "inputtest" not in st.session_state:

    st.session_state["inputtest"]=[]
if "answertest" not in st.session_state:
    st.session_state["answertest"]=[]

    

if st.session_state.qa:
    
    st.subheader('You can chat with me now')
    question=st.chat_input("Ask a question about your documents")
    if question :
        with get_openai_callback() as cb:
            with st.spinner("Searching for the answer..."):
                    x=st.session_state.qa.run(question)
                    st.session_state["answertest"].append(x)
                    st.session_state["inputtest"].append(question)
                    st.session_state["cost1"].append([len(st.session_state["cost1"]),cb.total_cost,cb.prompt_tokens,cb.completion_tokens])
    for i in range(len(st.session_state.inputtest)):
        
        with st.chat_message("user"): 
            st.markdown(st.session_state.inputtest[i])
        with st.chat_message("assistant"):
            st.markdown(st.session_state.answertest[i])
else:
    st.header('🤠Welcome to our :blue[customer service center] ',divider='rainbow')
    st.subheader('Please Embedding first')
            
            



    