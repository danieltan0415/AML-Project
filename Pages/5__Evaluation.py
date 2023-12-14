from langchain.retrievers import ParentDocumentRetriever
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.evaluation import load_evaluator
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import Document
import time
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.evaluation import load_evaluator
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
import time
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
loaders = [
    PyMuPDFLoader('byd_info.pdf'),
    PyMuPDFLoader('BYD-ATTO-Malaysia.pdf'),
    PyMuPDFLoader('dolphin-pricelist.pdf'),
    PyMuPDFLoader('dolphin-specs.pdf'),
]
docs = []
for l in loaders:
    docs.extend(l.load())
with st.sidebar:
    st.subheader(' :blue[Step 1:] :sunglasses:')
    x=st.button('Embedding Documents ')
    if x:
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
                You are a assistant chatbot based on pdf.\
                You answer the question only base on pdf, dont give me any answer from your dataset
                The answer do not over 100 words, write as short answer as possible to user
                \
                {context}
                """
        general_user_template = "Question:```{question}```"
        messages = [
                    SystemMessagePromptTemplate.from_template(general_system_template),
                    HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages( messages )
        
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0,openai_api_key="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"),
                                        chain_type="stuff",
                                        retriever=big_chunks_retriever,
                                        chain_type_kwargs={
                                                    "verbose": False,
                                                    "prompt": qa_prompt
                                                }
                                 )
        st.session_state.qa=qa
        
        
question=["Considering the differences in battery capacity between the standard and extended versions of the BYD ATTO 3, how does this impact their driving range?",
    "In terms of safety, how do the BYD ATTO 3 and Dolphin compare, and what should a safety-conscious buyer consider?",
    "How do the interior design elements of the BYD ATTO 3 and Dolphin cater to a tech-savvy customer?",
    "How does the warranty coverage differ between the BYD ATTO 3 and BYD Dolphin, and what are the implications for a potential buyer?",
    "What are the components of drive unit",
    "What is the price of BYD Atto 3 (personal vehicle)",
    "Could you inform me about the inspection fee for the BYD Dolphin?",
    "Is there any road tax applicable to the BYD Dolphin?",
    "how much time do i spend to charge the car?",
    "What kind of advanced technology is used in the BYD Dolphin?",
    "how far can i drive with this car?"]


   



check=False
if "input" not in st.session_state:

    st.session_state["input"]=[]
if "answer" not in st.session_state:
    st.session_state["answer"]=[]
answer=[]
with st.sidebar:
    st.subheader(' :blue[Step 2:] 🎄')
    x=st.button('Search for answer')
    if x:
        check =True
if check:
    number=0
    f"There are {len(question)} questions in the searching list "
    for i in question:
        with st.spinner(f"Searching for the answer of question {number+1},  Current question is : {i}"):
            x=st.session_state.qa.run(i)
            st.session_state["answer"].append(x)
            st.session_state["input"].append(i)
            print(x)
            number=number+1
            #time.sleep(20)
           

for i in range(len(st.session_state.input)):
    
    with st.chat_message("user"): 
        st.markdown(st.session_state.input[i])
    with st.chat_message("assistant"):
        st.markdown(st.session_state.answer[i])
reference_list=[
     "The extended version, with a larger battery capacity (60.48kWh compared to 49.92kWh in the standard version), offers a longer driving range of 480km, compared to 410km in the standard version",
    "Both models likely have advanced safety features, but specific details like the number of airbags, safety ratings, and assistance systems would need to be compared for an informed decision (specific safety features of both models would be required for a complete comparison).",
    "oth models feature modern interiors with touchscreens and connectivity options, but the BYD ATTO 3's larger touchscreen and specific tech features might appeal more to tech-savvy customers.",
    "The BYD ATTO 3 offers a 6-year vehicle warranty, an 8-year high-voltage blade battery warranty, and an 8-year drive unit warranty. The specifics of the BYD Dolphin's warranty would need to be compared to assess the differences (details from Dolphin's warranty information would be required for a complete answer).",
    " Motor 2. Motor Controller 3. DC Assembly 4. High Voltage 5. Electric Control Assembly ",
    " Standard Model: RM 149,800   Extended Model: RM 167,800",
    "The inspection fee for the BYD Dolphin is RM 200.",
    "The road tax for the BYD Dolphin is exempted.",
     "Blade Battery can support BYD ATTO 3 to charge from 0% to 80% within 50 mins*",
    "The BYD Dolphin uses Blade Battery E-platform 3.0 technology.",
    "Byd atto 3 can drive at a maximum range of 521km"
    ]

if "result" not in st.session_state:
    st.session_state["result"]=[]
if "result1" not in st.session_state:
    st.session_state["result1"]=[]
check=False
with st.sidebar:
    st.subheader(' :blue[Step 3:] 🥳')
    x=st.button('Evaluate the result')
    if x:
        check =True
if check:

    llm = ChatOpenAI(openai_api_key=st.session_state.OPENAI_API_KEY,model_name='gpt-3.5-turbo')
        #x=st.button('Evaluation the multi-pdf chatgpt')
        #if x:
    evaluator = load_evaluator("labeled_criteria", criteria="correctness",llm=llm)
    evaluator_a = load_evaluator("string_distance")
    number=0
    correct=0
    result=[]
    result1=[]
    for i in range(len(question)):
        print(f"Evaluation {i}")
        input=st.session_state.input[i]
        answer1=st.session_state.answer[i]
        reference=reference_list[i]
        result.append(evaluator.evaluate_strings(
            input=input,
            prediction=answer1,
            reference=reference,
            ))
        x=evaluator_a.evaluate_strings(prediction=input,reference=reference)
        result1.append(x["score"])
        number=number+1
        
    
        st.session_state["result1"]=result1
        
        st.session_state["result"]=result
        #time.sleep(20)
      
check=False
with st.sidebar: 
    x=st.button("Show the evaluation results")
    if x:
        check=True
if check:
    correct=0
    final_score=0
    for i in range(len(question)):
            if st.session_state["result"][i]["score"]==1:
                correct=correct+1
            elif " Y " in st.session_state["result"][i]:
                st.session_state["result"][i]["score"]=1
                correct=correct+1
            else:
                st.session_state["result"][i]["score"]=0

            score1=(st.session_state["result"][i]["score"]*0.7+(1- st.session_state["result1"][i])*0.3)*100
            final_score=final_score+score1
    final_score=final_score/len(question)
    st.session_state.final_score=final_score
    st.session_state.accurancy=(correct/len(question))*100
    st.write(f"Score: {st.session_state.final_score}")
    st.write(f"Accurancy: {st.session_state.accurancy}%")
st.write(st.session_state.result)
st.write(st.session_state.result1)