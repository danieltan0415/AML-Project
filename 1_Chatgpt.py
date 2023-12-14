import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# Initialize the ChatOpenAI object

st.set_page_config(page_title="Welcome to ASL", layout="wide")

st.title("🤠 Welcome to ASL")
st.session_state["OPENAI_API_KEY"]='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'
chat = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
if "cost" not in st.session_state:
    st.session_state["cost"]=[]
elif st.session_state["OPENAI_API_KEY"] != "":
    with st.sidebar:
        model_name=["gpt-4-1106-preview","gpt-4","gpt-3.5-turbo-1106","ft:gpt-3.5-turbo-1106:personal::8T7wYoTJ"]
        option = st.selectbox(
        "Select your model",
        model_name,
        label_visibility="visible",
        disabled=False,
        )
        temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.5,step=0.01)
        max_value=[4096,4096,4096,4096]
        index_of_model=model_name.index(option)
        max_tokens=st.slider('Maximum tokens', min_value=0, max_value=max_value[index_of_model], value=256,step=1)
        chat = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"],temperature=temperature,model_name=option,max_tokens=max_tokens)
    

    for message in st.session_state["messages"]:
        if isinstance(message, HumanMessage):
            
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    prompt = st.chat_input("Type something...")
        
    with  st.spinner("Searching for answer"):
        with get_openai_callback() as cb:
        
            if prompt:
                st.session_state["messages"].append(HumanMessage(content=prompt))
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                ai_message = chat([HumanMessage(content=prompt)])
                    
                
                st.session_state["messages"].append(ai_message)
                
                with st.chat_message("assistant"):
                    st.markdown(ai_message.content)
            
                st.session_state["cost"].append([len(st.session_state["cost"]),cb.total_cost,cb.prompt_tokens,cb.completion_tokens])
                

        
        
            
else:
    with st.container():
        st.warning("Please set your OpenAI API key in the settings page.")

