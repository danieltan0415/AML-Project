import streamlit as st

if "result" not in st.session_state:
    st.session_state["result"] = []
if "result1" not in st.session_state:
    st.session_state["result1"] = []
if "input" not in st.session_state:
    st.session_state["input"] = []
if 'final_score' not in st.session_state:
    st.session_state["final_score"] = ""
if "accurancy" not in st.session_state:
    st.session_state["accurancy"]=""
if st.session_state.input:
    max=len(st.session_state.input)-1
    option = st.slider('Select the question id you want to check', min_value=0, max_value=max, value=0,step=1)

write=False 

with st.sidebar:
    check=st.checkbox("Show the evaluation details")
    if check:
        if option:
            write=True
Write=False   
with st.sidebar:
    check=st.checkbox("Show the total evaluation result")
    if check:
        Write=True



if write:
    if option==0:
        st.write(st.session_state["result"][0])
    else:
        st.write(st.session_state["result"][option])
if Write:
    
    if option==0:
        st.write(st.session_state["result1"][0])
    else:
        st.write(st.session_state["result1"][option])

with st.sidebar:
    x=st.button("Show final result")
    if x:
        
        f"The evaluation result is : {st.session_state.final_score}"
        f"0~100  \n larger -> better"
        f"The accuracy is : {st.session_state.accurancy} %"


  

       
    