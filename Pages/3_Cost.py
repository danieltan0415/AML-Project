import streamlit as st
import numpy as np
import pandas as pd


if "cost" not in st.session_state:
    st.session_state["cost"]=[]
if "cost1" not in st.session_state:
    st.session_state["cost1"]=[]
if "cost2" not in st.session_state:
    st.session_state["cost2"]=[]
st.subheader("Total Cost analysis")

cost=pd.DataFrame(data=st.session_state["cost"],columns=["Query time","Total cost","Prompt tokens","Completion tokens"]) 

cost1=pd.DataFrame(data=st.session_state["cost1"],columns=["Query time","Total cost","Prompt tokens","Completion tokens"]) 
cost2=pd.DataFrame(data=st.session_state["cost2"],columns=["Query time","Total cost","Prompt tokens","Completion tokens"]) 

sum_cost=sum(cost["Total cost"])
sum_cost1=sum(cost1["Total cost"])
sum_cost2=sum(cost2["Total cost"])



with st.sidebar:
    checked = st.checkbox('Show CHATGPT cost analysis')
    checked1 = st.checkbox('Show PDF chat cost analysis')
    checked2 = st.checkbox('Show Evaluation chat cost analysis')
    
value=0
if checked:
    value = 1
if value:
  
        
    st.line_chart(cost,x="Query time",y="Total cost")
    st.line_chart(cost,x="Query time",y=["Prompt tokens","Completion tokens"])
    st.write(cost)
    st.write("The total cost for normal chatgpt is : {:5f}".format(sum_cost))
value=0
if checked1:
    value = 1
if value:
    st.line_chart(cost1,x="Query time",y="Total cost")
    st.line_chart(cost1,x="Query time",y=["Prompt tokens","Completion tokens"])
    st.write(cost1)
    st.write("The total cost for pdf reader is : {:5f}".format(sum_cost1))
value=0
if checked2:
    value = 1
if value:
    st.line_chart(cost2,x="Query time",y="Total cost")
    st.line_chart(cost2,x="Query time",y=["Prompt tokens","Completion tokens"])
    st.write(cost2)
    st.write("The total cost for evaluation is : {:5f}".format(sum_cost1))
