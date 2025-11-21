"""
@Author l
@Date 2025/8/18 17:07
"""
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from pydantic import SecretStr
import dotenv
import os

dotenv.load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODEL"),
    base_url=os.getenv("BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"))

st.title("Dataframe")
types = st.radio("选择数据格式", ["csv", "excel"], horizontal=True)
upload_file = st.file_uploader("上传文件", type='xlsx' if types == 'excel' else 'csv')
preview_placeholder = st.empty()
if upload_file:
    if types == 'excel':
        sheet_names = st.radio("选择sheet", [sheet for sheet in pd.ExcelFile(upload_file).sheet_names], horizontal=True)
        df = pd.read_excel(upload_file, sheet_name=sheet_names)
    else:
        df = pd.read_csv(upload_file)
    with st.expander("数据预览"):
        st.dataframe(df)
    qus = st.chat_input("请输入问题")
    if qus:
        st.chat_message("human").write(qus)
        with st.spinner("思考中..."):
            agent = create_pandas_dataframe_agent(
                llm=model,
                df=df,
                verbose=True,
                max_iterations=10,
                allow_dangerous_code=True,
                agent_executor_kwargs={
                    'handle_parsing_errors': True
                }
            )
            result = agent.invoke({"input": qus})
            st.chat_message("ai").write(result['output'])
            with st.expander("更新后数据预览", expanded=True):
                st.dataframe(df)
