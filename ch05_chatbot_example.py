import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm 
import ast 
import openai
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Streamlit 페이지 기본 설정
st.set_page_config(page_title="굼봇", page_icon="🤖")

# API 키 설정 방식 개선
api_key = None
# 1. 환경 변수에서 키 확인
if os.getenv("OPENAI_API_KEY"):
    api_key = os.getenv("OPENAI_API_KEY")
# 2. Streamlit 시크릿에서 키 확인
elif "OPENAI_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_KEY"]

# API 키 유효성 검사 및 설정
if not api_key:
    st.error("⚠️ OpenAI API 키가 설정되지 않았습니다!")
    st.info("해결 방법: Streamlit Cloud에서는 'OPENAI_KEY'를 시크릿에 설정하세요. 로컬에서는 .env 파일에 'OPENAI_API_KEY'를 설정하세요.")
    st.stop()

# API 키 설정    
openai.api_key = api_key

def get_embedding(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"임베딩 생성 중 오류가 발생했습니다: {str(e)}")
        return None

folder_path = "./data"
file_name = 'embeddings.csv'
file_path = os.path.join(folder_path, file_name)

# 데이터 로딩 부분
try:
    print("======파일 경로=======")
    print(file_path)
    
    # 임베딩 파일이 존재하는 경우 불러오기
    if os.path.isfile(file_path):
        print("File already exists")
        df = pd.read_csv(file_path)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    else:
        # 파일이 없는 경우만 임베딩 생성
        st.info("임베딩 파일을 생성합니다. 잠시만 기다려주세요...")
        
        # 데이터 폴더 확인
        if not os.path.exists(folder_path):
            st.error(f"데이터 폴더({folder_path})가 존재하지 않습니다.")
            st.stop()
            
        txt_file = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
        if not txt_file:
            st.error(f"데이터 폴더({folder_path})에 .txt 파일이 없습니다.")
            st.stop()
            
        print("======txt 파일 목록 =======")
        print(txt_file)
        data = []
        for file in txt_file:
            txt_file_path = os.path.join(folder_path, file)
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                data.append(text)
        
        df = pd.DataFrame(data, columns=['text'])
        df['embedding'] = df.apply(lambda row: get_embedding(row.text), axis=1)
        
        # None 값이 있는지 확인하고 제거
        if df['embedding'].isna().any():
            st.error("일부 임베딩 생성에 실패했습니다. API 키를 확인하세요.")
            st.stop()
            
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다: {str(e)}")
    st.stop()

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        st.error("질문 임베딩 생성에 실패했습니다.")
        st.stop()
        
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)

    return top_three_doc

def create_prompt(df, query):
    print("======프롬프트 생성=======")
    result = return_answer_candidate(df, query)
    system_role = f"""Your are an aritifical intelligence language model named "굼봇" that specializes in summarizing \
    and answering questions about "굼벵이", developed by developers 조상은.
    You need to take a given document and return a very detailed summary of the document in the query language.
    Here are the document:
        doc 1: """ + str(result.iloc[0]['text']) + """
        doc 2: """ + str(result.iloc[1]['text']) + """
        doc 3: """ + str(result.iloc[2]['text']) + """
    You must return in Korean. Return a accurate answer based on the document. 
    """

    user_content = f"""User question: "{str(query)}"."""

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]
    print(messages)
    return messages 

def generate_answer(messages):
    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.4,
            max_tokens=500
        )
        return result.choices[0].message.content
    except Exception as e:
        st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다. API 키를 확인하거나 나중에 다시 시도해주세요."

print("======이미지 보여주기=======")
# 이미지 파일 확인
try:
    st.image('images/ask_me_chatbot.png')
except Exception as e:
    st.warning("이미지를 불러올 수 없습니다. 그러나 챗봇은 정상적으로 작동합니다.")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('굼벵이에 대해 물어보세요!: ', key='input')
    submitted = st.form_submit_button('전송')


if submitted and user_input:
    try:
        prompt = create_prompt(df, user_input)
        print("======프롬프트 생성=======")
        print(prompt)
        chatbot_response = generate_answer(prompt)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(chatbot_response)
    except Exception as e:
        st.error(f"질문 처리 중 오류가 발생했습니다: {str(e)}")

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))

# 폴더 및 API 키 설정 방법 안내
with st.sidebar:
    st.subheader("사용 가이드")
    st.markdown("""
    ### 데이터 폴더 설정
    - `./data` 폴더에 텍스트 파일(`.txt`)을 넣으세요.
    - 처음 실행 시 임베딩이 자동으로 생성됩니다.
    
    ### API 키 설정
    - 로컬 실행: `.env` 파일에 `OPENAI_API_KEY=your_key_here` 추가
    - Streamlit Cloud: 시크릿에 `OPENAI_KEY` 추가
    """)


