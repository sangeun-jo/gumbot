import os
import pandas as pd
import numpy as np
from numpy.linalg import norm 
import ast 
import openai
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# 파일 경로 설정
folder_path = "C:\\dev\\chat-gpt-prg\\ch05\\data"
file_name = 'embeddings.csv'
embeddings_file_path = os.path.join(folder_path, file_name)

# 파일 경로 확인 출력
print("======파일 경로=======")
print(f"임베딩 파일 경로: {embeddings_file_path}")
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"폴더 존재 여부: {os.path.isdir(folder_path)}")
print(f"임베딩 파일 존재 여부: {os.path.isfile(embeddings_file_path)}")

# 폴더 내용 확인
if os.path.isdir(folder_path):
    print("======폴더 내용=======")
    for f in os.listdir(folder_path):
        print(f)

# 임베딩 처리
if os.path.isfile(embeddings_file_path):
    print("임베딩 파일이 이미 존재합니다. 파일을 로드합니다.")
    df = pd.read_csv(embeddings_file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
else:
    print("임베딩 파일이 존재하지 않습니다. 새로 생성합니다.")
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    print(f"======txt 파일 {len(txt_files)}개 발견=======")
    
    data = []
    for txt_file in txt_files:
        txt_file_path = os.path.join(folder_path, txt_file)
        print(f"처리 중: {txt_file}")
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                data.append(text)
        except Exception as e:
            print(f"파일 {txt_file} 읽기 오류: {e}")
    
    df = pd.DataFrame(data, columns=['text'])
    print(f"{len(df)}개 텍스트 데이터 로딩 완료")
    
    print("임베딩 생성 중...")
    df['embedding'] = df.apply(lambda row: get_embedding(row.text), axis=1)
    
    print(f"결과 저장 중: {embeddings_file_path}")
    df.to_csv(embeddings_file_path, index=False, encoding='utf-8-sig')
    print("임베딩 생성 및 저장 완료") 