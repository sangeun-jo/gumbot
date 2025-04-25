# 굼봇 챗봇 애플리케이션

굼벵이에 대해 질문하고 답변을 받을 수 있는 스트림릿 기반 챗봇 애플리케이션입니다.

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

### 로컬 실행
1. `.env` 파일을 생성하고 OpenAI API 키를 설정합니다:
```
OPENAI_API_KEY=your_api_key_here
```

2. 애플리케이션 실행:
```bash
streamlit run ch05_chatbot_example.py
```

### Streamlit Cloud 배포
1. Streamlit Cloud 대시보드에서 시크릿 설정:
   - `OPENAI_KEY`: 당신의 OpenAI API 키

## 주요 기능
- 굼벵이에 관한 질문 입력 및 답변 제공
- 임베딩 기반 문서 검색을 통한 정확한 답변 생성