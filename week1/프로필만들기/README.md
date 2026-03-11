# 프로필만들기

부산대학교 물리학과 4학년 김충현을 소개하는 FastAPI 기반 자기소개 웹페이지 초안입니다.

## 사용 기술

- FastAPI
- Jinja2 템플릿
- Tailwind CSS CDN
- Custom CSS

## 실행 방법

1. 필요한 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

2. 서버를 실행합니다.

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

3. 브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8000
```

Windows에서는 `run_portfolio.bat`를 실행해도 됩니다.
