from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="김충현 소개 페이지",
    description="초발수표면 연구와 부산대학교 4학년 김충현을 소개하는 웹페이지 초안",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

PROFILE = {
    "name": "김충현",
    "headline": "초발수표면을 연구하는 부산대학교 4학년",
    "summary": (
        "표면의 미세구조와 젖음성 제어에 관심을 두고, "
        "물방울이 머무르지 않는 초발수표면의 물리와 응용 가능성을 탐구하고 있습니다."
    ),
    "university": "부산대학교",
    "major": "물리학과",
    "year": "4학년",
    "research_theme": "Superhydrophobic Surface",
    "draft_note": (
        "현재 버전은 자기소개용 랜딩 페이지 초안이며, "
        "세부 연구 데이터와 성과는 추후 보강 예정입니다."
    ),
}

RESEARCH_POINTS = [
    {
        "title": "핵심 관심사",
        "body": "미세 패턴 구조, 표면 에너지 제어, 접촉각 분석을 중심으로 초발수 현상을 이해합니다.",
    },
    {
        "title": "연구 방향",
        "body": "자연계의 연잎 효과에서 영감을 받아 self-cleaning과 anti-wetting 응용 가능성을 살핍니다.",
    },
    {
        "title": "작업 방식",
        "body": "실험 설계, 결과 정리, 시각화까지 한 흐름으로 묶어 물리적 해석이 가능한 형태로 다룹니다.",
    },
]

TIMELINE = [
    {
        "title": "학부 과정",
        "body": "부산대학교 물리학과에서 기초 물리, 수치 계산, 데이터 해석 역량을 쌓고 있습니다.",
    },
    {
        "title": "현재 연구",
        "body": "초발수표면의 구조와 성능 사이의 관계를 관찰하며 실험 중심의 관점을 키우고 있습니다.",
    },
    {
        "title": "다음 목표",
        "body": "연구 내용을 더 체계적으로 정리해 포트폴리오와 프로젝트 기록으로 확장할 계획입니다.",
    },
]

KEYWORDS = [
    "Superhydrophobicity",
    "Surface Physics",
    "Contact Angle",
    "Microstructure",
    "Self-Cleaning",
]

SNAPSHOTS = [
    {"label": "Research Core", "value": "초발수표면"},
    {"label": "Current Stage", "value": "학부 4학년"},
    {"label": "Base", "value": "부산대학교"},
]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "profile": PROFILE,
            "research_points": RESEARCH_POINTS,
            "timeline": TIMELINE,
            "keywords": KEYWORDS,
            "snapshots": SNAPSHOTS,
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
