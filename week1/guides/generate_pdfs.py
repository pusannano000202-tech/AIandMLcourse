from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Ensure output directory exists
output_dir = 'week1/guides'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Try to register a Korean font if available, otherwise fallback to standard
# In a typical Windows environment, Malgun Gothic might be available.
# If not, we will use standard English fonts to avoid errors, but warn about Korean support.
try:
    pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
    font_name = 'Malgun'
    print("Korean font 'Malgun Gothic' registered.")
except:
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
        font_name = 'Arial'
        print("Korean font not found, using Arial.")
    except:
        font_name = 'Helvetica'
        print("System fonts not found, using Helvetica (No Korean support).")

def create_pdf(filename, title, content_sections, references):
    c = canvas.Canvas(os.path.join(output_dir, filename), pagesize=A4)
    width, height = A4
    
    # Title
    c.setFont(font_name, 24)
    c.drawCentredString(width / 2, height - 1.5 * inch, title)
    
    # Content
    y = height - 2.5 * inch
    c.setFont(font_name, 12)
    line_height = 18
    
    for section_title, lines in content_sections:
        # Section Title
        if y < 1.5 * inch:
            c.showPage()
            y = height - 1.5 * inch
            c.setFont(font_name, 12)
            
        c.setFont(font_name, 14)
        c.drawString(1 * inch, y, section_title)
        y -= line_height * 1.5
        c.setFont(font_name, 11)
        
        for line in lines:
            if y < 1 * inch:
                c.showPage()
                y = height - 1.5 * inch
                c.setFont(font_name, 11)
            c.drawString(1.2 * inch, y, line)
            y -= line_height
        
        y -= line_height
        
    # References
    if y < 2 * inch:
        c.showPage()
        y = height - 1.5 * inch
        
    c.setFont(font_name, 14)
    c.drawString(1 * inch, y, "참고문헌 (References)")
    y -= line_height * 1.5
    c.setFont(font_name, 10)
    
    for ref in references:
        if y < 1 * inch:
            c.showPage()
            y = height - 1.5 * inch
            c.setFont(font_name, 10)
        c.drawString(1.2 * inch, y, f"- {ref}")
        y -= line_height

    c.save()
    print(f"Created {filename}")

# 1. Antigravity Installation
create_pdf(
    "01_Antigravity_Installation.pdf",
    "Antigravity 설치 가이드",
    [
        ("1. Antigravity란?", [
            "Antigravity는 Google Deepmind에서 개발한 고급 AI 코딩 에이전트입니다.",
            "VS Code 및 Cursor와 같은 IDE 내에서 실행되며,",
            "복잡한 코딩 작업을 자율적으로 수행할 수 있습니다."
        ]),
        ("2. 설치 방법 (VS Code/Cursor Extension)", [
            "1) Cursor 또는 VS Code를 실행합니다.",
            "2) 왼쪽 사이드바의 'Extensions' (블록 모양 아이콘)을 클릭합니다.",
            "3) 검색창에 'Antigravity'를 입력합니다.",
            "4) 'Google Deepmind - Antigravity'를 찾아 'Install'을 클릭합니다.",
            "5) 설치 후 IDE를 재시작합니다."
        ]),
        ("3. 초기 설정", [
            "1) 설치가 완료되면 사이드바에 Antigravity 아이콘이 나타납니다.",
            "2) 아이콘을 클릭하고 Google 계정으로 로그인합니다.",
            "3) 필요한 권한을 승인하면 사용 준비가 완료됩니다."
        ])
    ],
    [
        "Google Deepmind Antigravity Official Documentation",
        "VS Code Marketplace: Antigravity Extension"
    ]
)

# 2. Cursor Installation
create_pdf(
    "02_Cursor_Installation.pdf",
    "Cursor IDE 설치 가이드",
    [
        ("1. Cursor란?", [
            "VS Code를 기반으로 만들어진 AI 중심의 코드 에디터입니다.",
            "GPT-4, Claude 등의 최신 AI 모델이 내장되어 있어",
            "코드 작성, 수정, 디버깅을 대화하듯이 처리할 수 있습니다."
        ]),
        ("2. 다운로드 및 설치", [
            "1) 웹브라우저에서 https://cursor.sh 에 접속합니다.",
            "2) 메인 화면의 'Download' 버튼을 클릭합니다.",
            "   (Windows, Mac, Linux 버전을 자동으로 감지합니다)",
            "3) 다운로드된 설치 파일(Installer)을 실행합니다.",
            "4) 설치 마법사의 지시에 따라 설치를 완료합니다."
        ]),
        ("3. 기본 설정", [
            "1) 처음 실행 시 'Keyboard Interface'를 선택할 수 있습니다.",
            "   (VS Code 단축키를 그대로 사용할 수 있습니다)",
            "2) 'Import Extensions'를 통해 기존 VS Code 설정을 가져올 수 있습니다.",
            "3) Ctrl+K (코드 생성), Ctrl+L (채팅) 단축키를 기억하세요."
        ])
    ],
    [
        "Cursor Official Website (https://cursor.sh)",
        "Cursor User Guide"
    ]
)

# 3. uv Installation
create_pdf(
    "03_uv_Installation.pdf",
    "uv 패키지 매니저 설치 가이드",
    [
        ("1. uv란?", [
            "Rust로 작성된 초고속 Python 패키지 관리자입니다.",
            "pip, poetry 등을 대체할 수 있으며 속도가 매우 빠릅니다.",
            "가상환경 관리와 패키지 설치를 통합적으로 제공합니다."
        ]),
        ("2. 설치 방법 (Windows)", [
            "PowerShell을 관리자 권한으로 실행하고 다음 명령어를 입력합니다:",
            "",
            "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"",
            "",
            "설치가 완료되면 터미널을 껐다 켜거나 `refreshenv`를 입력합니다."
        ]),
        ("3. 기본 사용법", [
            "- 버전 확인: uv --version",
            "- 가상환경 생성: uv venv",
            "- 패키지 설치: uv pip install numpy pandas",
            "- 스크립트 실행: uv run python main.py"
        ])
    ],
    [
        "Astral uv GitHub Repository (https://github.com/astral-sh/uv)",
        "Python Packaging User Guide"
    ]
)

# 4. Development Environment
create_pdf(
    "04_Development_Environment.pdf",
    "현대적인 AI 개발 환경 구축",
    [
        ("1. 개발 환경의 중요성", [
            "효율적이고 재현 가능한 연구를 위해서는 좋은 도구가 필수적입니다.",
            "특히 AI/물리학 융합 연구는 복잡한 라이브러리 의존성을 가집니다."
        ]),
        ("2. 핵심 도구 소개", [
            "1) IDE (Cursor):",
            "   - AI가 코딩의 단순 반복 작업을 대신해줍니다.",
            "   - 복잡한 수식 구현이나 에러 디버깅에 탁월합니다.",
            "",
            "2) 패키지 매니저 (uv):",
            "   - 프로젝트별로 독립된 환경(Virtual Environment)을 만듭니다.",
            "   - 버전 충돌을 방지하고 설치 속도를 획기적으로 줄입니다.",
            "",
            "3) 버전 관리 (Git):",
            "   - 연구 과정의 모든 변경 사항을 기록합니다.",
            "   - 실수로 코드를 지워도 언제든 복구할 수 있습니다."
        ]),
        ("3. 권장 워크플로우", [
            "1. uv로 가상환경 생성 (uv venv)",
            "2. 필요한 라이브러리 설치 (uv pip install ...)",
            "3. Cursor에서 코드 작성 및 AI 도움 받기",
            "4. Git으로 진행 상황 저장 (git commit)"
        ])
    ],
    [
        "Wilson G. et al., 'Good enough practices in scientific computing', PLOS Biology (2017)",
        "The Turing Way Community, 'The Turing Way: A handbook for reproducible data science'"
    ]
)

# 5. AI in Physics
create_pdf(
    "05_AI_in_Physics.pdf",
    "물리학에서의 인공지능 활용",
    [
        ("1. 개요", [
            "AI, 특히 딥러닝은 물리학의 난제들을 해결하는 새로운 도구가 되고 있습니다.",
            "데이터 분석을 넘어 물리 법칙을 학습하거나 시뮬레이션을 가속화합니다."
        ]),
        ("2. 주요 응용 분야", [
            "1) 입자 물리학 (High Energy Physics):",
            "   - CERN의 LHC 충돌 데이터에서 희귀한 입자 붕괴를 식별합니다.",
            "   - 배경 노이즈를 제거하고 신호를 증폭하는 데 사용됩니다.",
            "",
            "2) 천체 물리학 (Astrophysics):",
            "   - 은하의 형태 분류, 외계 행성 탐색에 활용됩니다.",
            "   - 중력파 신호 검출 정밀도를 높입니다.",
            "",
            "3) 물질 과학 (Materials Science):",
            "   - 새로운 초전도체나 배터리 소재를 설계합니다 (Generative Models).",
            "   - 분자 동역학 시뮬레이션을 가속화합니다.",
            "",
            "4) 미분방정식 풀이 (PINNs):",
            "   - Physics-Informed Neural Networks를 사용하여",
            "   - 복잡한 편미분방정식(Navier-Stokes 등)을 데이터 없이도 풉니다."
        ]),
        ("3. 전망", [
            "AI는 물리학자를 대체하는 것이 아니라, 물리학자의 직관을 확장하는 도구입니다.",
            "'AI for Science'는 차세대 과학 연구의 핵심 패러다임이 될 것입니다."
        ])
    ],
    [
        "Carleo, G. et al., 'Machine learning and the physical sciences', Rev. Mod. Phys. 91 (2019)",
        "Karniadakis, G. E. et al., 'Physics-informed machine learning', Nature Reviews Physics 3 (2021)"
    ]
)
