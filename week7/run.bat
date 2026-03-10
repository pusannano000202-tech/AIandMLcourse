@echo off
echo ========================================
echo Week 7: Large Language Models (LLM)
echo Running all Python files...
echo ========================================
echo.

echo [1/4] Running 01_tokens_and_embeddings.py...
uv run python 01_tokens_and_embeddings.py
if errorlevel 1 (
    echo ERROR: 01_tokens_and_embeddings.py failed!
    pause
    exit /b 1
)
echo.

echo [2/4] Running 02_gpt_bert_architectures.py...
uv run python 02_gpt_bert_architectures.py
if errorlevel 1 (
    echo ERROR: 02_gpt_bert_architectures.py failed!
    pause
    exit /b 1
)
echo.

echo [3/4] Running 03_pretraining_finetuning.py...
uv run python 03_pretraining_finetuning.py
if errorlevel 1 (
    echo ERROR: 03_pretraining_finetuning.py failed!
    pause
    exit /b 1
)
echo.

echo [4/4] Running 04_claude_api_simple.py...
uv run python 04_claude_api_simple.py
if errorlevel 1 (
    echo ERROR: 04_claude_api_simple.py failed!
    pause
    exit /b 1
)
echo.

echo ========================================
echo All files completed successfully!
echo Check the outputs/ directory for results.
echo ========================================
pause
