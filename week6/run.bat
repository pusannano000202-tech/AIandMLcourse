@echo off
echo ========================================
echo Week 6: Transformer and Attention
echo Running all Python files...
echo ========================================
echo.

echo [1/5] Running 01_attention_basics.py...
uv run python 01_attention_basics.py
if errorlevel 1 (
    echo ERROR: 01_attention_basics.py failed!
    pause
    exit /b 1
)
echo.

echo [2/5] Running 02_self_attention.py...
uv run python 02_self_attention.py
if errorlevel 1 (
    echo ERROR: 02_self_attention.py failed!
    pause
    exit /b 1
)
echo.

echo [3/5] Running 03_positional_encoding.py...
uv run python 03_positional_encoding.py
if errorlevel 1 (
    echo ERROR: 03_positional_encoding.py failed!
    pause
    exit /b 1
)
echo.

echo [4/5] Running 04_transformer_block.py...
uv run python 04_transformer_block.py
if errorlevel 1 (
    echo ERROR: 04_transformer_block.py failed!
    pause
    exit /b 1
)
echo.

echo [5/5] Running 05_sequence_modeling.py...
uv run python 05_sequence_modeling.py
if errorlevel 1 (
    echo ERROR: 05_sequence_modeling.py failed!
    pause
    exit /b 1
)
echo.

echo ========================================
echo All files completed successfully!
echo Check the outputs/ directory for results.
echo ========================================
pause
