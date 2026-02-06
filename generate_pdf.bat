@echo off
echo ========================================
echo   Research Paper PDF Generator
echo ========================================
echo.

REM Check if PlantUML is available (try multiple methods)
set PLANTUML_CMD=
where plantuml >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PLANTUML_CMD=plantuml
) else if exist plantuml.bat (
    set PLANTUML_CMD=plantuml.bat
) else if exist plantuml.jar (
    where java >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PLANTUML_CMD=java -jar plantuml.jar
    )
)

if "%PLANTUML_CMD%"=="" (
    echo [ERROR] PlantUML not found!
    echo.
    echo Options:
    echo   1. Download plantuml.jar from https://github.com/plantuml/plantuml/releases
    echo   2. Or use online tool: https://www.plantuml.com/plantuml/uml/
    echo   3. See INSTALL_PLANTUML.md for detailed instructions
    echo.
    pause
    exit /b 1
)

REM Check if pdflatex is available
where pdflatex >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pdflatex not found. Please install MiKTeX first.
    echo        Run: winget install MiKTeX.MiKTeX
    pause
    exit /b 1
)

echo [1/4] Generating architecture diagram...
%PLANTUML_CMD% architecture_diagram.puml
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to generate architecture_diagram.png
)

echo [2/4] Generating sequence diagram...
%PLANTUML_CMD% sequence_diagram.puml
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to generate sequence_diagram.png
)

echo [3/4] Generating training pipeline diagram...
%PLANTUML_CMD% training_pipeline.puml
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to generate training_pipeline.png
)

echo [4/4] Generating recommendation flow diagram...
%PLANTUML_CMD% recommendation_flow.puml
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to generate recommendation_flow.png
)

echo.
echo [5/6] Compiling LaTeX (Pass 1 of 2)...
pdflatex -interaction=nonstopmode IEEE_Research_Paper.tex >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] First compilation had some issues. Continuing...
)

echo [6/6] Compiling LaTeX (Pass 2 of 2)...
pdflatex -interaction=nonstopmode IEEE_Research_Paper.tex >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Second compilation had some issues.
)

echo.
echo ========================================
echo   Done! Check IEEE_Research_Paper.pdf
echo ========================================
echo.

if exist IEEE_Research_Paper.pdf (
    echo [SUCCESS] PDF generated successfully!
    echo.
    echo Opening PDF...
    start IEEE_Research_Paper.pdf
) else (
    echo [ERROR] PDF was not created. Check the log files for errors.
)

pause

