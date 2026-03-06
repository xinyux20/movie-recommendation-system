@echo off

echo Setting up environment...

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate

pip install -r requirements.txt

echo Running app...

streamlit run app.py

pause
