python -m venv venv
.\venv\Scripts\activate
pip install
pip install -r requirements.txt
python entrenar.py
uvicorn main:app --reload