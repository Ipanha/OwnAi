Project (LLM)
1. Collect Data in PDF 
2. Fine-Tuning
3. Get APi Key of Model : https://aistudio.google.com/
4. Set Up Virtual Environment: python -m venv venv

5. Install Libraries: With the virtual environment active, run this single command to install everything we need:
pip install fastapi "uvicorn[standard]" langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv

6. Create file .env : GOOGLE_API_KEY="PASTE_YOUR_API_KEY_HERE"
7. install and enable CORS in your backend: pip install fastapi[all]

8. Run Project : python -m uvicorn main:app --reload

