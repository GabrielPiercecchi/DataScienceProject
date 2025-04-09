# DataScienceProject
Progetto di Data Science

**Per il chatbot è necessario python 3.10**

Su Win 11:
- creare env: *python3.10 -m venv ./venv*
- accedere all'env: *sudo .\venv\Scripts\activate*
  - **Se non funziona apire CMD ed eseguire questo comando**: *.\\.venv\Scripts\activate.bat*
  - Altrimenti eseguire: *Set-ExecutionPolicy Bypass -Scope Process* e poi *.\\.venv\Scripts\Activate.ps1*
- installare rasa: 
  - *pip install uv* 
  - *uv pip install rasa* 
  - **Seguire la documentazione di rasa per ottenere la chiave per Rasa Pro**
    - Creato e formattato il *.env*:
      - *uv pip install rasa-plus --extra-index-url=https://europe-west3-python.pkg.dev/rasa-releases/rasa-plus-py/simple/*
  
Su Ubuntu\WSL:

Prima Esecuzione:
- python3.10 -m venv ./venv-wsl
- source .venv-wsl/bin/activate
- spacy download it_core_news_md
- ngrok http 5005
- rasa run actions
- rasa train 
- rasa run

Esecuzioni Successive:
- source .venv-wsl/bin/activate
- ngrok http 5005
- rasa run actions
- rasa train 
- rasa run
