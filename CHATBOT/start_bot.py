import os
import subprocess
from dotenv import load_dotenv

# Carica il file .env
load_dotenv()

# Controlla se le variabili sono state caricate
print("🔍 Controllo variabili d'ambiente:")
print("TELEGRAM_ACCESS_TOKEN:", os.getenv("TELEGRAM_ACCESS_TOKEN"))
print("TELEGRAM_VERIFY:", os.getenv("TELEGRAM_VERIFY"))
print("NGROK_URL:", os.getenv("NGROK_URL"))

# Avvia Rasa
print("\n🚀 Avvio del chatbot...")
subprocess.run(["rasa", "run", "--enable-api"])
