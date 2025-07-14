- [DataScienceProject](#datascienceproject)
  - [BERT](#bert)
    - [Code](#code)
    - [PDF](#pdf)
  - [CHATBOT](#chatbot)
    - [Quick startup guide](#quick-startup-guide)
      - [0. Clone the repository](#0-clone-the-repository)
      - [1. Create and access virtual environment](#1-create-and-access-virtual-environment)
      - [2. Install dependencies](#2-install-dependencies)
      - [3. Train the model](#3-train-the-model)
      - [4. Start the action server](#4-start-the-action-server)
      - [5. Start the chatbot](#5-start-the-chatbot)
    - [PDF](#pdf-1)
  - [NLP](#nlp)
    - [Code](#code-1)
    - [PDF](#pdf-2)
  - [SNA](#sna)
    - [PDF](#pdf-3)
  - [STCC](#stcc)
    - [PDF](#pdf-4)


# DataScienceProject

## BERT

### Code

https://colab.research.google.com/drive/14UPvDfX4DBBMDb52A87pTw5hW5kMngMt?usp=sharing

### PDF

[Bert_DS.pdf](BERT/Bert_DS.pdf)

## CHATBOT

### Quick startup guide

> **Note:** This project uses Python 3.10.0. Rasa does not support Python 3.11 yet.

#### 0. Clone the repository
```bash
git clone https://github.com/GabrielPiercecchi/DataScienceProject.git
cd CHATBOT
```

#### 1. Create and access virtual environment
```bash
python3.10 -m venv ./venv
source venv/bin/activate
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Or:
```bash
pip3.10 install -r requirements.txt
```

Then create a `.env` file with:

```bash
RASA_PRO_LICENSE=<your_rasa_pro_license>
TELEGRAM_ACCESS_TOKEN=<your_telegram_access_token>
TELEGRAM_VERIFY=<your_telegram_verify>
NGROK_URL=<your_ngrok_url>/webhooks/telegram/webhook
```

Finally:
```bash
spacy download it_core_news_md
```
> **Note:** The `it_core_news_md` model is used for the Italian language. 

#### 3. Train the model
```bash
rasa train
```

#### 4. Start the action server
```bash
rasa run actions
```

#### 5. Start the chatbot

In a different terminal:

```bash
rasa shell
```

### PDF

[Chatbot_DS.pdf](CHATBOT/Chatbot_DS.pdf)

## NLP

### Code

https://colab.research.google.com/drive/14UPvDfX4DBBMDb52A87pTw5hW5kMngMt?usp=sharing

### PDF

[NLP_DS.pdf](NLP/NLP_DS.pdf)

## SNA

### PDF

[Social_Network_Analysis_DS.pdf](SNA/Social_Network_Analysis_DS.pdf)

## STCC

### PDF

[Serie_temporali_DS.pdf](STCC/Serie_temporali_DS.pdf)