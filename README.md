# Mark XLVII — Bot Telegram

Versão do assistente JARVIS adaptada para o Telegram, rodando na nuvem do Railway.

## O que funciona no Telegram

| Feature | Status |
|---|---|
| Conversa com JARVIS (Gemini 2.5 Flash) | ✅ |
| Busca na web (Google via Gemini + DuckDuckGo) | ✅ |
| Previsão do tempo | ✅ |
| Análise de arquivos enviados | ✅ |
| Memória de longo prazo por usuário | ✅ |
| Histórico de conversa (últimas 20 msgs) | ✅ |
| Controle do computador / apps | ❌ (não faz sentido em cloud) |
| Captura de tela / câmera | ❌ (não faz sentido em cloud) |
| Áudio / voz | ❌ (Telegram bot é text-only aqui) |

---

## Deploy no Railway

### 1. Crie o bot no Telegram
1. Abra o [@BotFather](https://t.me/BotFather) no Telegram
2. `/newbot` → escolha um nome → escolha um username (ex: `meu_jarvis_bot`)
3. Copie o **token** que ele te dá

### 2. Pega a Gemini API Key
1. Acesse [aistudio.google.com](https://aistudio.google.com)
2. Crie uma API Key gratuita

### 3. Deploy no Railway
1. Faça upload desta pasta para um repositório GitHub
2. No [Railway](https://railway.app), crie um novo projeto → **Deploy from GitHub repo**
3. Configure as **variáveis de ambiente** (Settings → Variables):

```
TELEGRAM_BOT_TOKEN = <token do BotFather>
GEMINI_API_KEY     = <sua chave do Google AI Studio>
WEBHOOK_URL        = https://<seu-projeto>.railway.app
```

> A variável `WEBHOOK_URL` deve ser a URL pública do seu serviço no Railway.
> Você encontra ela em **Settings → Domains** do seu serviço.

4. O Railway detecta automaticamente o `Procfile` e inicia `python bot.py`

---

## Teste local

```bash
pip install -r requirements.txt

# Crie o arquivo de configuração
mkdir -p config
echo '{
  "gemini_api_key": "SUA_CHAVE_AQUI",
  "telegram_bot_token": "SEU_TOKEN_AQUI"
}' > config/api_keys.json

# Rode em modo polling (sem WEBHOOK_URL)
python bot.py
```

---

## Comandos do bot

| Comando | Descrição |
|---|---|
| `/start` | Apresentação do JARVIS |
| `/reset` | Limpa o histórico de conversa |
| `/memory` | Mostra o que o JARVIS lembra de você |
| `/help` | Ajuda |

Envie qualquer mensagem de texto e ele responde normalmente.  
Envie um arquivo/documento com uma legenda e ele analisa.

---

## Estrutura do projeto

```
mark-xlvii-telegram/
├── bot.py                  ← ponto de entrada principal
├── requirements.txt
├── Procfile
├── railway.json
├── core/
│   ├── prompt.txt          ← personalidade do JARVIS
│   └── llm_client.py
├── memory/
│   ├── memory_manager.py   ← memória de longo prazo
│   └── long_term.json      ← criado automaticamente
├── actions/
│   ├── web_search.py       ← busca na web
│   ├── file_processor.py   ← análise de arquivos
│   └── ...
└── config/
    └── api_keys.json       ← opcional, use env vars no Railway
```
