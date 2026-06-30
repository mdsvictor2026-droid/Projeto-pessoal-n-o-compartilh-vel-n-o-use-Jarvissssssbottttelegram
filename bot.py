"""
Mark XLVII — Telegram Bot
Roda na nuvem do Railway. Converte o assistente JARVIS em um bot Telegram.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from telegram import Update, BotCommand
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from google import genai
from google.genai import types

from memory.memory_manager import (
    load_memory,
    update_memory,
    format_memory_for_prompt,
)
from actions.web_search import web_search as web_search_action
from actions.weather_report import weather_action
from actions.file_processor import file_processor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mark47-telegram")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH = BASE_DIR / "core" / "prompt.txt"

# ── Config ────────────────────────────────────────────────────────────────────
TEXT_MODEL = "gemini-2.5-flash"

# Sessões de conversa por usuário  {user_id: [messages]}
_sessions: dict[int, list[dict]] = {}
# Memória por usuário             {user_id: dict}
_memories: dict[int, dict] = {}


def _get_api_key() -> str:
    """Pega a chave da API do Gemini — env var tem prioridade sobre config file."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["gemini_api_key"]
    except Exception:
        raise RuntimeError(
            "GEMINI_API_KEY não definida. "
            "Configure a env var GEMINI_API_KEY no Railway."
        )


def _get_telegram_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        try:
            token = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get(
                "telegram_bot_token", ""
            )
        except Exception:
            pass
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN não definida. "
            "Configure a env var TELEGRAM_BOT_TOKEN no Railway."
        )
    return token


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool. "
            "You are running as a Telegram bot, so you cannot control the computer, "
            "open apps, or access the screen. Focus on: web search, weather, "
            "file analysis, answering questions, and memory. "
            "Always call sir to user."
        )


# ── Tool declarations (apenas as que fazem sentido em cloud/Telegram) ─────────
TOOL_DECLARATIONS = [
    {
        "name": "web_search",
        "description": (
            "Searches the web. Use for ANY question about current facts, events, prices, "
            "or topics — always prefer this over guessing. "
            "Modes: 'search' (default), 'news' (latest headlines on a topic), "
            "'research' (deep comprehensive answer), 'price' (product cost lookup), "
            "'compare' (side-by-side comparison of items)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query or topic. For compare, list items separated by comma."},
                "mode":   {"type": "STRING", "description": "search | news | research | price | compare"},
                "aspect": {"type": "STRING", "description": "Comparison aspect: price | specs | reviews | features"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "weather_report",
        "description": "Gets the weather report for a city using web search.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"},
                "time": {"type": "STRING", "description": "When: today, tomorrow, this week"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "save_memory",
        "description": (
            "Saves important information about the user to long-term memory. "
            "Use this automatically when the user shares personal info, preferences, "
            "or important context. Categories: identity, preferences, projects, "
            "relationships, wishes, notes. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": "identity | preferences | projects | relationships | wishes | notes",
                },
                "key":   {"type": "STRING", "description": "Memory key (snake_case)"},
                "value": {"type": "STRING", "description": "Value to remember (in English)"},
            },
            "required": ["category", "key", "value"],
        },
    },
    {
        "name": "forget_memory",
        "description": "Deletes a specific memory entry when the user asks to forget something.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "description": "Memory category"},
                "key":      {"type": "STRING", "description": "Key to forget"},
            },
            "required": ["category", "key"],
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────
def _execute_tool(name: str, args: dict, user_id: int) -> str:
    """Executa uma tool e retorna o resultado como string."""
    try:
        if name == "web_search":
            return web_search_action(parameters=args)

        if name == "weather_report":
            # weather_action abre browser — aqui usamos web_search no lugar
            city = args.get("city", "")
            when = args.get("time", "today")
            return web_search_action(
                parameters={"query": f"weather in {city} {when}", "mode": "search"}
            )

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if not key or not value:
                return "Memory key or value missing."
            update_memory({category: {key: {"value": value}}})
            _memories[user_id] = load_memory()
            return f"Remembered: {category}/{key} = {value}"

        if name == "forget_memory":
            from memory.memory_manager import forget_memory
            category = args.get("category", "notes")
            key      = args.get("key", "")
            result   = forget_memory(key, category)
            _memories[user_id] = load_memory()
            return result

        return f"Tool '{name}' not available in Telegram mode."

    except Exception as e:
        logger.error(f"Tool {name} error: {e}", exc_info=True)
        return f"Tool {name} failed: {e}"


# ── Gemini call com tool loop ──────────────────────────────────────────────────
def _call_gemini(user_id: int, user_message: str) -> str:
    """
    Chama o Gemini com histórico da conversa + memória + tools.
    Retorna a resposta final em texto.
    """
    api_key = _get_api_key()
    client  = genai.Client(api_key=api_key)

    # Carrega memória do usuário
    memory = _memories.get(user_id) or load_memory()
    _memories[user_id] = memory
    memory_text = format_memory_for_prompt(memory)

    # Monta system prompt
    base_prompt = _load_system_prompt()
    system_prompt = base_prompt
    if memory_text:
        system_prompt = base_prompt + "\n\n" + memory_text

    # Histórico de conversa do usuário
    history = _sessions.get(user_id, [])
    history.append({"role": "user", "parts": [{"text": user_message}]})

    # Converte tools para o formato do Gemini
    gemini_tools = []
    for td in TOOL_DECLARATIONS:
        props = {}
        for pname, pdata in td["parameters"]["properties"].items():
            ptype = pdata["type"].lower()
            # Gemini só aceita: string, number, integer, boolean, object
            if ptype in ("array", "object"):
                ptype = "string"
            props[pname] = {
                "type":        ptype,
                "description": pdata.get("description", ""),
            }
        gemini_tools.append({
            "name":        td["name"],
            "description": td["description"],
            "parameters": {
                "type":       "object",
                "properties": props,
                "required":   td["parameters"].get("required", []),
            },
        })

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(function_declarations=gemini_tools)],
        temperature=0.7,
    )

    # Tool loop (máximo 5 rodadas)
    contents = [
        types.Content(
            role=turn["role"],
            parts=[types.Part(text=p["text"]) for p in turn["parts"] if "text" in p],
        )
        for turn in history
    ]

    final_text = ""
    for _ in range(5):
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        parts      = candidate.content.parts

        # Coleta texto e tool calls
        text_parts     = []
        function_calls = []

        for part in parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)

        # Adiciona resposta do modelo ao histórico
        contents.append(types.Content(role="model", parts=parts))

        if not function_calls:
            # Sem tool calls — resposta final
            final_text = " ".join(text_parts).strip()
            break

        # Executa tools e injeta resultados
        tool_result_parts = []
        for fc in function_calls:
            args   = dict(fc.args) if fc.args else {}
            result = _execute_tool(fc.name, args, user_id)
            logger.info(f"[Tool] {fc.name}({args}) → {result[:80]}…")
            tool_result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        contents.append(
            types.Content(role="user", parts=tool_result_parts)
        )

    # Salva histórico (últimas 20 mensagens)
    model_reply_parts = [{"text": final_text}] if final_text else [{"text": "..."}]
    history.append({"role": "model", "parts": model_reply_parts})
    _sessions[user_id] = history[-20:]

    return final_text or "I'm sorry, sir. I could not generate a response."


# ── Handlers do Telegram ───────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    name = user.first_name or "Sir"
    await update.message.reply_text(
        f"Welcome, {name}. I am JARVIS, at your service.\n\n"
        "You can talk to me normally. I can:\n"
        "• 🔍 Search the web\n"
        "• 🌤 Check the weather\n"
        "• 📁 Analyze files (send a document)\n"
        "• 🧠 Remember things about you\n\n"
        "Use /reset to clear our conversation history.\n"
        "Use /memory to see what I remember about you.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    _sessions.pop(user_id, None)
    await update.message.reply_text(
        "Conversation history cleared, sir. Starting fresh."
    )


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    memory  = _memories.get(user_id) or load_memory()
    text    = format_memory_for_prompt(memory)
    if not text:
        await update.message.reply_text("I have no stored memories about you yet, sir.")
    else:
        # Remove o header de instrução interna
        lines = text.split("\n")
        clean = "\n".join(lines[1:]) if lines[0].startswith("[WHAT") else text
        await update.message.reply_text(
            f"📋 *What I know about you:*\n\n{clean.strip()}",
            parse_mode=ParseMode.MARKDOWN,
        )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*Mark XLVII — JARVIS Telegram Bot*\n\n"
        "Just talk to me naturally. Commands:\n"
        "/start — Introduction\n"
        "/reset — Clear conversation history\n"
        "/memory — Show what I remember about you\n"
        "/help — This message\n\n"
        "You can also send me *files/documents* and ask me to analyze them.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text    = (update.message.text or "").strip()

    if not text:
        return

    # Mostra "digitando..."
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        reply = await asyncio.get_event_loop().run_in_executor(
            None, _call_gemini, user_id, text
        )
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        reply = f"I'm sorry, sir. An error occurred: {e}"

    # Telegram tem limite de 4096 chars por mensagem
    if len(reply) > 4096:
        for i in range(0, len(reply), 4096):
            await update.message.reply_text(reply[i : i + 4096])
    else:
        await update.message.reply_text(reply)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Recebe arquivos e processa com file_processor."""
    user_id = update.effective_user.id
    doc     = update.message.document
    caption = (update.message.caption or "Analyze this file.").strip()

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        # Baixa o arquivo para um temp dir
        import tempfile
        file = await doc.get_file()
        suffix = Path(doc.file_name or "file").suffix or ".bin"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        # Chama o file_processor
        result = file_processor(
            parameters={
                "action":      "auto",
                "file_path":   tmp_path,
                "instruction": caption,
            }
        )

        # Limpa o arquivo temp
        Path(tmp_path).unlink(missing_ok=True)

        # Injeta resultado no histórico como se o JARVIS tivesse visto
        history = _sessions.get(user_id, [])
        history.append({
            "role":  "user",
            "parts": [{"text": f"[File: {doc.file_name}]\n{caption}"}],
        })
        history.append({
            "role":  "model",
            "parts": [{"text": result}],
        })
        _sessions[user_id] = history[-20:]

        reply = result

    except Exception as e:
        logger.error(f"Document error: {e}", exc_info=True)
        reply = f"Sorry sir, I couldn't process the file: {e}"

    if len(reply) > 4096:
        for i in range(0, len(reply), 4096):
            await update.message.reply_text(reply[i : i + 4096])
    else:
        await update.message.reply_text(reply)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    token = _get_telegram_token()

    app = Application.builder().token(token).build()

    # Comandos
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("reset",  cmd_reset))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("help",   cmd_help))

    # Mensagens de texto
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Arquivos / documentos
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Mark XLVII Telegram Bot starting (polling mode)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()