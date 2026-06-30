"""
Mark XLVII — Telegram Bot
Roda na nuvem do Railway. Converte o assistente JARVIS em um bot Telegram.
Backend de LLM: tenta Gemini primeiro, cai para Groq automaticamente em caso de erro.
"""

import json
import logging
import os
import asyncio
from pathlib import Path

import requests
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from memory.memory_manager import (
    load_memory,
    update_memory,
    format_memory_for_prompt,
)
from actions.web_search import web_search as web_search_action
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
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

GROQ_MODEL    = "openai/gpt-oss-120b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Sessões de conversa por usuário  {user_id: [messages]}  (formato OpenAI: role/content)
_sessions: dict[int, list[dict]] = {}
# Memória por usuário             {user_id: dict}
_memories: dict[int, dict] = {}


def _get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["gemini_api_key"]
    except Exception:
        return ""


def _get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["groq_api_key"]
    except Exception:
        raise RuntimeError(
            "GROQ_API_KEY não definida. "
            "Configure a env var GROQ_API_KEY no Railway."
        )


def _get_telegram_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
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


# ── Tool declarations (formato OpenAI — compatível com Groq e usado também p/ Gemini) ──
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Searches the web. Use for ANY question about current facts, events, prices, "
                "or topics — always prefer this over guessing. "
                "Modes: 'search' (default), 'news' (latest headlines on a topic), "
                "'research' (deep comprehensive answer), 'price' (product cost lookup), "
                "'compare' (side-by-side comparison of items, list them comma-separated in query)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string", "description": "Search query or topic"},
                    "mode":   {"type": "string", "description": "search | news | research | price | compare"},
                    "aspect": {"type": "string", "description": "Comparison aspect: price | specs | reviews | features"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_report",
            "description": "Gets the weather report for a city using web search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "time": {"type": "string", "description": "When: today, tomorrow, this week"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Saves important information about the user to long-term memory. "
                "Use this automatically when the user shares personal info, preferences, "
                "or important context. Categories: identity, preferences, projects, "
                "relationships, wishes, notes. "
                "Values must be in English regardless of the conversation language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "identity | preferences | projects | relationships | wishes | notes",
                    },
                    "key":   {"type": "string", "description": "Memory key (snake_case)"},
                    "value": {"type": "string", "description": "Value to remember (in English)"},
                },
                "required": ["category", "key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "Deletes a specific memory entry when the user asks to forget something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Memory category"},
                    "key":      {"type": "string", "description": "Key to forget"},
                },
                "required": ["category", "key"],
            },
        },
    },
]


def _openai_tools_to_gemini(tools: list) -> list:
    """Converte declarações de tool no formato OpenAI para o formato esperado pela API REST do Gemini."""
    fn_decls = []
    for t in tools:
        fn = t["function"]
        props = {}
        for pname, pdata in fn["parameters"]["properties"].items():
            ptype = pdata["type"].lower()
            if ptype in ("array", "object"):
                ptype = "string"
            props[pname] = {
                "type":        ptype.upper(),
                "description": pdata.get("description", ""),
            }
        fn_decls.append({
            "name":        fn["name"],
            "description": fn["description"],
            "parameters": {
                "type":       "OBJECT",
                "properties": props,
                "required":   fn["parameters"].get("required", []),
            },
        })
    return fn_decls


# ── Tool executor ─────────────────────────────────────────────────────────────
def _execute_tool(name: str, args: dict, user_id: int) -> str:
    """Executa uma tool e retorna o resultado como string."""
    try:
        if name == "web_search":
            return web_search_action(parameters=args)

        if name == "weather_report":
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


# ── Backend: Groq (formato OpenAI) ─────────────────────────────────────────────
def _run_groq(messages: list, user_id: int) -> str:
    """
    Roda o tool loop completo usando a Groq.
    `messages` já vem no formato OpenAI (incluindo system prompt).
    Levanta exceção se a chamada falhar.
    """
    api_key = _get_groq_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    msgs = list(messages)  # cópia local, não polui o histórico do chamador
    final_text = ""

    for _ in range(5):
        payload = {
            "model":       GROQ_MODEL,
            "messages":    msgs,
            "tools":       TOOLS_OPENAI,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens":  1024,
        }
        resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")

        data       = resp.json()
        msg        = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        msgs.append(msg)

        if not tool_calls:
            final_text = (msg.get("content") or "").strip()
            break

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            result = _execute_tool(fn_name, fn_args, user_id)
            logger.info(f"[Tool/Groq] {fn_name}({fn_args}) → {str(result)[:80]}…")
            msgs.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      str(result),
            })

    if not final_text:
        raise RuntimeError("Groq returned empty response after tool loop.")
    return final_text


# ── Backend: Gemini (REST API) ─────────────────────────────────────────────────
def _openai_messages_to_gemini_contents(messages: list) -> tuple[str, list]:
    """
    Separa o system prompt e converte o resto do histórico (formato OpenAI)
    para o formato 'contents' da API REST do Gemini.
    """
    system_text = ""
    contents = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_text = m.get("content", "")
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": m.get("content", "")}]})
        elif role == "assistant":
            content = m.get("content") or ""
            contents.append({"role": "model", "parts": [{"text": content}]})
        elif role == "tool":
            # Gemini espera functionResponse — aqui simplificamos como texto de usuário
            contents.append({
                "role": "user",
                "parts": [{"text": f"[Tool result] {m.get('content', '')}"}],
            })
    return system_text, contents


def _run_gemini(messages: list, user_id: int) -> str:
    """
    Roda o tool loop completo usando o Gemini (REST API direta, sem SDK).
    Levanta exceção se a chamada falhar (key inválida, API desativada, quota etc).
    """
    api_key = _get_gemini_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured.")

    system_text, base_contents = _openai_messages_to_gemini_contents(messages)
    gemini_tools = _openai_tools_to_gemini(TOOLS_OPENAI)

    contents = list(base_contents)
    final_text = ""

    for _ in range(5):
        payload = {
            "system_instruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "tools": [{"function_declarations": gemini_tools}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
        }
        resp = requests.post(
            f"{GEMINI_ENDPOINT}?key={api_key}",
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts     = []
        function_calls = []
        for p in parts:
            if "text" in p and p["text"]:
                text_parts.append(p["text"])
            if "functionCall" in p:
                function_calls.append(p["functionCall"])

        contents.append({"role": "model", "parts": parts})

        if not function_calls:
            final_text = " ".join(text_parts).strip()
            break

        function_response_parts = []
        for fc in function_calls:
            fn_name = fc.get("name", "")
            fn_args = fc.get("args", {}) or {}
            result  = _execute_tool(fn_name, fn_args, user_id)
            logger.info(f"[Tool/Gemini] {fn_name}({fn_args}) → {str(result)[:80]}…")
            function_response_parts.append({
                "functionResponse": {
                    "name": fn_name,
                    "response": {"result": str(result)},
                }
            })
        contents.append({"role": "user", "parts": function_response_parts})

    if not final_text:
        raise RuntimeError("Gemini returned empty response after tool loop.")
    return final_text


# ── Orquestrador: tenta Gemini, cai para Groq ──────────────────────────────────
def _call_llm(user_id: int, user_message: str) -> str:
    """
    Tenta Gemini primeiro. Se falhar por QUALQUER motivo (key inválida, API
    desativada, quota excedida, timeout, etc), cai automaticamente para Groq.
    Retorna o texto final já com a etiqueta do motor usado.
    """
    memory = _memories.get(user_id) or load_memory()
    _memories[user_id] = memory
    memory_text = format_memory_for_prompt(memory)

    base_prompt   = _load_system_prompt()
    system_prompt = base_prompt + ("\n\n" + memory_text if memory_text else "")

    history = _sessions.get(user_id, [])
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": system_prompt}] + history

    engine_used = None
    final_text  = ""

    # 1) Tenta Gemini
    try:
        final_text  = _run_gemini(messages, user_id)
        engine_used = "gemini"
    except Exception as e:
        logger.warning(f"[LLM] Gemini failed, falling back to Groq: {e}")

        # 2) Cai para Groq
        try:
            final_text  = _run_groq(messages, user_id)
            engine_used = "groq"
        except Exception as e2:
            logger.error(f"[LLM] Groq also failed: {e2}", exc_info=True)
            raise RuntimeError(f"Both Gemini and Groq failed. Last error: {e2}")

    # Salva histórico real da conversa (sem system prompt, últimas 20 mensagens)
    history.append({"role": "assistant", "content": final_text or "..."})
    _sessions[user_id] = history[-20:]

    tag = "🟦 Raciocínio feito usando Gemini" if engine_used == "gemini" else "🟧 Raciocínio feito usando Groq"
    return f"{final_text}\n\n_{tag}_"


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
        "You can also send me *files/documents* and ask me to analyze them.\n\n"
        "I try Gemini first for every reply, and automatically fall back to Groq "
        "if Gemini is unavailable. I'll always tell you which engine answered.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text    = (update.message.text or "").strip()

    if not text:
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        reply = await asyncio.get_event_loop().run_in_executor(
            None, _call_llm, user_id, text
        )
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        reply = f"I'm sorry, sir. An error occurred: {e}"

    if len(reply) > 4096:
        for i in range(0, len(reply), 4096):
            await update.message.reply_text(reply[i : i + 4096], parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)


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
        import tempfile
        file   = await doc.get_file()
        suffix = Path(doc.file_name or "file").suffix or ".bin"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        result = file_processor(
            parameters={
                "action":      "auto",
                "file_path":   tmp_path,
                "instruction": caption,
            }
        )

        Path(tmp_path).unlink(missing_ok=True)

        history = _sessions.get(user_id, [])
        history.append({"role": "user", "content": f"[File: {doc.file_name}]\n{caption}"})
        history.append({"role": "assistant", "content": result})
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

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("reset",  cmd_reset))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("help",   cmd_help))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Mark XLVII Telegram Bot starting (polling mode, Gemini→Groq fallback)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()