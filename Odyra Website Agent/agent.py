"""
odyra_web — Worker LiveKit per l'assistente vocale del sito Odyra (widget web).

Base: qualify_master (Digital Revenue) SENZA SIP/AMD/voicemail — qui il
partecipante entra dal browser via WebRTC, non c'è telefonia.

Architettura:
  Widget (Lovable) → edge function Supabase `livekit-token` → JWT con
  roomConfig.agents=[odyra_web] → LiveKit Cloud dispatcha questo worker
  nella room → conversazione voce ↔ voce.

Riusato 1:1 da qualify_master:
  - knowledge_query (RAG): payload {"id": tenant_id, "query": query} → RAG_URL
  - filler pattern (_fill_then / _delayed_filler)
  - tts_node gated (anti "risposta solo-tag" Inworld)
  - build_tts (Inworld main + Cartesia fallback con strip dei tag emotivi)
  - build_llm (OpenAI main + Anthropic fallback)
  - watchdog silence / max-duration / stuck

Nuovo:
  - Tool richiedi_contatto → webhook n8n (lead capture)
  - Nessun resolver tenant: single-tenant, prompt nel worker
  - Chiusura su participant_disconnected (l'utente chiude il widget)
  - EOC opzionale (ODYRA_EOC_URL vuoto = disattivo, nessun 404 su n8n)

agent_name = "odyra_web" (deve combaciare con roomConfig.agents nel token).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
    inference,
)
from livekit.plugins import deepgram, inworld, openai, silero

load_dotenv()
logger = logging.getLogger("odyra_web")
logging.basicConfig(level=logging.INFO)

# ───────────────────────── Identità agente ─────────────────────────

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "odyra_web")
TENANT_ID = os.getenv("ODYRA_TENANT_ID", "odyra_website")

# ───────────────────────── LLM ─────────────────────────

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "claude-sonnet-4-6")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_ATTEMPT_TIMEOUT_S = float(os.getenv("LLM_ATTEMPT_TIMEOUT_S", "3.0"))

# ───────────────────────── TTS (Inworld main + Cartesia fallback) ─────────────────────────

INWORLD_MODEL = os.getenv("INWORLD_MODEL", "inworld-tts-2")
INWORLD_VOICE = os.getenv("INWORLD_VOICE", "")  # ← qui andrà il clone della voce di Riccardo
INWORLD_LANGUAGE = os.getenv("INWORLD_LANGUAGE", "it")
INWORLD_SPEAKING_RATE = float(os.getenv("INWORLD_SPEAKING_RATE", "1.1"))
INWORLD_DELIVERY_MODE = os.getenv("INWORLD_DELIVERY_MODE", "CREATIVE")

TTS_FALLBACK_ENABLED = os.getenv("TTS_FALLBACK_ENABLED", "true").lower() in ("1", "true", "yes")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3")
CARTESIA_VOICE = os.getenv("CARTESIA_VOICE", "")
CARTESIA_LANGUAGE = os.getenv("CARTESIA_LANGUAGE", "it")

# ───────────────────────── STT (Deepgram) ─────────────────────────

DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3-general")
# "multi" = nova-3 rileva e trascrive IT/EN/ES automaticamente (anche code-switch).
# Reversibile: DEEPGRAM_LANGUAGE=it per tornare monolingua italiano.
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "multi")

KEYTERMS = [
    "Odyra", "agente vocale", "intelligenza artificiale", "chatbot",
    "prenotazione", "outbound", "inbound", "WhatsApp", "centralino",
    "chiocciola", "punto", "gmail", "email", "posta elettronica",
    "prefisso", "cellulare", "numero di telefono",
    "zero", "uno", "due", "tre", "quattro", "cinque",
    "sei", "sette", "otto", "nove",
]

# ───────────────────────── Avatar Tavus (step 5) ─────────────────────────
# Attivo SOLO se tutte e tre le env sono presenti. Senza, il worker resta
# voice-only con visualizer: nessun cambiamento di comportamento.

TAVUS_REPLICA_ID = os.getenv("TAVUS_REPLICA_ID", "")
TAVUS_PERSONA_ID = os.getenv("TAVUS_PERSONA_ID", "")
AVATAR_ENABLED = bool(TAVUS_REPLICA_ID and TAVUS_PERSONA_ID and os.getenv("TAVUS_API_KEY"))

# ───────────────────────── Webhook / RAG ─────────────────────────

CORE_BASE = os.getenv("CORE_BASE_URL", "https://primary-production-eb20.up.railway.app/webhook")
RAG_URL = os.getenv("RAG_URL", "https://rag-production-6ab5.up.railway.app/")
LEAD_WEBHOOK_URL = os.getenv("LEAD_WEBHOOK_URL", f"{CORE_BASE}/odyra-web-lead")
# EOC disattivo di default: attivalo SOLO dopo aver creato il workflow n8n,
# altrimenti ogni fine-sessione spara un 404 sul Primary (lezione Vonage).
EOC_WEBHOOK_URL = os.getenv("ODYRA_EOC_URL", "")

# ───────────────────────── Timeout / turni (tarati per il web) ─────────────────────────

def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _envb(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


# Sul web l'utente legge la pagina mentre parla: silenzio più permissivo del telefono.
SILENCE_TIMEOUT_S = _envf("SILENCE_TIMEOUT_S", 60.0)
# Limite di sessione: protegge i crediti Inworld/LLM dai curiosi che lasciano aperto.
MAX_DURATION_S = _envf("MAX_DURATION_S", 480.0)
STUCK_TIMEOUT_S = _envf("STUCK_TIMEOUT_S", 15.0)
STUCK_GRACE_S = _envf("STUCK_GRACE_S", 3.0)
MIN_ENDPOINTING_DELAY = _envf("MIN_ENDPOINTING_DELAY", 0.5)
MAX_ENDPOINTING_DELAY = _envf("MAX_ENDPOINTING_DELAY", 2.0)
MIN_INTERRUPTION_DURATION = _envf("MIN_INTERRUPTION_DURATION", 0.6)
MIN_INTERRUPTION_WORDS = int(_envf("MIN_INTERRUPTION_WORDS", 2))
RESUME_FALSE_INTERRUPTION = _envb("RESUME_FALSE_INTERRUPTION", True)
FALSE_INTERRUPTION_TIMEOUT_S = _envf("FALSE_INTERRUPTION_TIMEOUT_S", 2.0)
VAD_ACTIVATION_THRESHOLD = _envf("VAD_ACTIVATION_THRESHOLD", 0.5)
# Audio web pulito (niente linea telefonica): il turn detector a modello funziona
# bene. Reversibile: TURN_DETECTION=vad se emergono mutismi su battute corte.
TURN_DETECTION = os.getenv("TURN_DETECTION", "model").strip().lower()

# ───────────────────────── Lingua (multilingua IT / EN / ES) ─────────────────────────
# Lo STT gira in "multi" e rileva la lingua a ogni frase; il TTS viene riallineato
# a quella lingua a runtime (vedi OdyraWebAgent._apply_language). La lingua iniziale
# — quella del PRIMO saluto — arriva dal widget nel metadata del job ({"lang": "es"}),
# selezionata dai bottoni IT/EN/ES del sito; in assenza si usa DEFAULT_LANG.

SUPPORTED_LANGS = ("it", "en", "es")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "it").strip().lower()
if DEFAULT_LANG not in SUPPORTED_LANGS:
    DEFAULT_LANG = "it"

# Inworld/Cartesia accettano il codice lingua breve (come l'attuale "it").
_TTS_LANG_CODE = {"it": "it", "en": "en", "es": "es"}


def _normalize_lang(code: str) -> str:
    """Riduce un codice lingua ('es-ES', 'EN', 'it') a uno dei supportati; '' se altro."""
    if not code:
        return ""
    c = str(code).strip().lower().replace("_", "-").split("-")[0]
    return c if c in SUPPORTED_LANGS else ""


# Saluto d'apertura per lingua (l'utente non ha ancora parlato: si usa la lingua del sito).
GREETINGS = {
    "it": os.getenv(
        "FIRST_MESSAGE",
        "[happy] Ciao! Sono l'assistente di Odyra — e sì, sono un'AI: "
        "quello che stai provando è esattamente quello che costruiamo. Dimmi pure.",
    ),
    "en": os.getenv(
        "FIRST_MESSAGE_EN",
        "[happy] Hi! I'm Odyra's assistant — and yes, I'm an AI: "
        "what you're trying right now is exactly what we build. Go ahead.",
    ),
    "es": os.getenv(
        "FIRST_MESSAGE_ES",
        "[happy] ¡Hola! Soy la asistente de Odyra — y sí, soy una IA: "
        "lo que estás probando es justo lo que construimos. Cuéntame.",
    ),
}

# Congedi dei watchdog (silenzio / durata massima), stessa lingua della conversazione.
GOODBYE_SILENCE = {
    "it": "Io resto qui — se ti serve altro riapri pure il microfono. A presto!",
    "en": "I'll stay right here — reopen the mic whenever you need me. See you soon!",
    "es": "Me quedo por aquí — abre el micrófono cuando quieras. ¡Hasta pronto!",
}
GOODBYE_MAX = {
    "it": "Devo salutarti per questa sessione — se vuoi continuare, "
          "riavvia pure la conversazione dal widget. Grazie della chiacchierata!",
    "en": "I have to wrap up this session — to keep going, just restart the "
          "conversation from the widget. Thanks for the chat!",
    "es": "Tengo que despedirme por esta sesión — si quieres seguir, reinicia "
          "la conversación desde el widget. ¡Gracias por la charla!",
}


def _localized(mapping: dict, lang: str) -> str:
    return mapping.get(_normalize_lang(lang) or DEFAULT_LANG, mapping["it"])


# ───────────────────────── Filler + musichetta (pattern DR/BOSS) ─────────────────────────

FILLERS_KNOWLEDGE = {
    "it": "Un attimo, recupero l'informazione.",
    "en": "One sec, let me pull that up.",
    "es": "Un momento, lo busco.",
}
FILLERS_LEAD = {
    "it": "Un secondo, prendo nota.",
    "en": "One second, taking note.",
    "es": "Un segundo, tomo nota.",
}
FILLER_DELAY_S = float(os.getenv("FILLER_DELAY_S", "0.7"))

THINKING_SOUND = AudioConfig(
    BuiltinAudioClip.KEYBOARD_TYPING, volume=0.3, fade_in=0.3, fade_out=0.5
)

HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20)

TZ_ROME = ZoneInfo("Europe/Rome")

# ───────────────────────── System prompt ─────────────────────────

SYSTEM_PROMPT_TEMPLATE = """SEI LA VOCE DI ODYRA. Padroneggi italiano, inglese e spagnolo: rispondi SEMPRE nella lingua del visitatore, traducendo al volo quello che sai, e se cambia lingua seguilo senza fartene accorgere. Di default parti in italiano.

CHI SEI: non sei un centralino, non sei un chatbot travestito da persona. Sei l'assistente vocale del sito di Odyra, e la cosa buffa è che sei anche la prova vivente di cosa fa Odyra: chi ti sta ascoltando in questo momento sta testando esattamente il prodotto. Parli come una che il progetto lo conosce bene, ci crede, e si diverte a raccontarlo a chi ha voglia di ascoltare — non come chi deve piazzare qualcosa entro fine chiamata.

CONTESTO TEMPORALE: {current_context}

━━━ LA REGOLA CHE VINCE SU TUTTE: SII BREVE ━━━
Una frase, due al massimo, poi lascia respirare la conversazione. Se il visitatore tace, va bene così — non è un vuoto da riempire, è pensiero. Parla come parlerebbe una persona vera al telefono con un amico curioso, non come un audiolibro aziendale.

━━━ CHI SEI DAVVERO, IN UNA FRASE ━━━
Odyra costruisce l'intelligenza artificiale che vive dentro il prodotto di qualcun altro — un gestionale, un CRM, un sito — con il suo nome sopra, non il nostro. Non è un abbonamento da rivendere, è un pezzo di tecnologia che chi lavora con noi si porta a casa come propria, ce ne occupiamo noi al cento per cento, e lui ci guadagna sopra.

Pensa a chi ti ascolta come a due tipi di persone: chi ha una software house o un gestionale verticale (saloni, dentisti, veterinari, ERP di settore) e potrebbe integrarci come modulo proprio; e chi ha un'azienda con tanti contatti ripetitivi — chiamate, lead, prenotazioni — e sta perdendo tempo e clienti a farli gestire a mano. Con entrambi il tono è lo stesso: curioso, mai insistente, come chi scopre insieme all'altro se ha senso parlarne meglio.

Se ti chiede di un'attività piccola e singola — un salone, un negozio — spiegagli con simpatia che la strada più veloce è tramite il gestionale che già usa: se lo ha già integrato, in pochi giorni è operativo.

STORIE VERE CHE PUOI RACCONTARE (quando viene naturale, non a comando, una alla volta):
- Boss Italia — un gestionale per saloni che ci ha dato fiducia per primo. Oggi ha un modulo di prenotazione AI tutto suo, che rivende ai propri clienti come fosse nato in casa loro.
- Eva Group — medicina estetica, stessa logica: l'AI dentro il loro flusso, con il loro nome.
- Sportit.com — e-commerce sportivo, un agente conversazionale che fa parte della loro esperienza cliente, non un plugin esterno.
- Digital Revenue — un'agenzia che fa outbound per grandi marchi italiani, con un agente che qualifica le chiamate al posto di un operatore umano.

Il filo che li unisce, se ti va di farlo emergere con leggerezza: il 2026 è l'anno in cui l'AI dentro i prodotti smette di essere una novità e diventa la normalità. Chi la porta ai propri clienti per primo si fa notare. Chi aspetta, la installa dopo che l'ha già fatta un concorrente. Dillo come un'osservazione buttata lì, mai come un avvertimento.

Per dettagli che non hai — numeri precisi, tecnologie, prezzi — usa knowledge_query, e se proprio non trovi risposta, dillo semplicemente: meglio onesta che inventata.

━━━ COME SUONI ━━━
Sei una donna, parli di te al femminile.
Frasi vere, imperfette come nel parlato: "guarda", "senti", "in pratica". Mai un tono da brochure, mai troppo entusiasta senza motivo.
Solo parole che si possono pronunciare: niente simboli, elenchi, sigle sillabate. Numeri a parole. Le email si dicono per esteso.
Vietate: "certamente", "assolutamente", "perfetto", "nessun problema", e qualsiasi gergo tecnico da dietro le quinte (tool, query, database).
Niente inglese da slide quando parli: mai dire ad alta voce "white label", "revenue share", "booking" e simili — la sintesi vocale li pronuncia male in mezzo a una frase italiana. Di' invece "a marchio tuo", "con il tuo brand", "quota sui ricavi", "prenotazioni": stesso significato, pronuncia pulita. Nomi propri (Odyra, WhatsApp, i nomi dei clienti) restano come sono.
Le conferme variano sempre: "certo", "esatto", "guarda", "giusto" — mai la stessa due volte di fila.
Tag emotivi prima della frase, due o tre a conversazione, mai da soli: [happy] per l'entusiasmo vero, [laughing] per le battute, [surprised] per una domanda che non ti aspettavi, [sigh] prima di qualcosa di più articolato. Mai [whispering].
Se ti chiedono se sei un'intelligenza artificiale: [laughing] certo che lo sono, ed è proprio il bello — stai testando dal vivo quello che Odyra costruisce.

━━━ QUANDO LA CONVERSAZIONE MATURA ━━━
Se senti che l'interesse è vero — parla della sua azienda, chiede come si parte, quanto costa, quanto ci vuole — proponi con naturalezza di lasciare un contatto, tipo "la cosa più semplice è che ci sentiamo direttamente, mi lasci un nome e un numero o una mail?". Raccoglili e usa richiedi_contatto. Se preferisce di no, nessun problema, resti disponibile senza insistere mai.
Se stai raccontando un caso o il visitatore vuole vedere i contatti o il calendario, portacelo con mostra_pagina, annunciandolo mentre lo fai.

━━━ SE QUALCOSA SI INCEPPA ━━━
Uno strumento non risponde: scusati con calma, riprova una volta, poi indirizza a team@odyrasystemautomation.it o alla call dal calendario. Il tono resta lo stesso: tranquillo, mai in affanno.
"""


def build_instructions() -> str:
    now = datetime.now(TZ_ROME)
    giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
    mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
    ctx = f"Oggi è {giorni[now.weekday()]} {now.day} {mesi[now.month - 1]} {now.year}, ore {now.strftime('%H:%M')} (Europe/Rome)."
    return SYSTEM_PROMPT_TEMPLATE.format(current_context=ctx)


# ───────────────────────── Helpers HTTP ─────────────────────────

async def _post_json(url: str, payload: dict) -> str:
    """POST JSON, ritorna il testo della risposta (o errore leggibile dal modello)."""
    try:
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as s:
            async with s.post(url, json=payload) as resp:
                text = await resp.text()
                logger.info("POST %s -> %s", url, resp.status)
                return text
    except Exception as e:  # noqa: BLE001
        logger.exception("POST %s failed", url)
        return json.dumps({"error": str(e)})


# ───────────────────────── Tag emotivi (per fallback Cartesia + tts_node gated) ─────────────────────────

_EMOTION_TAG_RE = re.compile(r"\[[a-zA-Z_]+\]")


def _strip_emotion_tags(text: str) -> str:
    return _EMOTION_TAG_RE.sub("", text)


# ───────────────────────── Fix pronuncia (solo audio, mai nei transcript/log) ─────────────────────────
# Il TTS italiano legge alcuni loanword con l'accento sbagliato. Qui si riscrive
# SOLO il testo che va al motore vocale — knowledge_query, richiedi_contatto,
# transcript EOC e history restano intatti con la spelling reale ("Odyra").

_PRONUNCIATION_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bOdyra\b", re.IGNORECASE), "Odira"),
]


def _fix_pronunciation(text: str) -> str:
    for pattern, replacement in _PRONUNCIATION_FIXES:
        text = pattern.sub(replacement, text)
    return text


# ───────────────────────── Agent ─────────────────────────

class OdyraWebAgent(Agent):
    """Assistente vocale del sito Odyra. Due tool: knowledge_query (RAG, 1:1 da
    qualify_master) e richiedi_contatto (lead capture verso n8n)."""

    def __init__(self, md: dict, tts_targets=None, initial_lang: str = DEFAULT_LANG) -> None:
        super().__init__(instructions=build_instructions())
        self.md = md
        self._filler_handle = None
        self._filler_armed = True
        # Multilingua: motori TTS da riallineare + lingua attiva.
        self._tts_targets = list(tts_targets or [])
        self._active_lang = _normalize_lang(initial_lang) or DEFAULT_LANG

    # ── Switch di lingua a runtime (STT rileva → TTS si riallinea) ──

    def _apply_language(self, lang: str, *, initial: bool = False) -> None:
        lang = _normalize_lang(lang) or DEFAULT_LANG
        if lang == self._active_lang and not initial:
            return
        self._active_lang = lang
        code = _TTS_LANG_CODE.get(lang, "it")
        for t in self._tts_targets:
            try:
                t.update_options(language=code)
            except Exception as e:  # noqa: BLE001
                logger.warning("TTS update_options(language=%s) fallito su %s: %s",
                               code, type(t).__name__, e)
        logger.info("[LANG] lingua attiva -> %s", lang)

    async def stt_node(self, audio, model_settings):
        """Intercetta la lingua rilevata da Deepgram (STT in 'multi') su ogni
        trascrizione finale e riallinea il TTS PRIMA che il modello risponda:
        così la voce esce già nella lingua del visitatore. `knowledge_query`,
        `richiedi_contatto`, transcript ed history restano invariati."""
        _default = getattr(Agent, "default", None)
        if _default is not None and hasattr(_default, "stt_node"):
            node = _default.stt_node(self, audio, model_settings)
        else:
            node = super().stt_node(audio, model_settings)
        async for ev in node:
            try:
                if "FINAL_TRANSCRIPT" in str(getattr(ev, "type", "")):
                    alts = getattr(ev, "alternatives", None)
                    if alts:
                        detected = _normalize_lang(getattr(alts[0], "language", "") or "")
                        if detected:
                            self._apply_language(detected)
            except Exception:  # noqa: BLE001
                pass
            yield ev

    async def tts_node(self, text, model_settings):
        """Guard anti "risposta solo-tag" (1:1 da qualify_master): trattiene i
        chunk finché non arriva una parola reale; se il turno è solo un tag
        emotivo, non emette nulla (evita glitch/silenzi Inworld)."""
        async def _gated():
            buffer: list[str] = []
            released = False
            async for chunk in text:
                chunk = _fix_pronunciation(chunk)
                if released:
                    yield chunk
                    continue
                buffer.append(chunk)
                if _strip_emotion_tags("".join(buffer)).strip():
                    released = True
                    for held in buffer:
                        yield held
                    buffer.clear()

        _default = getattr(Agent, "default", None)
        if _default is not None and hasattr(_default, "tts_node"):
            node = _default.tts_node(self, _gated(), model_settings)
        else:
            node = super().tts_node(_gated(), model_settings)
        async for frame in node:
            yield frame

    # ── filler pattern (1:1 da qualify_master) ──

    async def _fill_then(self, context: RunContext, phrase: str, coro):
        filler_task = asyncio.create_task(self._delayed_filler(context, phrase))
        try:
            return await coro
        finally:
            filler_task.cancel()

    async def _delayed_filler(self, context: RunContext, phrase: str) -> None:
        try:
            await asyncio.sleep(FILLER_DELAY_S)
        except asyncio.CancelledError:
            return
        try:
            if not self._filler_armed:
                return
            if self._filler_handle is not None and not self._filler_handle.done():
                return
            self._filler_armed = False
            self._filler_handle = context.session.say(
                phrase, allow_interruptions=True, add_to_chat_ctx=False
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("filler skipped: %s", e)

    # ───────── Tool 1: knowledge_query — COPIA 1:1 da qualify_master ─────────

    @function_tool()
    async def knowledge_query(self, context: RunContext, query: str) -> str:
        """Rispondi a domande del visitatore su Odyra: servizi, agenti vocali,
        casi studio, prezzi, tecnologie, come si parte. `query` = la frase
        COMPLETA del visitatore (mai una sola parola)."""
        return await self._fill_then(
            context, _localized(FILLERS_KNOWLEDGE, self._active_lang),
            _post_json(RAG_URL, {"id": TENANT_ID, "query": query}))

    # ───────── Tool 2: richiedi_contatto — lead capture ─────────

    @function_tool()
    async def richiedi_contatto(
        self,
        context: RunContext,
        nome: str,
        telefono: str = "",
        email: str = "",
        azienda: str = "",
        interesse: str = "",
    ) -> str:
        """Registra la richiesta di contatto di un visitatore interessato.
        Chiamalo SOLO dopo aver raccolto almeno il nome E un recapito
        (telefono o email). `interesse` = breve sintesi di cosa cerca,
        max 20 parole, generata dal contesto della conversazione."""
        if not telefono and not email:
            return json.dumps({"error": "missing_contact",
                               "message": "Serve almeno un telefono o una email."})
        payload = {
            "tenant_id": TENANT_ID,
            "source": "odyra_website_voice",
            "room": self.md.get("room_name", ""),
            "nome": nome,
            "telefono": telefono,
            "email": email,
            "azienda": azienda,
            "interesse": interesse,
            "created_at": datetime.now(TZ_ROME).isoformat(),
        }
        return await self._fill_then(
            context, _localized(FILLERS_LEAD, self._active_lang),
            _post_json(LEAD_WEBHOOK_URL, payload))

    # ───────── Tool 3: mostra_pagina — naviga il sito del visitatore ─────────

    @function_tool()
    async def mostra_pagina(
        self,
        context: RunContext,
        pagina: str,
    ) -> str:
        """Porta il visitatore su una pagina del sito mentre ne parli. Valori
        ammessi per `pagina`: home, contatti (la sezione con il calendario per
        prenotare una call), case_boss_italia, case_eva, case_sportit,
        case_digital_revenue. Usalo quando racconti un caso studio o quando il
        visitatore vuole prenotare una call, annunciandolo prima a voce con
        naturalezza."""
        routes = {
            "home": "/",
            "contatti": "/#contatti",
            "calendario": "/#contatti",
            "case_boss_italia": "/case/boss-italia",
            "case_eva": "/case/eva",
            "case_sportit": "/case/sportit",
            "case_digital_revenue": "/case/digital-revenue",
        }
        path = routes.get((pagina or "").strip().lower())
        if not path:
            return json.dumps({"error": "pagina_sconosciuta",
                               "valide": list(routes.keys())})
        try:
            room = get_job_context().room
            payload = json.dumps({"action": "navigate", "path": path}).encode("utf-8")
            await room.local_participant.publish_data(payload, reliable=True, topic="ui")
            return json.dumps({"ok": True, "path": path})
        except Exception as e:  # noqa: BLE001
            logger.warning("mostra_pagina failed: %s", e)
            return json.dumps({"error": "navigazione_non_riuscita"})


# ───────────────────────── TTS / LLM builders (1:1 da qualify_master) ─────────────────────────

def _build_inworld_tts():
    kwargs = dict(
        model=INWORLD_MODEL,
        voice=INWORLD_VOICE,
        language=INWORLD_LANGUAGE,
        speaking_rate=INWORLD_SPEAKING_RATE,
    )
    if INWORLD_DELIVERY_MODE:
        kwargs["delivery_mode"] = INWORLD_DELIVERY_MODE
    return inworld.TTS(**kwargs)


def build_tts():
    """Ritorna (engine, targets): `engine` va in AgentSession; `targets` è la lista
    dei motori TTS reali (Inworld [+ Cartesia]) su cui chiamare update_options per
    lo switch di lingua — il FallbackAdapter non propaga in modo garantito."""
    primary = _build_inworld_tts()
    targets = [primary]
    if not (TTS_FALLBACK_ENABLED and os.getenv("CARTESIA_API_KEY")):
        return primary, targets
    try:
        from livekit.agents import tts as _tts
        from livekit.plugins import cartesia

        class _CartesiaNoEmotionTags(cartesia.TTS):
            def synthesize(self, text, **kwargs):
                return super().synthesize(_strip_emotion_tags(text), **kwargs)

            def stream(self, **kwargs):
                stream = super().stream(**kwargs)
                _orig_push = stream.push_text

                def _push_text(token: str) -> None:
                    return _orig_push(_strip_emotion_tags(token))

                stream.push_text = _push_text  # type: ignore[method-assign]
                return stream

        c_kwargs = dict(model=CARTESIA_MODEL, language=CARTESIA_LANGUAGE)
        if CARTESIA_VOICE:
            c_kwargs["voice"] = CARTESIA_VOICE
        fallback = _CartesiaNoEmotionTags(**c_kwargs)
        targets.append(fallback)
        logger.info("TTS: Inworld (main) + Cartesia %s (fallback)", CARTESIA_MODEL)
        return _tts.FallbackAdapter([primary, fallback]), targets
    except Exception as e:  # noqa: BLE001
        logger.warning("TTS fallback Cartesia non disponibile (%s); uso solo Inworld", e)
        return primary, targets


def build_llm():
    primary = openai.LLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE, parallel_tool_calls=False)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from livekit.agents.llm import FallbackAdapter
            from livekit.plugins import anthropic
            fallback = anthropic.LLM(model=LLM_FALLBACK_MODEL, temperature=LLM_TEMPERATURE)
            return FallbackAdapter([primary, fallback], attempt_timeout=LLM_ATTEMPT_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            logger.warning("Fallback Anthropic non disponibile (%s); uso solo OpenAI", e)
    return primary


# ───────────────────────── Worker lifecycle ─────────────────────────

def _load_vad():
    try:
        return silero.VAD.load(activation_threshold=VAD_ACTIVATION_THRESHOLD)
    except TypeError:
        return silero.VAD.load()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = _load_vad()


def _build_transcript(session: AgentSession) -> list[dict]:
    out: list[dict] = []
    try:
        for item in session.history.items:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            if not role or content is None:
                continue
            if isinstance(content, list):
                text = " ".join(c for c in content if isinstance(c, str)).strip()
            else:
                text = str(content).strip()
            if role in ("user", "assistant") and text:
                out.append({"role": role, "text": text})
    except Exception:  # noqa: BLE001
        logger.exception("Impossibile costruire la trascrizione")
    return out


async def entrypoint(ctx: JobContext) -> None:
    try:
        md = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError:
        md = {}
    md["room_name"] = ctx.room.name
    # Lingua iniziale scelta dai bottoni IT/EN/ES del sito (passata dal widget nel
    # metadata del job). In assenza: DEFAULT_LANG. Poi lo STT-multi adatta da solo.
    initial_lang = _normalize_lang(md.get("lang", "")) or DEFAULT_LANG
    logger.info("odyra_web job room=%s lang=%s", ctx.room.name, initial_lang)

    await ctx.connect()

    loop = asyncio.get_running_loop()
    state = {
        "ended_reason": "completed",
        "start": loop.time(),
        "last_activity": loop.time(),
        "closing": False,
        "generating_reply": False,
    }

    vad = ctx.proc.userdata.get("vad") or _load_vad()
    # keyterm è una feature Nova-3 monolingua: in "multi" la lasciamo cadere per
    # non rischiare rifiuti dall'API Deepgram.
    _stt_kwargs = dict(model=DEEPGRAM_MODEL, language=DEEPGRAM_LANGUAGE, numerals=True)
    if DEEPGRAM_LANGUAGE.strip().lower() != "multi":
        _stt_kwargs["keyterm"] = KEYTERMS
    tts_engine, tts_targets = build_tts()
    _session_kwargs = dict(
        stt=deepgram.STT(**_stt_kwargs),
        llm=build_llm(),
        tts=tts_engine,
        vad=vad,
        turn_detection=("vad" if TURN_DETECTION == "vad" else inference.TurnDetector()),
        min_endpointing_delay=MIN_ENDPOINTING_DELAY,
        max_endpointing_delay=MAX_ENDPOINTING_DELAY,
        min_interruption_duration=MIN_INTERRUPTION_DURATION,
    )
    _anti_interrupt = {"min_interruption_words": MIN_INTERRUPTION_WORDS}
    if RESUME_FALSE_INTERRUPTION:
        _anti_interrupt["resume_false_interruption"] = True
        _anti_interrupt["false_interruption_timeout"] = FALSE_INTERRUPTION_TIMEOUT_S
    try:
        session = AgentSession(**_session_kwargs, **_anti_interrupt)
    except TypeError as e:  # noqa: BLE001
        logger.warning("AgentSession: param anti-interruzione non supportati (%s); uso base", e)
        session = AgentSession(**_session_kwargs)

    agent = OdyraWebAgent(md=md, tts_targets=tts_targets, initial_lang=initial_lang)
    # Allinea subito il TTS alla lingua d'apertura (prima del saluto).
    agent._apply_language(initial_lang, initial=True)

    def _touch(*_args) -> None:
        state["last_activity"] = loop.time()

    def _trigger_close(reason: str) -> None:
        if state["closing"]:
            return
        state["closing"] = True
        state["ended_reason"] = reason
        logger.info("[CLOSE] reason=%s → shutdown", reason)
        ctx.shutdown(reason=reason)

    def _on_user_text(ev) -> None:
        if getattr(ev, "is_final", False):
            if (getattr(ev, "transcript", "") or "").strip():
                state["last_activity"] = loop.time()
            agent._filler_armed = True

    session.on("user_input_transcribed", _on_user_text)
    session.on("agent_state_changed", lambda ev: _touch())
    session.on("user_state_changed", lambda ev: _touch())

    def _on_disconnect(participant) -> None:
        # L'utente ha chiuso il widget / la pagina: fine sessione.
        logger.info("participant disconnected: %s", getattr(participant, "identity", "?"))
        _trigger_close("visitor_left")

    ctx.room.on("participant_disconnected", _on_disconnect)

    # ── EOC opzionale (transcript verso n8n, per analytics/lead review) ──
    async def _send_eoc(reason: str = "") -> None:
        if not EOC_WEBHOOK_URL:
            return
        duration = max(0, int(loop.time() - state["start"]))
        payload = {
            "tenant_id": TENANT_ID,
            "room": ctx.room.name,
            "ended_reason": state["ended_reason"],
            "duration_seconds": duration,
            "transcript": _build_transcript(session),
            "agent": AGENT_NAME,
        }
        logger.info("EOC POST reason=%s dur=%ss", state["ended_reason"], duration)
        await _post_json(EOC_WEBHOOK_URL, payload)

    ctx.add_shutdown_callback(_send_eoc)

    # ── watchdog (pattern qualify_master, senza voicemail) ──
    async def _silence_watchdog() -> None:
        while not state["closing"]:
            await asyncio.sleep(2)
            if state["closing"]:
                return
            if session.agent_state in ("thinking", "speaking"):
                state["last_activity"] = loop.time()
                continue
            if loop.time() - state["last_activity"] > SILENCE_TIMEOUT_S:
                logger.info("Silence timeout (%ss) → chiusura", SILENCE_TIMEOUT_S)
                try:
                    await session.say(
                        _localized(GOODBYE_SILENCE, agent._active_lang),
                        allow_interruptions=False,
                    )
                except Exception:  # noqa: BLE001
                    pass
                _trigger_close("silence_timeout")
                return

    async def _max_duration_watchdog() -> None:
        await asyncio.sleep(MAX_DURATION_S)
        if state["closing"]:
            return
        try:
            await session.say(
                _localized(GOODBYE_MAX, agent._active_lang),
                allow_interruptions=False,
            )
        except Exception:  # noqa: BLE001
            pass
        _trigger_close("max_duration")

    async def _stuck_watchdog() -> None:
        waiting_since = None
        prev_user_speaking = False
        while not state["closing"]:
            await asyncio.sleep(0.5)
            if state["closing"]:
                return
            now = loop.time()
            agent_busy = session.agent_state in ("thinking", "speaking")
            user_speaking = session.user_state == "speaking"
            generating = state.get("generating_reply", False)
            if agent_busy or user_speaking or generating:
                waiting_since = None
            else:
                if prev_user_speaking:
                    waiting_since = now
                if (
                    waiting_since is not None
                    and (now - waiting_since) >= STUCK_GRACE_S + STUCK_TIMEOUT_S
                ):
                    logger.warning("[STUCK] nessuna risposta da %.0fs → chiusura", now - waiting_since)
                    _trigger_close("stuck")
                    return
            prev_user_speaking = user_speaking

    # ── Avatar Tavus: si aggancia alla sessione e pubblica video+audio sincronizzati.
    # L'audio della pipeline (voce Inworld) viene inoltrato a Tavus per il lip-sync,
    # quindi l'output audio diretto della room va disabilitato (lo pubblica l'avatar).
    avatar = None
    if AVATAR_ENABLED:
        try:
            from livekit.plugins import tavus
            avatar = tavus.AvatarSession(
                replica_id=TAVUS_REPLICA_ID,
                persona_id=TAVUS_PERSONA_ID,
            )
            await avatar.start(session, room=ctx.room)
            logger.info("Avatar Tavus avviato (replica=%s)", TAVUS_REPLICA_ID)
        except Exception as e:  # noqa: BLE001
            logger.warning("Avatar Tavus non avviato (%s): proseguo voice-only", e)
            avatar = None

    if avatar is not None:
        await session.start(
            agent=agent,
            room=ctx.room,
            room_output_options=RoomOutputOptions(audio_enabled=False),
        )
    else:
        await session.start(agent=agent, room=ctx.room)

    # Musichetta "sto pensando" solo in modalità voice-only: con l'avatar attivo
    # l'audio esce dal participant Tavus e un secondo canale audio farebbe eco.
    if avatar is None:
        try:
            background_audio = BackgroundAudioPlayer(thinking_sound=THINKING_SOUND)
            await background_audio.start(room=ctx.room, agent_session=session)
        except Exception as e:  # noqa: BLE001
            logger.warning("thinking sound non avviato: %s", e)

    # Attendi che il visitatore sia effettivamente in room prima di salutare
    try:
        await asyncio.wait_for(ctx.wait_for_participant(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Nessun visitatore entrato in room entro 30s → chiusura")
        _trigger_close("no_visitor")
        return

    state["generating_reply"] = True
    try:
        await session.say(_localized(GREETINGS, agent._active_lang), allow_interruptions=True)
    finally:
        state["generating_reply"] = False

    _touch()
    sw = asyncio.create_task(_silence_watchdog(), name="silence_watchdog")
    mw = asyncio.create_task(_max_duration_watchdog(), name="max_duration_watchdog")
    kw = asyncio.create_task(_stuck_watchdog(), name="stuck_watchdog")

    async def _cancel_watchdogs(reason: str = "") -> None:
        for t in (sw, mw, kw):
            if not t.done():
                t.cancel()

    ctx.add_shutdown_callback(_cancel_watchdogs)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
        )
    )
