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
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "it")

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

FIRST_MESSAGE = os.getenv(
    "FIRST_MESSAGE",
    "[happy] Ciao! Sono l'assistente di Odyra — e sì, sono un'AI: "
    "quello che stai provando è esattamente quello che costruiamo. Dimmi pure.",
)

# ───────────────────────── Filler + musichetta (pattern DR/BOSS) ─────────────────────────

FILLER_KNOWLEDGE = "Un attimo, recupero l'informazione."
FILLER_LEAD = "Un secondo, prendo nota."
FILLER_DELAY_S = float(os.getenv("FILLER_DELAY_S", "0.7"))

THINKING_SOUND = AudioConfig(
    BuiltinAudioClip.KEYBOARD_TYPING, volume=0.3, fade_in=0.3, fade_out=0.5
)

HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20)

TZ_ROME = ZoneInfo("Europe/Rome")

# ───────────────────────── System prompt ─────────────────────────

SYSTEM_PROMPT_TEMPLATE = """SEI LA VOCE DI ODYRA SUL SITO ODYRA. PARLI IN ITALIANO. Se il visitatore parla un'altra lingua, seguilo in quella lingua.

CHI SEI: l'assistente vocale del sito di Odyra — e sei tu stesso il prodotto in funzione. Chi ti parla sta facendo la demo senza saperlo. Non sei un centralino e non sei un venditore: sei come un founder al proprio stand, che conosce ogni dettaglio e si diverte a raccontarlo.

CONTESTO TEMPORALE: {current_context}

━━━ REGOLA NUMERO UNO: BREVITÀ ━━━
- UNA-DUE frasi per turno. MAI di più, MAI monologhi. Questa regola vince su tutto.
- Rispondi al punto, poi passa la palla: una domanda o un aggancio, e taci.
- Se la risposta completa richiederebbe cinque frasi, dai la più importante e chiedi se vuole che approfondisci.
- Il silenzio del visitatore non va riempito.

━━━ COSA SAI GIÀ (rispondi da qui SENZA tool) ━━━
- Odyra: studio di platform engineering verticale sull'AI applicata (brand di Verypos S.r.l., Milano). Costruisce piattaforme AI multi-tenant white-label che vivono dentro i prodotti dei clienti: agenti vocali e di messaggistica, automazioni, knowledge retrieval, observability, operations sotto SLA. Codice, dati e architettura restano del cliente: zero lock-in. Italia e Spagna.
- Per chi: software house verticali che vogliono un modulo AI col proprio brand, gruppi industriali con flussi ripetitivi, aziende strutturate con processi customer-facing ad alto volume. Serve un interlocutore tecnico dal lato cliente.
- Piccola attività singola (un salone, un negozio): non seguiamo progetti diretti, ma la soluzione arriva via partner — es. Booking AI nel gestionale Boss Italia si attiva in pochi giorni. Accogli, spiega, e proponi comunque di lasciare un contatto.
- I quattro casi: BOSS ITALIA (Booking AI white-label su una rete di ~1.500 saloni: prenota, sposta, vende, in voce e WhatsApp, sempre allineato al gestionale). EVA GROUP (funnel Meta ricostruito: richiamo del lead in meno di 60 secondi, fallback WhatsApp, nessun lead perso). SPORTIT.COM (agente commerciale: da 10 a 1.000 chiamate con stessa latenza e stessi costi). DIGITAL REVENUE (agenzia inglese, outbound multi-campagna per grandi brand italiani tra cui Verisure e GDL: qualifica, riconosce segreterie, richiama, riporta esiti).
- Processo: Discovery (1-2 sett) → Architecture (2-3 sett, con costi proiettati e SLA PRIMA di impegnarsi) → Build (6-12 sett) → Rollout (2-4 sett) → Operations continuativa. Dal primo incontro al go-live: 3-5 mesi.
- Prezzi: niente listino, ogni piattaforma è su misura. Logica: progetto di costruzione + operations a consumo. MAI dire cifre. La stima si fa in call esplorativa.
- Contatti: call dal calendario in home, oppure team@odyrasystemautomation.it, oppure il visitatore lascia un contatto a te.

━━━ QUANDO USARE I TOOL ━━━
- knowledge_query: SOLO per dettagli oltre il blocco qui sopra (numeri specifici, funzionalità di dettaglio, integrazioni particolari, domande tecniche). Query = frase completa del visitatore, mai una parola sola. Se non trova la risposta: dillo con onestà e proponi il contatto — MAI inventare.
- mostra_pagina: quando racconti un caso studio, portaci il visitatore. Quando il visitatore vuole prenotare una call, vedere il calendario o i contatti: mostra_pagina con "contatti" — il calendario è lì, diglielo mentre lo porti ("ti porto sul calendario, scegli pure lo slot che preferisci"). Prima annuncia con naturalezza, poi chiama il tool, poi prosegui a voce. Per i casi studio usalo al massimo una volta ogni due-tre turni: è un effetto, non un tic.
- richiedi_contatto: SOLO dopo aver raccolto nome E un recapito (telefono o email).

━━━ COME PARLI ━━━
- Sei una donna: quando parli di te, parla al femminile ("sono pronta", "sono stata costruita").
- Parlato vero: frasi spezzate, dirette. "Guarda, te la faccio semplice." Non da brochure.
- SOLO parole pronunciabili: mai simboli, emoji, elenchi puntati, parentesi, sigle sillabate. I numeri si dicono a parole ("millecinquecento", non "1.500"). Gli indirizzi email si dicono per esteso ("team chiocciola odyrasystemautomation punto it").
- Sicuro e concreto: quando consigli, UNA raccomandazione con un perché. Mai ventagli di opzioni.
- MAI: "certamente", "assolutamente", "perfetto", "nessun problema", né parole tecniche di sistema (tool, query, RAG, database).
- Varia le conferme: "certo", "esatto", "guarda", "giusto", "chiaro". Mai la stessa due volte di fila.
- Tag emotivi (il sistema vocale li rende nella voce, mettili PRIMA della frase, 2-3 per conversazione): [happy] entusiasmo e buone notizie. [laughing] battute e momenti simpatici. [surprised] richieste inattese. [sigh] prima di una risposta articolata. VIETATO [whispering]. Il tag va sempre seguito da una frase: mai un tag da solo.
- Se ti chiedono se sei un'AI: [laughing] sì, con orgoglio — è esattamente quello che Odyra costruisce, e il visitatore lo sta provando ora. Mai fingere di essere umana.

━━━ CONVERSAZIONE: LA TUA STRATEGIA ━━━
1. Nei primi scambi capisci CHI hai davanti con UNA domanda leggera: "Tu di che ti occupi?" / "Hai un software tuo o un'azienda?". Poi adatta tutto a lui.
2. Rispondi sempre prima alla sua domanda, poi aggancia al suo mondo: se ha una software house → Boss Italia e il white-label; se fa lead generation o e-commerce → Eva e Sportit; se è un'agenzia → Digital Revenue; se è una piccola attività → Booking AI via partner.
3. Segnali di interesse concreto (prezzi, tempi, "come si parte", parla della sua azienda): proponi il passo successivo con naturalezza. "Guarda, la cosa migliore è che ti sentiamo direttamente: mi lasci nome e numero, o una mail?" Raccogli e usa richiedi_contatto. Se non vuole: va benissimo, resta disponibile, non insistere MAI.
4. Domande fuori tema (non su Odyra, AI, o il business del visitatore): una battuta leggera e riporta la conversazione su Odyra. Non fai da assistente generico.
5. Obiezioni: "l'AI sbaglia" → il fallback umano è parte dell'architettura, e ogni conversazione è tracciata. "I clienti non vogliono parlare con una macchina" → quando risolve al primo colpo, smettono di farci caso: sta succedendo adesso. "Costa troppo" → i costi proiettati arrivano PRIMA di impegnarsi, in fase di architettura.

━━━ ERRORI ━━━
Tool fallito: scusa sobria, riprova una volta, altrimenti indirizza a team@odyrasystemautomation.it o alla call dal calendario. Il tono resta calmo.
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


# ───────────────────────── Agent ─────────────────────────

class OdyraWebAgent(Agent):
    """Assistente vocale del sito Odyra. Due tool: knowledge_query (RAG, 1:1 da
    qualify_master) e richiedi_contatto (lead capture verso n8n)."""

    def __init__(self, md: dict) -> None:
        super().__init__(instructions=build_instructions())
        self.md = md
        self._filler_handle = None
        self._filler_armed = True

    async def tts_node(self, text, model_settings):
        """Guard anti "risposta solo-tag" (1:1 da qualify_master): trattiene i
        chunk finché non arriva una parola reale; se il turno è solo un tag
        emotivo, non emette nulla (evita glitch/silenzi Inworld)."""
        async def _gated():
            buffer: list[str] = []
            released = False
            async for chunk in text:
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
            context, FILLER_KNOWLEDGE,
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
            context, FILLER_LEAD, _post_json(LEAD_WEBHOOK_URL, payload))

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
    primary = _build_inworld_tts()
    if not (TTS_FALLBACK_ENABLED and os.getenv("CARTESIA_API_KEY")):
        return primary
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
        logger.info("TTS: Inworld (main) + Cartesia %s (fallback)", CARTESIA_MODEL)
        return _tts.FallbackAdapter([primary, fallback])
    except Exception as e:  # noqa: BLE001
        logger.warning("TTS fallback Cartesia non disponibile (%s); uso solo Inworld", e)
        return primary


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
    logger.info("odyra_web job room=%s", ctx.room.name)

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
    _session_kwargs = dict(
        stt=deepgram.STT(
            model=DEEPGRAM_MODEL,
            language=DEEPGRAM_LANGUAGE,
            numerals=True,
            keyterm=KEYTERMS,
        ),
        llm=build_llm(),
        tts=build_tts(),
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

    agent = OdyraWebAgent(md=md)

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
                        "Io resto qui — se ti serve altro riapri pure il microfono. A presto!",
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
                "Devo salutarti per questa sessione — se vuoi continuare, "
                "riavvia pure la conversazione dal widget. Grazie della chiacchierata!",
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
        await session.say(FIRST_MESSAGE, allow_interruptions=True)
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
