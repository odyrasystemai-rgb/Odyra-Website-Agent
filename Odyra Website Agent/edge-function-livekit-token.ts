// supabase/functions/livekit-token/index.ts
// Edge function di riferimento per il progetto Lovable "Odyra Portal".
//
// Cosa fa: firma un JWT LiveKit (HS256) per il visitatore del widget e
// include roomConfig.agents = [{ agentName: "odyra_web" }] → LiveKit Cloud
// dispatcha automaticamente il worker nella room. Niente dispatcher separato.
//
// Secrets Supabase richiesti:
//   LIVEKIT_URL        es. wss://odyra-poc-xxxxxxx.livekit.cloud
//   LIVEKIT_API_KEY
//   LIVEKIT_API_SECRET
//
// NB: rate-limit basilare incluso (per IP) per non farsi bruciare crediti.

import { SignJWT } from "https://deno.land/x/jose@v5.2.0/index.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

// rate limit in-memory (best effort: le edge function sono effimere)
const hits = new Map<string, { n: number; t: number }>();
const RATE_MAX = 5; // sessioni per IP
const RATE_WINDOW_MS = 10 * 60 * 1000; // in 10 minuti

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  const ip = req.headers.get("x-forwarded-for")?.split(",")[0] ?? "unknown";
  const now = Date.now();
  const h = hits.get(ip);
  if (h && now - h.t < RATE_WINDOW_MS) {
    if (h.n >= RATE_MAX) {
      return new Response(JSON.stringify({ error: "rate_limited" }), {
        status: 429,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    h.n++;
  } else {
    hits.set(ip, { n: 1, t: now });
  }

  const apiKey = Deno.env.get("LIVEKIT_API_KEY")!;
  const apiSecret = Deno.env.get("LIVEKIT_API_SECRET")!;
  const livekitUrl = Deno.env.get("LIVEKIT_URL")!;

  const room = `odyra-web-${crypto.randomUUID()}`;
  const identity = `visitor-${crypto.randomUUID().slice(0, 8)}`;

  const jwt = await new SignJWT({
    video: {
      roomJoin: true,
      room,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    },
    // Dispatch esplicito dell'agente alla creazione della room:
    roomConfig: {
      agents: [{ agentName: "odyra_web" }],
    },
    name: "Visitatore",
  })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuer(apiKey)
    .setSubject(identity)
    .setIssuedAt()
    .setExpirationTime("15m")
    .sign(new TextEncoder().encode(apiSecret));

  return new Response(
    JSON.stringify({ token: jwt, url: livekitUrl, room }),
    { headers: { ...corsHeaders, "Content-Type": "application/json" } },
  );
});
