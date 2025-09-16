// npm i express openai @pinecone-database/pinecone
import express from "express";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const {
  PORT = 3000,
  PINECONE_API_KEY,
  INDEX_NAME,                               // <-- make sure this is set
  INDEX_HOST,                               // e.g. ibm-watson-...b74a.pinecone.io (NO https://, NO /query)
  NAMESPACE = "__default__",
  OPENAI_API_KEY,
  EMBED_MODEL = "text-embedding-3-small",   // 1536 dims to match your index
  TOP_K = "5",
  KNOWLEDGE_API_KEY                         // optional shared secret for Orchestrate
} = process.env;

if (!PINECONE_API_KEY || !INDEX_HOST || !OPENAI_API_KEY || !INDEX_NAME) {
  console.error("Missing env: PINECONE_API_KEY, INDEX_HOST, OPENAI_API_KEY, INDEX_NAME");
  process.exit(1);
}

const app = express();
app.use(express.json({ limit: "1mb" }));


app.use((req, res, next) => {
  if (!KNOWLEDGE_API_KEY) return next();            // if no secret set, don’t block

  const auth = req.get('authorization') || '';
  const key =
    req.get('x-api-key') ||
    req.get('api-key') ||
    (auth.match(/^(?:apikey|bearer)\s+(.+)$/i)?.[1]); // pull token from Authorization

  if (key && key.trim() === KNOWLEDGE_API_KEY.trim()) return next();
  return res.status(401).json({ error: 'unauthorized' });
});


// // optional: protect with API key
// app.use((req, res, next) => {
//   if (!KNOWLEDGE_API_KEY) return next();
//   const key = req.get("x-api-key") || req.get("Api-Key");
//   if (key !== KNOWLEDGE_API_KEY) return res.status(401).json({ error: "unauthorized" });
//   next();
// });

// optional: health check
app.get("/", (_, res) => res.json({ ok: true }));

const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
// ✅ Correct way to get the index
const index = pc.index(INDEX_NAME, INDEX_HOST);

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

async function embed(text) {
  const out = await openai.embeddings.create({ model: EMBED_MODEL, input: text });
  return out.data[0].embedding; // length 1536
}

app.post("/", async (req, res) => {
  try {
    const { query, filter, metadata } = req.body || {};
    if (!query || typeof query !== "string") {
      return res.status(400).json({ error: "query (string) is required" });
    }

    const vector = Array.isArray(metadata?.vector) ? metadata.vector : await embed(query);

    const resp = await index.namespace(NAMESPACE).query({
      vector,
      topK: Number(TOP_K),
      includeMetadata: true,
      ...(filter && typeof filter === "object" ? { filter } : {})
    });

    const search_results = (resp.matches || []).map(m => {
      const md = m.metadata || {};
      return {
        result_metadata: { score: m.score },
        title: md.title || md.document_title || m.id,
        body: md.text || md.chunk_text || "",
        url: md.url || md.document_url || undefined
      };
    });

    res.json({ search_results });
  } catch (err) {
    console.error("Unhandled error:", err);
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.listen(PORT, () => console.log(`Wrapper listening on :${PORT}`));
