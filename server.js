// npm i express openai @pinecone-database/pinecone
import express from "express";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const {
  PORT = 3000,
  PINECONE_API_KEY,
  INDEX_HOST,                   // e.g. ibm-watson-...b74a.pinecone.io
  NAMESPACE = "__default__",
  OPENAI_API_KEY,
  EMBED_MODEL = "text-embedding-3-small", // 1536-dim to match your index
  TOP_K = "5"
} = process.env;

if (!PINECONE_API_KEY || !INDEX_HOST || !OPENAI_API_KEY) {
  console.error("Missing env: PINECONE_API_KEY, INDEX_HOST, OPENAI_API_KEY");
  process.exit(1);
}

const app = express();
app.use(express.json({ limit: "1mb" }));

const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.Index({ host: INDEX_HOST });
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

async function embed(text) {
  const res = await openai.embeddings.create({
    model: EMBED_MODEL,
    input: text
  });
  return res.data[0].embedding; // array length 1536
}

app.post("/", async (req, res) => {
  try {
    const { query, filter, metadata } = req.body || {};
    if (!query || typeof query !== "string") {
      return res.status(400).json({ error: "query (string) is required" });
    }

    // 1) embed the query (or allow a precomputed vector in metadata.vector)
    const vector = Array.isArray(metadata?.vector)
      ? metadata.vector
      : await embed(query);

    // 2) call Pinecone /query
    const q = await index.namespace(NAMESPACE).query({
      vector,
      topK: Number(TOP_K),
      includeMetadata: true,
      // pass through filter only if it is an object
      ...(filter && typeof filter === "object" ? { filter } : {})
    });

    // 3) map Pinecone matches -> Orchestrate schema
    const search_results = (q.matches || []).map(m => {
      const md = m.metadata || {};
      const title = md.title || md.document_title || m.id;
      const body  = md.text  || md.chunk_text  || "";
      return {
        result_metadata: { score: m.score },
        title,
        body,
        url: md.url || md.document_url || undefined,
        // optional highlight: if you have snippets
      };
    });

    res.json({ search_results });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.listen(PORT, () => {
  console.log(`Wrapper listening on :${PORT}`);
});
