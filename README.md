# RAG-Powered Q&A over Paul Graham's Essays

A retrieval-augmented generation system that answers questions grounded in Paul Graham's essays, built from scratch without LangChain or LlamaIndex (so I could understand every step of the process at a verbose level)

## Knowledge Base

**Choice: Paul Graham Essays** (Option A)

I'm going with Paul Graham's essays because they cover a diverse range of topics (startups, programming, writing, philosophy, education). The diversity of the content tests whether retrieval works well across the different topics. At the same time the consistent style makes faithfulness evaluation more straightforward i.e. a hallucinated claim is easier to detect when it doesn't match PG's recognizable way of arguing. 

## Framework

**Choice: Built from scratch** (no LangChain, no LlamaIndex)

The RAG pipeline here — scrape, chunk, embed, index, retrieve, generate — is simple enough that each step fits in a single short module. A framework would add indirection (chains, callbacks, prompt templates) without reducing complexity. Building directly means every design choice is visible in the code and easy to modify or explain.

## Setup

```bash
# 0. Create and activate virtual environment
virtuanlenv .venv
source .venv/bin/activate

# 1. Install dependencies
pip install -r requirements.txt

# we copy a placeholder template for API key
cp .env.example .env

# 2. Set your OpenAI API key (required for embedding + generation) IMPORTANT: KEY IS PRESENT IN EMAIL
# Edit .env and set OPENAI_API_KEY=<API_KEY_VALUE_HERE>

# 3. Scrape the essays (~220 essays, takes a few minutes with rate limiting)
python -m src.scraper

# 4. Build the FAISS index (embeds all chunks via OpenAI API)
python -m src.indexer

# 5. Ask questions
python main.py --query "What does Paul Graham think about startups?"

# Or run interactively
python main.py

# Or run the evaluation suite
python main.py --evaluate
```

## Architecture

```
Question → Embed query → FAISS top-k retrieval → LLM generation (with context) → Cited answer
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Scraper | `src/scraper.py` | Downloads essays from paulgraham.com as plain text |
| Chunker | `src/chunker.py` | Recursive text splitting (paragraphs → sentences → chars) |
| Indexer | `src/indexer.py` | Embeds chunks with OpenAI, builds FAISS index |
| Retriever | `src/retriever.py` | Cosine similarity search over the index |
| Generator | `src/generator.py` | GPT-4o-mini generation with source grounding |
| Pipeline | `src/pipeline.py` | Wires retriever + generator together |
| Evaluator | `src/evaluate.py` | Faithfulness scoring over manual Q&A pairs |

## Design Decisions

**Built RAG pipeline from scratch** The RAG pipeline is simple enough that a framework adds abstraction without value. Building directly exposes each decision point and makes the code easier to reason about.

**Chose FAISS with flat inner product** for a collection of ~220 essays (~5k chunks), brute-force search appears to be fast enough and accurate enough. Approximate indices.

**Using `text-embedding-3-small`** provides us with good quality-to-cost ratio for this use case. The 1536-dim embeddings capture semantic similarity well enough for essay-level retrieval.

**Using recursive chunking** allows for splitting on paragraph boundaries first preserves coherent ideas. Falling back to sentence and character splits handles long paragraphs. The 50-char overlap ensures context isn't lost at chunk boundaries.

**`chunk_size=500`** balances granularity (short enough to be specific in retrieval) against context (long enough to carry a complete thought). Tested empirically — shorter chunks fragment ideas; longer chunks dilute relevance.

**Chunking limitations:** The recursive splitter handles PG's essay structure well enough, but it has no awareness of semantic boundaries — a paragraph discussing two distinct ideas will stay as one chunk. The overlap mechanism can also push chunks slightly beyond `chunk_size` since it prepends context from the previous chunk. For more structured documents (tables, code, lists), a format-aware chunker would be needed.

**Hallucination mitigation strategy:**
1. **Prompt constraint** — System prompt instructs the model to answer ONLY from provided context and say "I don't have enough information" otherwise.
2. **Source grounding** — Retrieved chunks are numbered and the model cites them inline (`[1]`, `[2]`), making claims traceable.
3. **Low temperature (0.2)** — Reduces creative generation that might drift from sources.

## Evaluation

The evaluation suite (`python main.py --evaluate`) tests 7 manually-crafted question-answer pairs covering topics across PG's essays. Each answer is scored for **faithfulness** — whether every claim in the generated answer is supported by the retrieved context — using GPT-4o-mini as a judge (score 0.0–1.0).

### Results

| Question | Faithfulness | Sources |
|----------|:---:|:---:|
| What does Paul think about working on hard problems? | 1.00 | 4 |
| What is Graham's advice for startups? | 1.00 | 5 |
| What does the author say about essays and writing? | 1.00 | 3 |
| How does the writer view the relationship between programming and painting? | 1.00 | 2 |
| What does the narrator think about wealth creation? | 1.00 | 3 |
| What is the author's perspective on formal education? | 1.00 | 5 |
| What does Paul Graham say about the difference between a startup and a regular business? | 1.00 | 4 |
| **Average** | **1.00** | |

### Interpretation

On all 7 questions, the RAG pipeline achieves perfect faithfulness scores, meaning the LLM judge found every claim in each generated answer to be directly supported by the retrieved context passages. This is a great result for source of truth grounding, enforced by the system prompt's strict constraint to answer only from provided context along with the low generation temperature (0.2).

However, these scores should be interpreted with caution. The evaluation has two key biases: (1) the questions are broad and topically central to PG's essays, so retrieval is easy — adversarial or niche queries would likely score lower; and (2) using GPT-4o-mini as both generator and judge introduces systematic bias, since the judge may be lenient toward outputs that match its own generation patterns. A more robust evaluation would include human annotation, adversarial test cases (e.g., questions about topics PG never wrote about), and additional metrics like answer relevance and retrieval recall.

## Sample Queries

### Query 1
```
$ python main.py --query "What does Paul Graham think about startups?"

Answer:
Paul Graham believes that startups operate in a competitive environment that is
more akin to running than to team sports like soccer, suggesting that startups
should focus on their own progress rather than directly competing with others
for resources or ideas [1]. He also emphasizes that startup investing is
counterintuitive, with most returns coming from a few big winners and that the
best ideas often appear to be bad ideas initially [2]. Furthermore, he advises
potential founders to reflect on their experiences and insights when considering
starting a startup, indicating that understanding existing technology and
recognizing opportunities for improvement is crucial [5].

Sources:
  [1] Startup Investing Trends — https://paulgraham.com/invtrend.html
  [2] Black Swan Farming — https://paulgraham.com/swan.html
  [3] Before the Startup — https://paulgraham.com/before.html
  [5] How to Start a Startup — https://paulgraham.com/start.html
```

### Query 2
```
$ python main.py --query "How does Paul Graham view the relationship between programming and painting?"

Answer:
Paul Graham views the relationship between programming and painting as one where
both activities share fundamental similarities. He argues that both hackers
(programmers) and painters are makers, engaged in the creative process of
producing good things. He emphasizes that hacking is not merely a cold and
methodical task, nor is painting just a frenzied expression; instead, both
involve a learning process through doing, where each work builds on previous
experiences [1][5].

Sources:
  [1] Hackers and Painters — https://paulgraham.com/hp.html
  [2] How to Get Startup Ideas — https://paulgraham.com/startupideas.html
```

### Query 3
```
$ python main.py --query "What is Paul Graham's advice on writing essays?"

Answer:
Paul Graham's advice on writing essays includes several key points:
1. Start with a bad version as quickly as possible and rewrite multiple times [4].
2. Cut unnecessary content [4].
3. Write in a conversational tone [4].
4. Develop a nose for bad writing in your own work [4].
5. Imitate writers you admire [4].
6. Expect 80% of ideas to emerge after starting to write [4].
7. Writing is crucial not just for communication but for idea generation [5].

Sources:
  [1] The Best Essay — https://paulgraham.com/best.html
  [3] Is It Worth Being Wise? — https://paulgraham.com/wisdom.html
  [4] Writing, Briefly — https://paulgraham.com/writing44.html
```

## Known Limitations

- **Dense retrieval only** — No keyword/sparse retrieval (BM25). Queries using specific terminology that doesn't appear in semantically similar contexts may miss relevant chunks.
- **No cross-encoder reranking** — Retrieved chunks are ranked by embedding similarity only. A cross-encoder reranker would improve precision.
- **Single-turn only** — No conversation history or follow-up question handling.
- **LLM-as-judge evaluation** — Faithfulness scoring uses the same model family, which may have systematic biases. Human evaluation would be more reliable.
- **No hybrid search** — Combining dense and sparse retrieval would improve recall for factual/keyword queries.

## Additional steps for improvement I would take (however these are out of scope of the brief)

- Adding reciprocal rank fusion step
- Adding a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM`) between retrieval and generation
- Implementing chunk deduplication and diversity-based reranking
- Adding answer relevance and completeness metrics alongside faithfulness
- Adding streaming responses for the interactive mode
