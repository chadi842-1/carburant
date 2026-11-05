"""Retrieval over the project documentation: section chunking + BM25 with French normalisation.

BM25 is used rather than embeddings because it runs offline, is deterministic, and on a small
technical corpus with precise vocabulary (dbt, quarantaine, BM25...) lexical matching is strong.
An embedding retriever can replace `Retriever.search` without touching the agent.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import snowballstemmer

DOCS_DIR = Path(__file__).parent / "docs"

STOPWORDS = set("""
a au aux avec ce ces cet cette d dans de des du elle en est et etre il ils je l la le les leur
lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur
ta te tes toi ton tu un une vos votre vous c y quel quelle quels quelles comment pourquoi
est-ce ca cela fait faire t s n j m sont ont a-t-il il-y y-a plus tres sans entre
""".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


_STEMMER = snowballstemmer.stemmer("french")


def tokenize(text: str) -> list[str]:
    """Lower-case words -> drop stop words -> Snowball French stem (on accented form) -> strip
    accents. Stemming before accent stripping matters: 'évaluée' and 'évaluation' share a stem
    only when the stemmer sees the accents."""
    words = re.findall(r"[\w]+", text.lower())
    out = []
    for w in words:
        if normalize(w) in STOPWORDS or len(w) < 2:
            continue
        out.append(normalize(_STEMMER.stemWord(w)))
    return out


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    text: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize(s)).strip("-")


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """One chunk per '## ' section of each markdown file."""
    chunks = []
    for f in sorted(docs_dir.glob("*.md")):
        title, body = None, []
        for line in f.read_text(encoding="utf-8").splitlines() + ["## __end__"]:
            if line.startswith("## "):
                if title and "".join(body).strip():
                    chunks.append(Chunk(f"{f.stem}#{_slug(title)}", f.stem, title,
                                        " ".join(x.strip() for x in body if x.strip())))
                title, body = line[3:].strip(), []
            elif not line.startswith("# "):
                body.append(line)
    return chunks


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    matched_terms: int


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.k1, self.b = k1, b
        # titles count twice: a section title is a strong signal of what the section answers
        self._docs = [tokenize(c.title) * 2 + tokenize(c.text) for c in self.chunks]
        self._tf = [Counter(d) for d in self._docs]
        self._len = [len(d) for d in self._docs]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df = Counter(t for d in self._docs for t in set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = set(tokenize(query))
        hits = []
        for i, tf in enumerate(self._tf):
            score, matched = 0.0, 0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                matched += 1
                denom = f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
                score += self._idf[t] * f * (self.k1 + 1) / denom
            if score > 0:
                hits.append(Hit(self.chunks[i], score, matched))
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]
