"""Regenerate the bundled sample corpus.

Pulls ~5000 popular movies from public sources, embeds plot summaries
with the SDK's bundled ONNX encoder, and writes ``sample.parquet`` next
to this module. Run on a host with the ``[samples]`` extra plus pandas,
pyarrow, the ``datasets`` library, and ``requests``::

    pip install veep[samples,pandas] datasets requests
    python -m veep._sample_data.regenerate

## Sources

- ``vishnupriyavr/wiki-movie-plots-with-summaries`` (HF Hub, derived from
  the Wikipedia Movie Plots Kaggle dataset). Wikipedia text is **CC BY-SA
  4.0** — see ATTRIBUTION.md alongside this script.
- Wikidata SPARQL endpoint for popularity ranking via sitelinks count
  (number of language Wikipedias with an article about each item).
  Wikidata content is **CC0** — public domain.

Both sources are CC0 / CC BY-SA. There is no proprietary data in the
build pipeline. (Earlier iterations briefly experimented with TMDb's
``popularity`` field for ranking; that path was dropped before any
commercial release because TMDb's terms of service make commercial
re-derivation a grey area. See server-vbq8 for the decision record.)

## Selection

1. Wikidata SPARQL query — all film items (instance of Q11424) with
   ``sitelinks >= 8`` (i.e. articles in 8+ language Wikipedias). Returns
   a few tens of thousands of films with their global notability score.
2. Inner-join with the Wikipedia plot dataset on (lowercased title,
   release year). Drops any film without a plot summary.
3. Filter to English-speaking origins (American, British, Australian,
   Canadian) and plot text >= 200 chars.
4. Sort by sitelinks descending, take the top 5000.

## Shipped fields

Only Wikipedia-sourced text:
    id      — opaque film-NNNNN identifier
    title   — film title
    year    — release year
    genre   — comma/dash-separated genre list
    plot    — first ~800 chars of the plot summary
    vector  — 384-dim embedding from samples.encode(plot)

No Wikidata Q-ids, no sitelink counts, no TMDb anything.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

MAX_PLOT_CHARS = 800
TOP_N = 5000
MIN_SITELINKS = 8
ENGLISH_ORIGINS = {"American", "British", "Australian", "Canadian"}
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "veep-quickstart-corpus-build/0.5.2 (https://vectorpanda.com)"

SPARQL_QUERY = f"""
SELECT ?title ?year ?sitelinks WHERE {{
  ?film wdt:P31/wdt:P279* wd:Q11424 ;
        wdt:P577 ?date ;
        wikibase:sitelinks ?sitelinks .
  BIND(YEAR(?date) AS ?year)
  FILTER(?sitelinks >= {MIN_SITELINKS})
  ?article schema:about ?film ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?title .
  FILTER(LANG(?title) = "en")
}}
ORDER BY DESC(?sitelinks)
"""


def _norm_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _truncate(s: str) -> str:
    if len(s) <= MAX_PLOT_CHARS:
        return s
    cut = s[:MAX_PLOT_CHARS]
    last_space = cut.rfind(" ")
    if last_space > MAX_PLOT_CHARS - 100:
        cut = cut[:last_space]
    return cut + " …"


def _fetch_wikidata_films():
    import pandas as pd
    import requests

    print("Querying Wikidata for films + sitelinks ...")
    resp = requests.get(
        WIKIDATA_SPARQL,
        params={"query": SPARQL_QUERY, "format": "json"},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
        timeout=120,
    )
    resp.raise_for_status()
    rows = []
    for b in resp.json()["results"]["bindings"]:
        rows.append({
            "title": b["title"]["value"],
            "year": int(b["year"]["value"]),
            "sitelinks": int(b["sitelinks"]["value"]),
        })
    df = pd.DataFrame(rows)
    print(f"  Wikidata returned {len(df):,} films")
    return df


def main() -> None:
    import pandas as pd
    from datasets import load_dataset

    from veep._encoder import encode

    print("Loading Wikipedia plot dataset (vishnupriyavr/wiki-movie-plots-with-summaries) ...")
    wiki = load_dataset(
        "vishnupriyavr/wiki-movie-plots-with-summaries", split="train"
    ).to_pandas()
    print(f"  Wiki rows: {len(wiki):,}")

    # Wikidata sitelinks for popularity
    wd = _fetch_wikidata_films()
    wd["title_norm"] = wd["title"].astype(str).map(_norm_title)
    wd = wd.drop_duplicates(subset=["title_norm", "year"], keep="first")

    # Prep wiki side
    wiki_filt = wiki.dropna(subset=["Title", "Release Year", "Plot"]).copy()
    wiki_filt["title_norm"] = wiki_filt["Title"].astype(str).map(_norm_title)
    wiki_filt["year"] = wiki_filt["Release Year"].astype(int)
    wiki_filt["plot_len"] = wiki_filt["Plot"].astype(str).map(len)
    wiki_filt = wiki_filt[wiki_filt["plot_len"] >= 200]
    wiki_filt = wiki_filt[wiki_filt["Origin/Ethnicity"].isin(ENGLISH_ORIGINS)]
    print(f"  Wiki after filters: {len(wiki_filt):,}")

    # Inner-join
    joined = wiki_filt.merge(
        wd[["title_norm", "year", "sitelinks"]],
        on=["title_norm", "year"],
        how="inner",
    ).drop_duplicates(subset=["title_norm", "year"], keep="first")
    print(f"  Joined rows: {len(joined):,}")

    top = joined.nlargest(TOP_N, "sitelinks").reset_index(drop=True)
    print(f"  Selected top {len(top):,} by sitelinks "
          f"(range: {top['sitelinks'].min()} – {top['sitelinks'].max()})")

    # Build final shape — Wikipedia-sourced text only
    final = pd.DataFrame({
        "id":    [f"film-{i:05d}" for i in range(len(top))],
        "title": top["Title"].astype(str).str.strip().values,
        "year":  top["year"].astype(int).values,
        "genre": top["Genre"].astype(str).values,
        "plot":  top["Plot"].astype(str).map(_truncate).values,
    })

    # Embed plots with the bundled encoder
    print(f"Encoding {len(final):,} plots with the bundled INT8 ONNX ...")
    t0 = time.time()
    vectors = []
    for i, plot in enumerate(final["plot"]):
        if i and i % 500 == 0:
            rate = i / (time.time() - t0)
            eta = (len(final) - i) / rate
            print(f"  {i:,}/{len(final):,}  ({rate:.0f} rows/s, ETA {eta:.0f}s)")
        vectors.append(encode(plot))
    print(f"  done in {time.time() - t0:.1f}s")
    final["vector"] = vectors

    out = Path(__file__).resolve().parent / "sample.parquet"
    final.to_parquet(out, compression="zstd", index=False)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
