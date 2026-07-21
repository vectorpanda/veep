# Bundled sample data — attribution

The `sample.parquet` file shipped in this directory contains plot-text and
title metadata for ~5,000 globally-recognized movies, derived from English
Wikipedia articles. This text is licensed under
[**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**](https://creativecommons.org/licenses/by-sa/4.0/).

## Sources

- **English Wikipedia** (plot summaries, titles, release years, genres)
  — accessed via the `vishnupriyavr/wiki-movie-plots-with-summaries`
  dataset on Hugging Face Hub, which itself derives from the
  [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
  on Kaggle. **License:** CC BY-SA 4.0.
- **Wikidata** (sitelinks count — used at build time to rank candidates
  by global notability; not redistributed). **License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (public domain).

All shipped text fields (`title`, `plot`, `genre`, `year`) trace back to
Wikipedia contributors. The selection criterion (Wikidata sitelinks count)
is itself CC0. No proprietary or non-CC-licensed content is included in
this distribution or the build pipeline.

## Modifications

- The 5,000 most globally-notable titles were selected by inner-joining
  the Wikipedia plot dataset with a Wikidata SPARQL query for film items
  with sitelinks ≥ 8 (i.e. articles in 8+ language Wikipedias), then
  sorting by sitelinks descending.
- Plot text is truncated to ~800 characters per row.
- Embeddings (the `vector` column) were produced locally with the bundled
  ONNX export of `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0).

## Required notices

In line with CC BY-SA 4.0 §3, redistributions of this data must:

1. Credit the original authors (Wikipedia contributors).
2. Indicate that modifications were made (truncation, selection, embedding).
3. Distribute under the same CC BY-SA 4.0 license.
4. Link to the license: <https://creativecommons.org/licenses/by-sa/4.0/>.

This file satisfies (1) and (2). Customers redistributing the bundled data
inherit (3) and (4) — i.e., the data stays CC BY-SA 4.0 even when shipped
inside another product. Customer code/applications that *use* the data
(querying it, indexing it, displaying results) are unaffected — only
*redistribution of the text itself* is share-alike-bound.

## License of the SDK

The veep SDK code is MIT-licensed (see the project root `LICENSE`).
The CC BY-SA 4.0 obligation applies only to the bundled `sample.parquet`
content, not to the SDK code or the application code that uses it.
