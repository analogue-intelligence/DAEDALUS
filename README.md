# poem-cppn-neat

Evolve a **CPPN** with **NEAT** that maps the words of a poem to **colours, positions,
and sizes**, composed inside a target **shape** and rendered as an **SVG**.

```
poem -> embed -> word vectors -> CPPN(NEAT) -> (x, y, r, g, b, size) per word -> SVG art
```

## Two renderers

- `render_svg`      - the diagnostic graph (dark bg, labels, neighbour edges)
- `render_art_svg`  - the composition: soft colour-blobs, gradient-blended curved
                      threads, white background, points filling a circle/square/rect

## Run

```bash
pip install -r requirements.txt
python run_demo.py   # graph view (gen0 vs evolved), dark bg
python run_art.py    # art view: circular + square compositions, white bg
```

Both use **WordNet** embeddings so they run offline with real semantics.

- For per-occurrence colour (same word, different line) keep duplicate tokens:
  `tokenize(poem, keep_dupes=True)` with `SentenceTransformerEmbedder`.
