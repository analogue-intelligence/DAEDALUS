# poem-cppn-neat

Evolve a **CPPN** with **NEAT** that maps the words of a poem to **colours, positions,
and sizes**, composed inside a target **shape** and rendered as an **SVG**.

```
poem -> embed -> word vectors -> CPPN(NEAT) -> (x, y, r, g, b, size) per word -> SVG art
```

## Two renderers

- `render_svg`      — the diagnostic graph (dark bg, labels, neighbour edges)
- `render_art_svg`  — the composition: soft colour-blobs, gradient-blended curved
                      threads, white background, points filling a circle/square/rect

## Run

```bash
pip install -r requirements.txt
python run_demo.py   # graph view (gen0 vs evolved), dark bg
python run_art.py    # art view: circular + square compositions, white bg
```

Both use **WordNet** embeddings so they run fully offline with real semantics.

## The composition shape

`Evolver(..., shape="circle"|"square"|"rect")` constrains where points can land
(circle uses an area-preserving square→disk map). A **coverage** term in the
fitness rewards actually *filling* that shape, so the silhouette reads instead of
the points clumping in one corner.

## Tuning the look — `FitnessWeights`

| field          | effect                                            |
|----------------|---------------------------------------------------|
| `pos`          | similar words near each other (semantic clusters) |
| `color`        | similar words get similar colours                 |
| `color_spread` | use a full palette instead of one hue             |
| `coverage`     | fill the target shape (raise for a crisper shape) |
| `spread`       | use the canvas (anti-collapse)                    |
| `overlap`      | push apart points drawn on top of each other      |

`render_art_svg` knobs: `edge_k` (neighbours per word), `edge_opacity`, `curve`
(thread curvature), `blob_opacity`, `labels` (on/off), `bg`.

## Real embeddings (one-line swap for your own runs)

```python
from poem_cppn import SentenceTransformerEmbedder, GloVeEmbedder
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")  # strongest, handles OOV
words, vectors = embedder.embed(poem)
```

## Interactive (Picbreeder) mode

Set the fitness weights to ~0, render the top-K genomes each generation with
`render_art_svg`, let a human pick favourites, reproduce only from those.
`Evolver.layout_for(genome)` gives `(positions, colours, sizes)` for any genome.

## Files

```
poem_cppn/
  embed.py       embedding backends (WordNet / Hash / GloVe / SentenceTransformer)
  cppn.py        NEAT config + net -> (x,y,rgb,size), shape mapping
  fitness.py     corpus-free objective incl. shape coverage
  render.py      diagnostic graph SVG
  render_art.py  art SVG: blobs + gradient threads + shape, white bg
  evolve.py      the NEAT loop
run_demo.py      graph example
run_art.py       art example (circle + square)
```

## Next steps

- Animate by rendering one frame per generation and crossfading.
- Add output nodes for rotation / opacity / stroke to enrich the composition.
- For per-occurrence colour (same word, different line) keep duplicate tokens:
  `tokenize(poem, keep_dupes=True)` with `SentenceTransformerEmbedder`.
