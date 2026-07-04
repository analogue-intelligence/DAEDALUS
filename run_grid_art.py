"""
Reading-order colour-field renderer (anchor-free, perceptual colour).

Lays the poem on an invisible grid (line = row, word = column), colours each word from the perceptual shape of the embedding cloud (no colour table, no anchors),
and blends the colours into one continuous gradient field.

Saves into ./the_art :
  <slug>_field.svg          the colour field only
  <slug>_field_titled.svg   the field with the poem's name on top and each word labelled at its reading position

Run:  python run_grid_art.py
"""
import os
import html
import numpy as np
from poem_cppn import WordNetEmbedder, HashEmbedder
from poem_cppn.grid_field import (read_grid, grid_positions, position_colors, blend_field, field_to_svg, labels_svg)
from poem_cppn.manifold_color import manifold_colors, occurrence_vectors

POEM_NAME = "Blind Joe Death"  # for title and output filename I wanted to test a Kate Bush poem
POEM = """
The globe spins,
        Dragging collisions of clutches, to the end.
Blind Joe Death staggers to the instrument
        And caresses the soft wood of the neck.
He guides the dizzy fingers through
        The mist of Melancholy melody.
Blind Joe Death grins at Fahey
        And moves the cap up the strings.
        He stops.
Blind Joe Death dies.
He falls onto the round Persian mat
And swings the needle off the turntable.
        Fahey sleeps.
"""
COLOR_MODE = "linear"        # "plane" (angle=hue) | "linear" (3-comp) | "position"
CONTEXT = 0.35              # 0 = colour by word type; >0 = shade by line context
HUE_ROTATION = 0            # global hue dial, degrees (pure aesthetics)
WIDTH, HEIGHT = 1000, 720
MARGIN = 80
CELL = 12
OUT_DIR = "the_art"
# For sharper colour: swap the embedder below for
#   from poem_cppn import SentenceTransformerEmbedder
#   embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")



def slugify(name):
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "poem"


def wrap_with_title(inner_svg, title, width, height, header=80):
    inner = inner_svg[inner_svg.index(">") + 1: inner_svg.rindex("</svg>")]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height + header}" viewBox="0 0 {width} {height + header}">'
        f'<rect width="{width}" height="{height + header}" fill="#ffffff"/>'
        f'<text x="{width/2:.0f}" y="{header*0.6:.0f}" text-anchor="middle" '
        f'fill="#222222" font-family="Georgia, serif" font-style="italic" '
        f'font-size="32">{html.escape(title)}</text>'
        f'<g transform="translate(0,{header})">{inner}</g></svg>'
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    from poem_cppn import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    # embeddings for the vocabulary
    words, vectors = embedder.embed(POEM)
    w2v = {w: v for w, v in zip(words, vectors)}

    # reading-order grid (positions independent of meaning)
    lines = read_grid(POEM)
    tokens, positions, rc = grid_positions(lines, WIDTH, HEIGHT, margin=MARGIN)
    rows = [r for (r, _c) in rc]
    print(f"{len(lines)} lines, {len(tokens)} placed words")

    # one colour per placed token
    if COLOR_MODE == "position":
        token_colors = position_colors(positions, WIDTH, HEIGHT, margin=MARGIN)
    else:
        line_vecs = [np.array([w2v[w] for w in line]).mean(0) for line in lines]
        occ = occurrence_vectors(tokens, rows, w2v, line_vecs, context=CONTEXT)
        token_colors = manifold_colors(occ, mode=COLOR_MODE, hue_rotation=HUE_ROTATION)

    field = blend_field(positions, token_colors, WIDTH, HEIGHT, cell=CELL)
    slug = slugify(POEM_NAME)

    art = field_to_svg(field, WIDTH, HEIGHT)
    art_path = os.path.join(OUT_DIR, f"{slug}_field.svg")
    with open(art_path, "w") as f:
        f.write(art)

    labeled = field_to_svg(field, WIDTH, HEIGHT,
                           header_inner=labels_svg(tokens, positions))
    titled = wrap_with_title(labeled, POEM_NAME, WIDTH, HEIGHT)
    titled_path = os.path.join(OUT_DIR, f"{slug}_field_titled.svg")
    with open(titled_path, "w") as f:
        f.write(titled)

    print(f"saved:\n  {art_path}\n  {titled_path}")


if __name__ == "__main__":
    main()