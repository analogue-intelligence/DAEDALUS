"""
Geometric art from the poem: same reading-order layout and perceptual colours as the field renderer, but rendered as hard-edged shapes.

Saves into ./the_art :
  <slug>_voronoi.svg   flat-filled Voronoi cells (stained glass)
  <slug>_lowpoly.svg   Delaunay triangulation (low-poly)
  <slug>_tiles.svg     colour blocks on the reading grid

Run:  python run_geo_art.py
"""
import os
import numpy as np
import html
from poem_cppn import WordNetEmbedder, HashEmbedder
from poem_cppn.embed import SentenceTransformerEmbedder
from poem_cppn.grid_field import read_grid, grid_positions, labels_svg
from poem_cppn.manifold_color import manifold_colors, occurrence_vectors
from poem_cppn.geometric import voronoi_svg, lowpoly_svg, tiles_svg

POEM_NAME = "Emily" #by Joanna Newsom
POEM = """
The meadowlark and the chim-choo-ree and the sparrow
Set to the sky in a flying spree
For the sport over the pharaoh
Little while later the Pharisees dragged comb through the meadow
Do you remember what they called up to you and me, in our window?
There is a rusty light on the pines tonight
Sun pouring wine, lord, or marrow
Into the bones of the birches
And the spires of the churches
Jutting out from the shadows
The yoke, and the axe, and the old smokestacks and the bale and the barrow
And everything sloped like it was dragged from a rope
In the mouth of the south below
We've seen those mountains kneeling, felten and grey
We thought our very hearts would up and melt away
From that snow in the night time
Just going, and going
And the stirring of wind chimes
In the morning, in the morning
Helps me find my way back in
From the place where I have been
And, Emily, I saw you last night by the river
I dreamed you were skipping little stones across the surface of the water
Frowning at the angle where they were lost, and slipped under forever
In a mud-cloud, mica-spangled, like the sky'd been breathing on a mirror
Anyhow, I sat by your side, by the water
You taught me the names of the stars overhead that I wrote down in my ledger
Though all I knew of the rote universe were those Pleiades loosed, in December
I promised you I'd set them to verse so I'd always remember
That the meteorite is a source of the light
And the meteor's just what we see
And the meteoroid is a stone that's devoid of the fire that propelled it to thee
And the meteorite's just what causes the light
And the meteor's how it's perceived
And the meteoroid's a bone thrown from the void that lies quiet in offering to thee
You came and lay a cold compress upon the mess I'm in
Threw the window wide and cried, amen, amen, amen
The whole world stopped to hear you hollering
And you looked down and saw now what was happening
The lines are fadin' in my kingdom
Though I have never known the way to border 'em in
So the muddy mouths of baboons and sows, and the grouse, and the horse and the hen
Grope at the gate of the looming lake that was once a tidy pen
And the mail is late and the great estates are not lit from within
The talk in town's becoming downright sickening
In due time we will see the far buttes lit by a flare
I've seen your bravery, and I will follow you there
And row through the night time
So healthy
Gone healthy all of a sudden
In search of the midwife
Who could help me
Who could help me
Help me find my way back in
And there are worries where I've been
And say, say, say in the lee of the bay, don't be bothered
Leave your troubles here where the tugboats shear the water from the water
Flanked by furrows, curling back, like a match held up to a newspaper
Emily, they'll follow your lead by the letter
And I make this claim, and I'm not ashamed to say I knew you better
What they've seen is just a beam of your sun that banishes winter
Let us go, though we know it's a hopeless endeavor
The ties that bind, they are barbed and spined and hold us close forever
Though there is nothing would help me come to grips with a sky that is gaping and yawning
There is a song I woke with on my lips as you sailed your great ship towards the morning
Come on home, the poppies are all grown knee-deep by now
Blossoms all have fallen, and the pollen ruins the plow
Peonies nod in the breeze and while they wetly bow, with
Hydrocephalitic listlessness ants mop up their brow
And everything with wings is restless, aimless, drunk and dour
Butterflies and birds collide at hot, ungodly hours
And my clay-colored motherlessness rangily reclines
Come on home now, all my bones are dolorous with vines
Pa pointed out to me, for the hundredth time tonight
The way the ladle leads to a dirt-red bullet of light
Squint skyward and listen
Loving him, we move within his borders
Just asterisms in the stars' set order
We could stand for a century
Staring, with our heads cocked
In the broad daylight at this thing
Joy, landlocked
In bodies that don't keep
Dumbstruck with the sweetness of being
Till we don't be
Told, take this
And eat this
Told, the meteorite is the source of the light
And the meteor's just what we see
And the meteoroid is a stone that's devoid of the fire that propelled it to thee
And the meteorite's just what causes the light
And the meteor's how it's perceived
And the meteoroid's a bone thrown from the void that lies quiet in offering to thee
"""
COLOR_MODE = "plane"        # "plane" | "linear"
CONTEXT = 0.3
HUE_ROTATION = 0
CHROMA = 0.16               # a touch punchier for flat regions
WIDTH, HEIGHT = 1600, 1370
MARGIN = 80
OUT_DIR = "the_art"


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

    words, vectors = embedder.embed(POEM)
    w2v = {w: v for w, v in zip(words, vectors)}
 
    lines = read_grid(POEM)
    tokens, positions, rc = grid_positions(lines, WIDTH, HEIGHT, margin=MARGIN)
    rows = [r for (r, _c) in rc]
 
    line_vecs = [np.array([w2v[w] for w in line]).mean(0) for line in lines]
    occ = occurrence_vectors(tokens, rows, w2v, line_vecs, context=CONTEXT)
    colors = manifold_colors(occ, mode=COLOR_MODE, hue_rotation=HUE_ROTATION,
                             chroma=CHROMA)
 
    # rectangles for the tile grid
    nlines = len(lines)
    iw, ih = WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN
    row_h = ih / nlines
    rects = []
    for (r, _c), (cx, cy) in zip(rc, positions):
        cell_w = iw / len(lines[r])
        rects.append((cx - cell_w / 2, cy - row_h / 2, cell_w, row_h))
 
    slug = slugify(POEM_NAME)
    styles = {
        "voronoi": voronoi_svg(positions, colors, WIDTH, HEIGHT),
        "lowpoly": lowpoly_svg(positions, colors, WIDTH, HEIGHT),
        "tiles": tiles_svg(rects, colors, WIDTH, HEIGHT),
    }
    overlay = labels_svg(tokens, positions)   # white text + dark halo
    for style, svg in styles.items():
        # 1) art only
        plain = os.path.join(OUT_DIR, f"{slug}_{style}.svg")
        with open(plain, "w") as f:
            f.write(svg)
        # 2) poem name on top + each word over its shape
        labeled = svg[:svg.rindex("</svg>")] + overlay + "</svg>"
        titled = wrap_with_title(labeled, POEM_NAME, WIDTH, HEIGHT)
        titled_path = os.path.join(OUT_DIR, f"{slug}_{style}_titled.svg")
        with open(titled_path, "w") as f:
            f.write(titled)
        print("saved", plain, "and", titled_path)
 
 
if __name__ == "__main__":
    main()