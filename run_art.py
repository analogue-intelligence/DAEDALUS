"""
Art pipeline: poem -> embeddings -> CPPN-NEAT (shape-constrained) -> art SVG.
"""
import os
import re
import html
from poem_cppn import (WordNetEmbedder, HashEmbedder, Evolver, FitnessWeights, render_art_svg)

# POEM_NAME = "Blind Joe Death"  # for title and output filename I wanted to test a Kate Bush poem
# POEM = """
# The globe spins,
#         Dragging collisions of clutches, to the end.
# Blind Joe Death staggers to the instrument
#         And caresses the soft wood of the neck.
# He guides the dizzy fingers through
#         The mist of Melancholy melody.
# Blind Joe Death grins at Fahey
#         And moves the cap up the strings.
#         He stops.
# Blind Joe Death dies.
# He falls onto the round Persian mat
# And swings the needle off the turntable.
#         Fahey sleeps.
# """
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
SHAPE = "square"          # "circle" | "square" | "rect"  (or loop over several)
OUT_DIR = "the_art"
SIZE = 1600
GENERATIONS = 300


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "poem"


def wrap_with_title(art_svg: str, title: str, width: int, height: int, header: int = 76) -> str:
    """
    Put the artwork in a band below a centred poem title (no overlap).
    """
    inner = art_svg[art_svg.index(">") + 1: art_svg.rindex("</svg>")]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height + header}" viewBox="0 0 {width} {height + header}">'
        f'<rect width="{width}" height="{height + header}" fill="#ffffff"/>'
        f'<text x="{width / 2:.0f}" y="{header * 0.6:.0f}" text-anchor="middle" '
        f'fill="#222222" font-family="Georgia, serif" font-style="italic" '
        f'font-size="30">{html.escape(title)}</text>'
        f'<g transform="translate(0,{header})">{inner}</g></svg>'
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        embedder = WordNetEmbedder(dim=8)
        backend = "WordNet"
    except Exception as e:
        print("WordNet unavailable, using hash fallback:", e)
        embedder = HashEmbedder(dim=8)
        backend = "Hash"
    words, vectors = embedder.embed(POEM)
    print(f"{len(words)} words via {backend}")

    ev = Evolver(words, vectors, width=SIZE, height=SIZE, pop_size=200, weights=FitnessWeights(), seed=7, shape=SHAPE, config_path=f"config-{SHAPE}.ini")
    winner, _ = ev.run(generations=GENERATIONS)
    pos, col, sizes = ev.layout_for(winner)

    slug = slugify(POEM_NAME)

    # 1) artwork only
    art = render_art_svg(words, pos, col, sizes, width=SIZE, height=SIZE, vectors=vectors, draw_edges=True, edge_k=1, labels=False, bg="#ffffff")
    art_path = os.path.join(OUT_DIR, f"{slug}_{SHAPE}.svg")
    with open(art_path, "w") as f:
        f.write(art)

    # 2) titled + word labels
    labeled = render_art_svg(words, pos, col, sizes, width=SIZE, height=SIZE, vectors=vectors, draw_edges=True, edge_k=1, labels=True, bg="#ffffff")
    titled = wrap_with_title(labeled, POEM_NAME, SIZE, SIZE)
    titled_path = os.path.join(OUT_DIR, f"{slug}_{SHAPE}_titled.svg")
    with open(titled_path, "w") as f:
        f.write(titled)

    print(f"saved:\n  {art_path}\n  {titled_path}")


if __name__ == "__main__":
    main()