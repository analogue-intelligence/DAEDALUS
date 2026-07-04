"""
End-to-end demo: poem -> WordNet embeddings -> CPPN-NEAT (corpus-free fitness)
-> SVG. Run:  python run_demo.py
"""
import os
import numpy as np
from poem_cppn import WordNetEmbedder, HashEmbedder, Evolver, FitnessWeights, save_svg

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

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

W, H = 900, 700


def main():
    try:
        embedder = WordNetEmbedder(dim=8)
        backend = "WordNet (semantic)"
    except Exception as e:
        print("WordNet unavailable, falling back to hash embeddings:", e)
        embedder = HashEmbedder(dim=8)
        backend = "Hash (non-semantic fallback)"

    words, vectors = embedder.embed(POEM)
    print(f"{len(words)} unique words via {backend}; embedding dim = {vectors.shape[1]}")

    ev = Evolver(words, vectors, width=W, height=H, pop_size=200, weights=FitnessWeights(),  seed=7)
    winner, stats = ev.run(generations=150, snapshot_at=(0,))

    gen0_genome, gen0_fit = ev.snapshots[0]
    fin_genome, fin_fit = ev.snapshots["final"]

    for tag, genome, fit in [("gen0", gen0_genome, gen0_fit),
                             ("final", fin_genome, fin_fit)]:
        pos, col = ev.layout_for(genome)
        save_svg(f"{OUT}/poem_map_{tag}.svg", words, pos, col, width=W, height=H, vectors=vectors, draw_neighbor_edges=True, title=f"{tag}  fitness={fit:.3f}  [{backend}]")
        print(f"  {tag}: fitness={fit:.3f} -> poem_map_{tag}.svg")

    print("done")


if __name__ == "__main__":
    main()
