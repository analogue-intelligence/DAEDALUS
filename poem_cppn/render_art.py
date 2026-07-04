"""
Art renderer: turn a layout into a composition rather than a node-link diagram.

Visual elements:
  - Each word becomes a soft radial colour-blob (size driven by the CPPN's 6th output). Two concentric circles per word: a large soft halo (radial gradient fading to
    transparent) and a small opaque core. This gives a glowing, painterly look.
  - Connections are curved gradient threads whose stroke blends from one endpoint's colour to the other (low opacity, so they read as flow, not graph edges).
  - White background; optional faint word labels above each blob.


SVG structure (layers, back to front):
    1. Background rect
    2. <defs>: linear gradients for edges + radial gradients for blobs
    3. Edge layer: quadratic Bézier curves connecting semantic neighbours
    4. Blob layer: halo circles + core circles
    5. Label layer (optional): word text above each blob
"""
from __future__ import annotations
import html
import math
import numpy as np


def _rgb(c) -> str:
    """
    \Convert [r, g, b] in [0, 1] to an SVG rgb() string.
    """
    return f"rgb({int(round(c[0]*255))},{int(round(c[1]*255))},{int(round(c[2]*255))})"


def _knn_edges(vectors, k=1):
    """
    Find k-nearest-neighbour edges in embedding space.

    For each word i, find the k closest words in embedding space (by Euclidean distance) and add an edge (i, j). Edges are deduplicated and stored as
    (min(i,j), max(i,j)) to avoid drawing the same connection twice.

    Parameters:
    - vectors : ndarray (N, d) - Embedding vectors.
    - k : int - Number of nearest neighbours per word. Default 1 (each word connects to its single closest semantic neighbour).

    Returns:
    - list of (int, int) - Sorted list of unique undirected edges.
    """
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(vectors))
    np.fill_diagonal(D, np.inf)
    edges = set()
    for i in range(len(vectors)):
        for j in np.argsort(D[i])[:k]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def render_art_svg(words, positions, colors, sizes=None, *, width=820, height=820, vectors=None, draw_edges=True, edge_k=1, labels=False,
                   bg="#ffffff", curve=0.18, edge_opacity=0.45, blob_opacity=0.85, title=None) -> str:
    """
    Render a word layout as an artistic composition SVG.

    Parameters:
    - words : list[str] - Word tokens.
    - positions : ndarray (N, 2) - Pixel positions.
    - colors : ndarray (N, 3) - Colours in [0, 1].
    - sizes : ndarray (N,) or None - Blob radius per word (from CPPN). Default: uniform 16px.
    - width, height : int - Canvas dimensions. Default 820x820.
    - vectors : ndarray (N, d) or None - Embeddings (needed for edge computation).
    - draw_edges : bool - Whether to draw gradient threads between neighbours. Default True.
    - edge_k : int - k for k-NN edge computation. Default 1.
    - labels : bool - Whether to show word labels above blobs. Default False.
    - bg : str - Background colour. Default "#ffffff" (white).
    - curve : float - Controls how much the edge curves bow. Default 0.18. The control point is offset perpendicular to the edge by
        curve * edge_length. Higher = more curved arcs.
    - edge_opacity : float - Opacity of the gradient threads. Default 0.45.
    - blob_opacity : float - Opacity of the blob cores and halos. Default 0.85.
    - title : str or None - Optional title text in the top-left corner.

    Returns:
    - str - Complete SVG markup.
    """
    n = len(words)
    if sizes is None:
        sizes = np.full(n, 16.0)

    defs, edge_layer, blob_layer, label_layer = [], [], [], []

    # edge layer: curved gradient threads (drawn underneath blobs) 
    if draw_edges and vectors is not None and n > 1:
        for k, (i, j) in enumerate(_knn_edges(vectors, edge_k)):
            (x1, y1), (x2, y2) = positions[i], positions[j]

            # define a linear gradient along the edge direction so the thread smoothly transitions from word i's colour to word j's colour.
            defs.append(
                f'<linearGradient id="e{k}" gradientUnits="userSpaceOnUse" '
                f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">'
                f'<stop offset="0%" stop-color="{_rgb(colors[i])}"/>'
                f'<stop offset="100%" stop-color="{_rgb(colors[j])}"/>'
                f'</linearGradient>')

            # compute the Bézier control point for an organic curve.
            # the control point is the midpoint of the edge, pushed perpendicular to the edge direction by `curve` times the edge vector.
            #
            # given edge vector (dx, dy), the perpendicular is (-dy, dx).
            # Control point:
            #     cx = (x1+x2)/2 - dy * curve
            #     cy = (y1+y2)/2 + dx * curve
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            cxp, cyp = mx - dy * curve, my + dx * curve

            # quadratic Bézier: M(start) Q(control end)
            edge_layer.append(
                f'<path d="M{x1:.1f},{y1:.1f} Q{cxp:.1f},{cyp:.1f} {x2:.1f},{y2:.1f}" '
                f'fill="none" stroke="url(#e{k})" stroke-width="2.2" '
                f'stroke-opacity="{edge_opacity}" stroke-linecap="round"/>')

    # blob layer: radial gradient halos + opaque cores 
    for i in range(n):
        col = _rgb(colors[i])

        # define a radial gradient for the halo effect:
        #   Centre (0%): full colour at blob_opacity
        #   Middle (45%): same colour at half opacity (soft falloff)
        #   Edge (100%): fully transparent
        defs.append(
            f'<radialGradient id="b{i}" cx="50%" cy="50%" r="50%">'
            f'<stop offset="0%" stop-color="{col}" stop-opacity="{blob_opacity}"/>'
            f'<stop offset="45%" stop-color="{col}" stop-opacity="{blob_opacity*0.5:.3f}"/>'
            f'<stop offset="100%" stop-color="{col}" stop-opacity="0"/>'
            f'</radialGradient>')

        x, y = positions[i]
        halo = sizes[i] * 2.2  # halo radius = 2.2x the core size

        #large transparent halo (the glow)
        blob_layer.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{halo:.1f}" fill="url(#b{i})"/>')
        # small opaque core (the solid centre)
        blob_layer.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{sizes[i]*0.42:.1f}" '
            f'fill="{col}" fill-opacity="0.95"/>')

        # optional word label above the halo
        if labels:
            label_layer.append(
                f'<text x="{x:.1f}" y="{y - halo - 3:.1f}" fill="#444" font-size="11" '
                f'text-anchor="middle" font-family="Georgia, serif" '
                f'fill-opacity="0.7">{html.escape(words[i])}</text>')

    #assemble SVG 
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        "<defs>", *defs, "</defs>",
        *edge_layer,   # edges underneath
        *blob_layer,   # blobs on top of edges
        *label_layer,  # labels on top of everything
    ]
    if title:
        parts.append(
            f'<text x="16" y="26" fill="#999" font-size="13" '
            f'font-family="monospace">{html.escape(title)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def save_art_svg(path, *args, **kwargs):
    """
    Render art SVG and write to a file.

    Parameters:
    - path : str - Output file path.
    *args, **kwargs - Forwarded to render_art_svg().
    """
    with open(path, "w") as f:
        f.write(render_art_svg(*args, **kwargs))