"""
Render a layout (words + positions + colours) to a standalone SVG string.

This is the "diagram" renderer: words are placed as coloured text on a dark background. Optionally draws a faint line from each word to its nearest
*semantic* neighbour (nearest in embedding space, not screen space). If the CPPN learned a good projection, those lines stay short - a quick visual check
on whether semantics survived the mapping.

Use this for debug/inspection. For the art output, use render_art.py instead.
"""
from __future__ import annotations
import html
import numpy as np


def _rgb(c) -> str:
    """
    Convert a colour array [r, g, b] in [0, 1] to an SVG-compatible rgb() string.

    Example: [0.5, 0.0, 1.0] -> "rgb(127,0,255)"
    """
    return f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"


def render_svg(words, positions, colors, width=900, height=700, vectors=None, draw_neighbor_edges=True, bg="#0e0e12", font_size=20, title=None) -> str:
    """
    Render a word layout as a diagnostic SVG.

    Parameters:
    - words : list[str] - Word tokens to display.
    - positions : ndarray or list (N, 2) - Pixel coordinates for each word.
    - colors : ndarray or list (N, 3) - RGB colours in [0, 1] for each word.
    -  width, height : int - SVG canvas dimensions.
    - vectors : ndarray (N, d) or None - Embedding vectors. If provided and draw_neighbor_edges is True, nearest-neighbour lines are drawn.
    - draw_neighbor_edges : bool - Whether to draw lines connecting each word to its nearest semantic neighbour (nearest in embedding space). Default True.
    - bg : str - Background colour. Default "#0e0e12" (near-black).
    - font_size : int - Text size in pixels. Default 20.
    - title : str or None - Optional title text shown in the top-left corner.

    Returns:
    - str - Complete SVG markup.

    Nearest-neighbour edges:
    For each word i, find:
        j* = argmin_{j != i} ||embed_i - embed_j||_2
    Then draw a faint white line from position_i to position_j*.

    These edges visualise whether the CPPN preserved semantic proximity: if similar words are placed nearby, the lines will be short. Long lines
    crossing the canvas indicate that the CPPN failed to cluster related words.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Georgia, serif">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
    ]

    # draw nearest-neighbour edges (underneath the text)
    if draw_neighbor_edges and vectors is not None and len(words) > 1:
        from scipy.spatial.distance import squareform, pdist
        # D[i,j] = Euclidean distance between embeddings i and j
        D = squareform(pdist(vectors))
        np.fill_diagonal(D, np.inf)  # exclude self-distances
        # nn[i] = index of i's nearest neighbour in embedding space
        nn = D.argmin(axis=1)
        for i, j in enumerate(nn):
            (x1, y1), (x2, y2) = positions[i], positions[j]
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#ffffff" stroke-opacity="0.10" stroke-width="1"/>')

    # draw word labels
    for w, (x, y), c in zip(words, positions, colors):
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{_rgb(c)}" font-size="{font_size}" '
            f'text-anchor="middle" dominant-baseline="middle">{html.escape(w)}</text>')

    # optional title in the top-left
    if title:
        parts.append(
            f'<text x="16" y="28" fill="#ffffff" fill-opacity="0.5" font-size="15" '
            f'font-family="monospace">{html.escape(title)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def save_svg(path: str, *args, **kwargs):
    """
    Render to SVG and write to a file.

    Parameters:
    - path : str - Output file path.
    *args, **kwargs - Forwarded to render_svg().
    """
    with open(path, "w") as f:
        f.write(render_svg(*args, **kwargs))