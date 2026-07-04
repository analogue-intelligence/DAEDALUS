"""
Reading-order colour field.

Lay the poem out on an invisible grid (line = row, word index = column), give each word a colour from its meaning, then blend all the colours into one continuous
field (normalised Gaussian splatting). No dots, no edges, no visible grid, just the poem read as a map of colour.

This module is deterministic: positions come from reading order, colours from the embedding. It does NOT use the CPPN/NEAT machinery (there are no positions to
evolve here).

Pipeline:
    1. read_grid()       : parse poem -> list of lines, each a list of tokens
    2. grid_positions()  : lay tokens on a uniform grid (row = line, col = word index)
    3. semantic_colors() : embed tokens, PCA(3), map to HSV -> vivid colours
    4. blend_field()     : Gaussian-splat the coloured points into a continuous field
    5. field_to_svg()    : render the field as a grid of SVG rects
    6. labels_svg()      : overlay word labels with halo effect for legibility
"""
from __future__ import annotations
import re
import colorsys
import numpy as np

# matches alphabetic words and contractions (e.g., "don't")
_TOKEN_RE = re.compile(r"[A-Za-z']+")


def read_grid(poem: str):
    """
    Parse a poem into a list of lines, each a list of lowercase tokens.

    Preserves reading order and line structure (blank lines are dropped). Unlike embed.tokenize, this keeps duplicate words because each occurrence
    occupies its own grid position.

    Parameters:
    - poem : str - Raw poem text with newlines.

    Returns:
    - list[list[str]] - E.g., [["the", "sun", "sets"], ["over", "the", "sea"]]
    """
    lines = []
    for raw in poem.splitlines():
        toks = [w.lower() for w in _TOKEN_RE.findall(raw)]
        if toks:
            lines.append(toks)
    return lines


def grid_positions(lines, width, height, margin=70):
    """
    Assign each token a position on a uniform reading-order grid.

    The grid has len(lines) rows. Within each row, tokens are evenly spaced across the available width:
        y_row = margin + (row + 0.5) * (inner_height / n_rows)
        x_col = margin + (col + 0.5) * (inner_width / n_cols_in_row)

    The +0.5 centres each token within its grid cell.

    Parameters:
    - lines : list[list[str]] - Output of read_grid().
    - width, height : int - Canvas dimensions.
    - margin : int - Border margin in pixels. Default 70.

    Returns:
    - tokens : list[str] - All tokens in reading order (flattened).
    - positions : ndarray (N, 2) - (x, y) pixel coordinates for each token.
    - rc : list of (row, col) - Row and column index for each token (useful for context-dependent colouring).
    """
    tokens, positions, rc = [], [], []
    n_rows = len(lines)
    iw, ih = width - 2 * margin, height - 2 * margin  # inner dimensions
    for r, line in enumerate(lines):
        y = margin + (r + 0.5) * (ih / n_rows)
        L = len(line)
        for c, tok in enumerate(line):
            x = margin + (c + 0.5) * (iw / L)
            tokens.append(tok)
            positions.append((x, y))
            rc.append((r, c))
    return tokens, np.array(positions), rc


def semantic_colors(words, vectors):
    """
    Map d-dim embeddings to vivid RGB via PCA(3) -> HSV colour space.

    Algorithm:
        1. Centre the embedding matrix (subtract mean across words).
        2. Compute SVD:  X = U @ S @ V^T
           Take the top 3 right-singular vectors (V^T[:3]) as principal directions.
        3. Project each word onto these 3 axes:
               comps = X_centred @ V^T[:3]^T    -> (N, 3) matrix
        4. Normalise each component to [0, 1] using robust percentile scaling:
               lo = 5th percentile,  hi = 95th percentile
               norm_j = clip((comp_j - lo) / (hi - lo), 0, 1)
           Using percentiles instead of min/max prevents a single outlier from compressing the colour range of all other words.
        5. Map the three normalised components to HSV:
               H = norm_0                        (hue: 0..1 = full colour wheel)
               S = 0.45 + 0.45 * norm_1          (saturation: always >= 0.45, never washed out)
               V = 0.62 + 0.33 * norm_2          (value/brightness: always >= 0.62, never too dark)
           Then convert HSV -> RGB.

    The saturation and value floors (0.45, 0.62) ensure that all colours are vivid and legible. The hue uses the full wheel so the palette is maximally diverse.

    Parameters:
    - words : list[str] - Token list (same order as vectors).
    - vectors : ndarray (N, d) -Standardised embedding vectors.

    Returns:
    - dict - {word: (r, g, b)} with values in [0, 1].
    """
    # centre the embedding matrix
    X = vectors - vectors.mean(axis=0, keepdims=True)
    # SVD: top-3 principal directions
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    comps = X @ Vt[:3].T  # (N, 3) projections onto top 3 PCA axes

    # robust normalisation: clip to 5th-95th percentile then scale to [0, 1]
    lo = np.percentile(comps, 5, axis=0)
    hi = np.percentile(comps, 95, axis=0)
    norm = np.clip((comps - lo) / (hi - lo + 1e-9), 0, 1)

    out = {}
    for w, (h, s, v) in zip(words, norm):
        # map normalised PCA components to HSV with guaranteed vividness
        rgb = colorsys.hsv_to_rgb(h, 0.45 + 0.45 * s, 0.62 + 0.33 * v)
        out[w] = rgb
    return out


def position_colors(positions, width, height, margin=70):
    """
    Alternative colour mode: colour purely from grid position (spatial gradient).

    Each word's colour depends only on where it sits on the canvas, not on its meaning. This creates a smooth gradient across the composition.

    Mapping:
        H (hue)       = (x - margin) / inner_width       -> full colour wheel left to right
        S (saturation) = 0.55                              -> fixed, vivid
        V (value)      = 0.55 + 0.35 * (1 - y_normalised) -> brighter at top, darker at bottom

    Parameters:
    - positions : list or ndarray of (x, y) - Pixel positions of each word.
    - width, height : int - Canvas dimensions.
    - margin : int - Border margin.

    Returns:
    - list of (r, g, b) - Colours in [0, 1] for each position.
    """
    iw, ih = width - 2 * margin, height - 2 * margin
    cols = []
    for x, y in positions:
        h = (x - margin) / iw              # hue: 0 at left edge, 1 at right
        v = 0.55 + 0.35 * (1 - (y - margin) / ih)  # brighter at top
        cols.append(colorsys.hsv_to_rgb(h % 1.0, 0.55, v))
    return cols


def blend_field(positions, colors, width, height, cell=12, sigma=None):
    """
    Blend point colours into a continuous field via normalised Gaussian splatting.

    Each word "emits" a Gaussian blob of its colour. At every pixel (grid cell), the final colour is the weighted average of all words' colours, where the
    weight is the Gaussian kernel evaluated at the distance from that pixel to each word's position.

    Formula for grid point g:
        weight_{g,i} = exp( -||g - p_i||^2 / (2 * sigma^2) )
        colour_g     = sum_i(weight_{g,i} * colour_i) / sum_i(weight_{g,i})

    This is normalised Gaussian splatting (a.k.a. Nadaraya-Watson kernel regression):
    weights sum to 1 at every point, so the field is a smooth interpolation of the input colours. Near a word, its colour dominates; between two words, their
    colours blend smoothly.

    Sigma (kernel bandwidth) controls how far each word's colour influence reaches:
        - Small sigma: sharp, localised colour patches
        - Large sigma: smooth, blurry field

    Default sigma heuristic:
        sigma = 0.85 * sqrt(canvas_area / N) / 1.4
    This scales with the average "territory" per word, so fields look similar regardless of word count or canvas size.

    Parameters:
    - positions : ndarray or list (N, 2) -Word positions.
    - colors : ndarray or list (N, 3) - Word colours in [0, 1].
    - width, height : int - Canvas dimensions.
    - cell : int - Grid cell size in pixels. Default 12. The field is computed at (width/cell) x (height/cell) resolution.
    - sigma : float or None - Gaussian kernel bandwidth. None uses the heuristic above.

    Returns:
    - ndarray (res_h, res_w, 3) - The blended colour field, RGB in [0, 1].
    """
    P = np.asarray(positions, float)   # (N, 2)
    C = np.asarray(colors, float)      # (N, 3)

    # compute grid resolution (number of cells in each dimension)
    res_w, res_h = max(2, round(width / cell)), max(2, round(height / cell))

    # centre of each grid cell (in pixel coordinates)
    xs = (np.arange(res_w) + 0.5) * (width / res_w)
    ys = (np.arange(res_h) + 0.5) * (height / res_h)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M, 2) where M = res_w * res_h

    # default sigma: scales with average territory per word
    if sigma is None:
        sigma = 0.85 * np.sqrt((width * height) / max(len(P), 1)) / 1.4

    # Squared Euclidean distances: d2[m, i] = ||grid_m - P_i||^2
    d2 = ((grid[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)  # (M, N)

    # Gaussian weights: w[m, i] = exp(-d2[m,i] / (2*sigma^2))
    w = np.exp(-d2 / (2 * sigma ** 2))  # (M, N)

    # normalised weighted average: field[m] = sum_i(w[m,i] * C[i]) / sum_i(w[m,i])
    field = (w @ C) / (w.sum(axis=1, keepdims=True) + 1e-12)  # (M, 3)

    return field.reshape(res_h, res_w, 3)


def field_to_svg(field, width, height, header_inner=""):
    """
    Render the blended colour field as a grid of SVG rects.

    Each cell in the field array becomes an SVG <rect> element filled with the corresponding colour. Rects overlap by 1 pixel (width/height + 1) to prevent
    gaps between cells that some SVG renderers show.

    Parameters:
    - field : ndarray (res_h, res_w, 3) - Colour field from blend_field(), RGB in [0, 1].
    - width, height : int - Canvas dimensions (the SVG viewport size).
    - header_inner : str - Extra SVG content to inject (e.g., labels from labels_svg()).

    Returns:
    - str - SVG markup.
    """
    res_h, res_w, _ = field.shape
    cw, ch = width / res_w, height / res_h  # pixel size of each cell
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}">']
    rects = []
    for i in range(res_h):
        y = i * ch
        for j in range(res_w):
            r, g, b = (np.clip(field[i, j], 0, 1) * 255).astype(int)
            # +1 to width/height prevents hairline gaps between adjacent rects
            rects.append(
                f'<rect x="{j*cw:.1f}" y="{y:.1f}" width="{cw+1:.1f}" '
                f'height="{ch+1:.1f}" fill="#{r:02x}{g:02x}{b:02x}"/>')
    parts.extend(rects)
    parts.append(header_inner)  # labels or other overlays go on top
    parts.append("</svg>")
    return "\n".join(parts)


def labels_svg(tokens, positions, font_size=15):
    """
    Generate SVG text labels with a dark halo for legibility over any background.

    Each word is rendered twice at the same position:
        1. First pass: dark stroke (no fill) — creates the halo/outline.
        2. Second pass: white fill (no stroke) — the readable text on top.

    This guarantees legibility regardless of the background colour beneath.

    Parameters:
    - tokens : list[str] - Word tokens to display.
    - positions : list or ndarray of (x, y) - Pixel positions for each token.
    - font_size : int - Font size in pixels. Default 15.

    Returns:
    - str -  SVG <text> elements (no enclosing <svg> tag — meant to be injected into another SVG via field_to_svg's header_inner parameter).
    """
    import html
    out = []
    for tok, (x, y) in zip(tokens, positions):
        t = html.escape(tok)
        common = (f'x="{x:.1f}" y="{y:.1f}" font-size="{font_size}" '
                  f'text-anchor="middle" dominant-baseline="middle" '
                  f'font-family="Georgia, serif"')
        # pass 1: dark halo (thick stroke, no fill)
        out.append(f'<text {common} fill="none" stroke="#000000" '
                   f'stroke-width="3" stroke-opacity="0.45">{t}</text>')
        # pass 2: white text on top
        out.append(f'<text {common} fill="#ffffff">{t}</text>')
    return "\n".join(out)