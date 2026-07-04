"""
Anchor-free colour: derive each word's colour from the *shape* of the embedding cloud, mapped into a perceptual colour space (Oklab) so that semantic similarity
becomes perceptual similarity and the palette stays vivid and evenly spread.

No colour table, no external grounding. The one remaining choice is a global hue rotation -- an aesthetic dial, not a source of grounding (there is no
anchor-free way to make "fire" specifically red; that information isn't in text).

Why Oklab?
Oklab is a perceptually uniform colour space designed by Björn Ottosson (2020). "Perceptually uniform" means that equal numerical distances in Oklab correspond
to equal perceived colour differences. This matters here because we want semantic distance (in embedding space) to map to perceptual colour distance (as the viewer
sees it). RGB and HSV are NOT perceptually uniform: e.g., a step of 0.1 in hue near green looks much smaller than the same step near yellow.

Oklab has three axes:
    L : lightness (0 = black, 1 = white)
    a : green-red opponent axis
    b : blue-yellow opponent axis

Two modes:
"plane" mode:
    1. PCA reduces embeddings to 3 components (c0, c1, c2).
    2. The first two components define a "semantic plane":
           angle = atan2(c1, c0)  -> hue rotation in Oklab
           radius = sqrt(c0^2 + c1^2) -> chroma (saturation)
       Words that are semantically related (nearby in embedding space) will have similar angles, so they get similar hues. Words that are far
       from the centroid get higher chroma (more saturated).
    3. The third component (c2) controls lightness.
    4. Convert (L, a, b) in Oklab -> sRGB.

"linear" mode:
    1. PCA reduces embeddings to 3 components.
    2. Map them directly to Oklab axes:
           L = base_lightness + 0.14 * c0_normalised
           a = chroma * c1_normalised
           b = chroma * c2_normalised
    3. Apply hue rotation (a rotation matrix in the a-b plane).
    4. Convert Oklab -> sRGB.

    Simpler and smoother than "plane" (no polar step), but the hue assignment is less interpretable.

Both return RGB in [0, 1].
"""
from __future__ import annotations
import numpy as np


def _oklab_to_srgb(L, a, b):
    """
    Convert Oklab colour values to sRGB.

    Oklab -> linear sRGB pipeline (from Björn Ottosson's specification):

    Step 1: Oklab (L, a, b) -> intermediate cone-like responses (l_, m_, s_):
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b

    Step 2: Undo the cube-root non-linearity:
        l = l_^3,  m = m_^3,  s = s_^3

    Step 3: Linear cone responses -> linear sRGB via the 3x3 matrix:
        r_linear =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g_linear = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_linear = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    Step 4: Clip negative linear RGB values to 0 (gamut clipping).

    Step 5: Linear sRGB -> sRGB (gamma correction):
        if c_linear <= 0.0031308:
            c_srgb = 12.92 * c_linear
        else:
            c_srgb = 1.055 * c_linear^(1/2.4) - 0.055

    Step 6: Clip sRGB to [0, 1].

    Parameters:
    - L, a, b : ndarray - Oklab values. L in ~[0, 1], a and b in ~[-0.5, 0.5].

    Returns:
    - ndarray (..., 3) - sRGB values clipped to [0, 1].
    """
    # step 1: Oklab -> intermediate responses
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # step 2: undo cube root
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3

    # step 3: cone responses -> linear sRGB
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    # step 4: clip negative values (gamut clipping)
    lin = np.clip(np.stack([r, g, bb], axis=-1), 0.0, None)

    # step 5: apply sRGB gamma curve
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.power(lin, 1 / 2.4) - 0.055)

    # step 6: clip to valid range
    return np.clip(srgb, 0.0, 1.0)


def _pca(vectors, k=3):
    """
    Compute top-k PCA projections of the embedding matrix.

    Steps:
        1. Centre: X = vectors - mean(vectors)
        2. SVD: X = U @ S @ V^T
        3. Project: comps = X @ V^T[:k]^T    (N x k matrix)

    If the embedding dimensionality is less than k, pad with zero columns.

    Parameters:
    - vectors : ndarray (N, d) - Embedding matrix.
    - k : int - Number of principal components. Default 3.

    Returns:
    - ndarray (N, k)- Projections onto the top-k PCA axes.
    """
    X = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    comps = X @ Vt[:k].T
    if comps.shape[1] < k:  # pad if fewer dims than k
        comps = np.hstack([comps, np.zeros((len(comps), k - comps.shape[1]))])
    return comps


def _robust01(x):
    """
    Normalisation to [0, 1] using 5th/95th percentiles.

    Formula:
        lo = percentile(x, 5)
        hi = percentile(x, 95)
        result = clip((x - lo) / (hi - lo), 0, 1)

    Using percentiles instead of min/max prevents outliers from compressing the range. The 5%/95% choice means ~10% of values will be clipped to
    0 or 1, which is acceptable for colour mapping.
    """
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _robust_signed(x):
    """
    Robust normalisation to [-1, 1] using 5th/95th percentiles.

    Formula:
        lo = percentile(x, 5)
        hi = percentile(x, 95)
        result = clip(2 * (x - lo) / (hi - lo) - 1,  -1, 1)

    This is _robust01 rescaled from [0, 1] to [-1, 1]. Used for Oklab a/b axes which are signed (green-red, blue-yellow).
    """
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip(2 * (x - lo) / (hi - lo + 1e-9) - 1, -1, 1)


def manifold_colors(vectors, mode="plane", hue_rotation=0.0, chroma=0.20, lightness=0.62, light_from_pc=True):
    """
    Derive colours from the embedding manifold structure in Oklab space.

    Parameters:
    - vectors : ndarray (N, d) - Embedding vectors (one per word occurrence, NOT per word type).
    - mode : str - "plane" or "linear". See module docstring.
    - hue_rotation : float - Global hue rotation in degrees. A pure aesthetic dial: rotates all colours around the Oklab a-b plane without changing their relationships.
    - chroma : float - Base Oklab chroma (distance from the neutral axis). Default 0.20.  Higher = more saturated colours overall.
    - lightness : float -  Base Oklab lightness. Default 0.62. Only used directly in "linear" mode; "plane" mode derives L from PC3.
    - light_from_pc : bool - In "plane" mode, whether to derive lightness from the 3rd principal component (True, default) or use a fixed lightness value.

    Returns:
    - ndarray (N, 3) - sRGB colours in [0, 1], one per input vector.
    """
    comps = _pca(vectors, k=3)
    hr = np.deg2rad(hue_rotation)  # convert degrees to radians

    if mode == "plane":
        c0, c1, c2 = comps[:, 0], comps[:, 1], comps[:, 2]

        # Hue: angle in the semantic plane (PC1, PC2) 
        # H = atan2(c1, c0) + hue_rotation
        # words with similar (c0, c1) projections get similar hues.
        H = np.arctan2(c1, c0) + hr

        # Chroma: distance from centroid in the semantic plane 
        # r = sqrt(c0^2 + c1^2)
        # normalise r to [0, 1] robustly, then scale to chroma range.
        # outlier words (far from centroid) get higher saturation.
        r = np.sqrt(c0 ** 2 + c1 ** 2)
        rn = _robust01(r)
        # C ranges from chroma * 0.45 to chroma * 1.0
        C = chroma * (0.45 + 0.55 * rn)

        # Lightness: from PC3 (if enabled) or fixed 
        # L ranges from 0.55 to 0.77 (always visible, never too dark or bright)
        L = (0.55 + 0.22 * _robust01(c2)) if light_from_pc else np.full_like(c0, lightness)

        # convert polar Oklab (L, C, H) to Cartesian Oklab (L, a, b)
        a, b = C * np.cos(H), C * np.sin(H)
        return _oklab_to_srgb(L, a, b)

    # linear mode 
    # PC1 -> lightness variation
    L = lightness + 0.14 * _robust_signed(comps[:, 0])
    # PC2, PC3 -> Oklab a, b axes (before hue rotation)
    a0 = chroma * _robust_signed(comps[:, 1])
    b0 = chroma * _robust_signed(comps[:, 2])

    # apply hue rotation as a 2D rotation matrix in the (a, b) plane:
    #   [a]   [cos(hr)  -sin(hr)] [a0]
    #   [b] = [sin(hr)   cos(hr)] [b0]
    ca, sa = np.cos(hr), np.sin(hr)
    a = ca * a0 - sa * b0
    b = sa * a0 + ca * b0
    return _oklab_to_srgb(L, a, b)


def occurrence_vectors(tokens, row_of_token, word_vec, line_vecs, context=0.0):
    """
    Build one embedding vector per *occurrence* (so duplicates can differ by line).

    The same word appearing in different lines of a poem might carry different semantic weight depending on its context. This function optionally blends
    each word's type-level embedding with its line's mean embedding.

    Formula for each occurrence:
        occ_vec = (1 - context) * word_vec[token] + context * line_vec[row]

    - context = 0.0 (default): every occurrence of "the" gets the same vector. Colour depends only on the word itself.
    - context = 0.5: the word's vector is a 50/50 blend of its base meaning and its line's average meaning. "the" in a fire-themed line might shade
      warmer than "the" in a sea-themed line.
    - context = 1.0: colour depends entirely on the line context, not the word.

    Parameters:
    - tokens : list[str] - Tokens in reading order (may have duplicates).
    - row_of_token : list[int] - Row index for each token (which line it's on).
    - word_vec : dict - {word: ndarray (d,)} type-level embedding for each unique word.
    - line_vecs : dict or list - {row: ndarray (d,)} mean embedding of all words in that line.
    - context : float - Blending weight in [0, 1]. Default 0.0 (no context influence).

    Returns:
    - ndarray (N, d) - One vector per occurrence.
    """
    occ = []
    for tok, r in zip(tokens, row_of_token):
        base = word_vec[tok]
        occ.append((1 - context) * base + context * line_vecs[r] if context else base)
    return np.array(occ)