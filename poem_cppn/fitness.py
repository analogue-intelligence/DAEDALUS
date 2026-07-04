"""
Corpus-free fitness. No labels, no training set => the objective is structural and aesthetic, which is the whole point of the evolutionary/DAEDALUS framing.

The fitness function scores a candidate layout (positions + colours on a canvas) against the word embeddings. It answers: "does this layout look good AND preserve
the semantic structure of the poem?"

There are six terms, each capturing a different quality:

  pos           : Do semantically similar words land near each other?
                  (Pearson correlation between embedding distances and layout distances)
                  MAXIMISE.

  color         : Do similar words get similar colours?
                  (Pearson correlation between embedding distances and colour distances)
                  MAXIMISE.

  color_spread  : Does the palette use a full range, not collapse to one hue?
                  (mean pairwise colour distance, normalised)
                  MAXIMISE.

  coverage      : Do the points fill the target shape (circle/square/rect)?
                  (fraction of grid cells occupied)
                  MAXIMISE.

  spread        : Do the points use the canvas (anti-collapse)?
                  (mean pairwise spatial distance, normalised)
                  MAXIMISE.

  overlap       : Are points piled on top of each other?
                  (fraction of pairs closer than overlap_radius)
                  MINIMISE.

Combined fitness formula:
    F = w_pos * pos
      + w_color * color
      + w_color_spread * cspread
      + w_coverage * coverage
      + w_spread * spread
      - w_overlap * overlap          (subtracted because it's a penalty)

Zero all weights and evaluate by hand for Picbreeder-style interactive evolution.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import pdist


@dataclass
class FitnessWeights:
    """
    Weights for each fitness term.

    Attributes:
    - pos : float - Weight for semantic-position correlation. Default 0.8. Higher -> CPPN is rewarded more for placing similar words nearby.
    - color : float - Weight for semantic-colour correlation. Default 0.4. Higher -> CPPN is rewarded more for colouring similar words alike.
    - color_spread : float - Weight for colour diversity. Default 0.5. Prevents the CPPN from assigning everything the same colour.
    - coverage : float - Weight for spatial coverage of the target shape. Default 1.1. This is the strongest default weight because "fill the shape" is
        the most visually important constraint: a composition that clusters in one corner looks broken regardless of its semantic structure.
    - spread : float - Weight for spatial spread (anti-collapse). Default 0.25. Lower than coverage because coverage already rewards distribution.
    - overlap : float - Penalty weight for overlapping points. Default 0.4. Subtracted from fitness. Prevents words from piling up.
    - overlap_radius : float - Two points closer than this (in pixels) count as overlapping. Default 26.0 (roughly the maximum blob radius in cppn.py).
    - grid : int - Resolution of the coverage grid. Default 8 -> 64 cells. Higher = finer spatial resolution for the coverage check, but
        harder to fill (you need more points to hit every cell).
    """
    pos: float = 0.8
    color: float = 0.4
    color_spread: float = 0.5
    coverage: float = 1.1
    spread: float = 0.25
    overlap: float = 0.4
    overlap_radius: float = 26.0
    grid: int = 8


class Fitness:
    """
    Evaluate a candidate layout against the word embeddings.

    Pre-computes the pairwise embedding distances once (O(N^2)) so that each fitness call only needs to compute layout and colour distances.

    Parameters:
    - vectors : ndarray (N, d) - Standardised word embeddings.
    - weights : FitnessWeights or None - Term weights. None uses defaults.
    - shape : str - Target shape: "circle", "square", or "rect". Determines which grid cells count as "inside" for the coverage term.
    - width, height : int - Canvas dimensions.
    - margin : int - Border margin in pixels.
    """

    def __init__(self, vectors, weights=None, shape="circle", width=820, height=820, margin=60):
        self.w = weights or FitnessWeights()
        self.shape = shape
        self.width, self.height, self.margin = width, height, margin

        # pre-compute pairwise Euclidean distances between embeddings.
        # pdist returns a condensed distance vector of length N*(N-1)/2.
        self.embed_d = pdist(vectors)

        # Ppe-compute centred embedding distances and their norm (for Pearson r).
        # Pearson correlation between two vectors a and b:
        #     r = (a - mean(a)) . (b - mean(b)) / (||a - mean(a)|| * ||b - mean(b)||)
        # we pre-compute (embed_d - mean) and its norm since they don't change.
        self._ed = self.embed_d - self.embed_d.mean()
        self._ed_norm = np.linalg.norm(self._ed) + 1e-12  # +eps to avoid /0

        # build the spatial grid for coverage evaluation
        self._build_grid()

    def _build_grid(self):
        """
        Build the grid used for the coverage fitness term.

        The canvas is divided into an (g x g) grid of cells. Depending on the target shape:
          - "circle": only cells whose centre is inside the inscribed circle count.
                      A cell (i, j) is "inside" if:
                          (i/g - 0.5)^2 + (j/g - 0.5)^2 <= 0.25
                      (i.e., the normalised centre is within radius 0.5 of the
                      normalised centre (0.5, 0.5)).
          - "square" or "rect": all cells count.

        Coverage = (number of inside cells that contain >= 1 point) / (total inside cells)
        """
        g = self.w.grid
        cx, cy = self.width / 2, self.height / 2
        iw, ih = self.width - 2 * self.margin, self.height - 2 * self.margin
        R = min(iw, ih) / 2

        # determine the coordinate space for mapping points to grid cells
        if self.shape in ("circle", "square"):
            # both circle and square use the inscribed region centred on canvas
            self.x0, self.y0, self.span = cx - R, cy - R, 2 * R
            self._iw = self._ih = 2 * R
        else:
            # "rect" uses the full inner rectangle
            self.x0, self.y0 = self.margin, self.margin
            self.span = None
            self._iw, self._ih = iw, ih

        # build normalised cell centres (0..1 in each axis)
        idx = (np.arange(g) + 0.5) / g  # cell centres: 0.0625, 0.1875, ..., 0.9375 for g=8
        gx, gy = np.meshgrid(idx, idx)

        # determine which cells are "inside" the target shape
        if self.shape == "circle":
            # circle test: (x - 0.5)^2 + (y - 0.5)^2 <= 0.25  (radius 0.5 in normalised coords)
            inside = ((gx - 0.5) ** 2 + (gy - 0.5) ** 2) <= 0.25
        else:
            inside = np.ones((g, g), dtype=bool)  # all cells are inside for square/rect

        self._inside = inside
        self._n_inside = int(inside.sum())  # total number of target cells

    def _cell_ids(self, positions):
        """
        Map point positions to grid cell indices.

        For each point (x, y), compute which cell it falls into:
            fx = (x - x0) / inner_width   -> fractional x in [0, 1]
            fy = (y - y0) / inner_height   -> fractional y in [0, 1]
            cx = floor(fx * g)             -> column index, clamped to [0, g-1]
            cy = floor(fy * g)             -> row index, clamped to [0, g-1]

        Returns (row_indices, col_indices) arrays.
        """
        g = self.w.grid
        fx = (positions[:, 0] - self.x0) / self._iw
        fy = (positions[:, 1] - self.y0) / self._ih
        cx = np.clip((fx * g).astype(int), 0, g - 1)
        cy = np.clip((fy * g).astype(int), 0, g - 1)
        return cy, cx  # (row, col) for array indexing

    def _coverage(self, positions):
        """
        Compute what fraction of the target shape's grid cells are occupied.

        Returns a value in [0, 1]. A layout that fills the shape evenly scores close to 1.0; one that clusters in a corner scores much lower.
        """
        g = self.w.grid
        hit = np.zeros((g, g), dtype=bool)
        cy, cx = self._cell_ids(positions)
        hit[cy, cx] = True  # mark each occupied cell
        # count occupied cells that are inside the target shape
        return float((hit & self._inside).sum()) / max(self._n_inside, 1)

    def _corr(self, other):
        """
        Pearson correlation between embedding distances and `other` distances.

        Pearson r measures linear correlation between two vectors:
            r = (a_centered . b_centered) / (||a_centered|| * ||b_centered||)

        where a_centered = a - mean(a) and b_centered = b - mean(b).

        The embedding side (a) is pre-computed in __init__. This method computes the centring and norm for `other` on the fly.

        Returns a value in [-1, 1]:
            +1 = perfect positive correlation (close in embedding -> close in layout)
            -1 = perfect negative correlation (close in embedding -> far in layout)
             0 = no linear relationship
        """
        od = other - other.mean()
        denom = self._ed_norm * (np.linalg.norm(od) + 1e-12)
        return float(self._ed @ od / denom)

    def __call__(self, positions, colors, width, height):
        """
        Evaluate the fitness of a candidate layout.

        Parameters:
        - positions : ndarray (N, 2) - Word positions on the canvas.
        - colors : ndarray (N, 3) - Word colours in [0, 1].
        - width, height : int - Canvas dimensions.

        Returns:
        - float - Scalar fitness score. Higher is better. Typical range depends on the weights but is roughly -1 to +3 for default weights.

        Fitness formula:
            F = w_pos * corr(embed_dist, position_dist)
              + w_color * corr(embed_dist, colour_dist)
              + w_color_spread * min(mean_colour_dist / 0.6, 1)
              + w_coverage * fraction_of_shape_filled
              + w_spread * min(mean_position_dist / (0.4 * diagonal), 1)
              - w_overlap * fraction_of_pairs_closer_than_radius
        """
        # pairwise Euclidean distances between positions (condensed vector)
        pos_d = pdist(positions)

        # degenerate check: if all points collapsed to one spot, penalise heavily
        if pos_d.std() < 1e-6:
            return -1.0

        # pairwise Euclidean distances between colours
        col_d = pdist(colors)

        # canvas diagonal (used to normalise spread)
        diag = (width ** 2 + height ** 2) ** 0.5

        # term 1: Position-embedding correlation
        # Do semantically similar words end up spatially close?
        pos_term = self._corr(pos_d)

        # term 2: Colour-embedding correlation 
        # Do semantically similar words get similar colours?
        col_term = self._corr(col_d) if col_d.std() > 1e-6 else 0.0

        # term 3: Colour spread 
        # mean_colour_dist / 0.6, capped at 1.0.
        # The 0.6 threshold is empirical: mean pairwise colour distance of ~0.6 corresponds to a well-spread palette in RGB space. Values above 0.6
        # don't get extra reward (capped at 1).
        cspread = min(col_d.mean() / 0.6, 1.0)

        # term 4: Spatial spread 
        # mean_position_dist / (0.4 * diagonal), capped at 1.0.
        # The 0.4 * diagonal threshold means points must on average be ~40% of the diagonal apart to get full spread credit.
        spread = min(pos_d.mean() / (0.4 * diag), 1.0)

        # term 5: Coverage 
        # fraction of target-shape grid cells that contain at least one point.
        coverage = self._coverage(positions)

        # term 6: Overlap penalty 
        # Fraction of all point pairs that are closer than overlap_radius pixels. This is subtracted (penalty) to discourage piling words on top of each other.
        overlap = float((pos_d < self.w.overlap_radius).mean())

        # combined fitness
        return (self.w.pos * pos_term
                + self.w.color * col_term
                + self.w.color_spread * cspread
                + self.w.coverage * coverage
                + self.w.spread * spread
                - self.w.overlap * overlap)