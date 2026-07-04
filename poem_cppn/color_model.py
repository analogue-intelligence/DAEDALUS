"""
Learned colour: map a word's embedding to an RGB colour, learned from a set of
(word, colour) ASSOCIATION pairs ("anchors") rather than from PCA axes. = FOR NOW I USE THIS BECAUSE IT'S EASIER TO CONTROL THAN PCA

How it works:
1. Take anchor pairs, e.g. fire -> orange-red, sea -> blue, grief -> blue-grey.
2. Embed the anchor words in the SAME space as the poem words (so distances are comparable). For transductive embedders like WordNet this means embedding the
   union of anchors + poem words together, which fit_predict handles.
3. Learn embedding -> RGB from the anchors (kNN by default, or ridge), then predict a colour for every poem word. Words near colour-associated anchors
   inherit that colour; e.g. "river" ~ water/sea/rain -> blue.

The ANCHORS table is small and hand-curated so this runs offline. For the real thing, swap in the NRC Word-Colour Association Lexicon (~14k crowdsourced
pairs) via load_nrc(path). Quality also depends a lot on the embedder: distributional vectors (GloVe / sentence-transformers) capture "fire ~ red" far better than
WordNet's taxonomy does.

Colour assignment methods:
kNN (default):
    For each poem word, find the k nearest anchor words in embedding space.
    Weight each anchor's colour by inverse-distance^power so that the closest anchor dominates. This is non-parametric: it doesn't assume any functional
    form between embedding and colour, just "nearby anchors vote".

    Formula for word i:
        d_j    = ||embed(word_i) - embed(anchor_j)||_2   (Euclidean distance)
        w_j    = 1 / (d_j + eps)^power                   (inverse-distance weight)
        RGB_i  = sum(w_j * RGB_j) / sum(w_j)             (weighted average)

    With power=4 (default), a word that is 2x farther from an anchor than another gets 1/16th the vote weight.

Ridge regression:
    Fit a linear model  RGB = X @ W  where X is the embedding matrix of anchor words (with a bias column appended), and W is solved via L2-regularised
    least squares:
        W = (X^T X + lambda * I)^{-1} X^T Y
    Then predict poem-word colours as  T @ W  where T is the poem embedding matrix (also with bias column). This assumes colour varies linearly across
    embedding space, which is a stronger (and often wrong) assumption than kNN.
"""
from __future__ import annotations
import numpy as np

# anchor table: word -> (R, G, B) in 0..255.
#
# Two categories:
#   - Colour terms (red, blue, ...): ground-truth for what the colour *name* should look like. Any embedder that puts "crimson" near "red" will give
#     crimson a red hue.
#   - Concept/emotion associations (fire, grief, ...): intuitions about what colour a concept "feels like"


ANCHORS = {
    # colour terms
    "red": (220, 30, 30), "crimson": (170, 20, 40), "scarlet": (230, 40, 30),
    "orange": (240, 140, 30), "amber": (240, 170, 40), "gold": (230, 190, 60),
    "yellow": (245, 225, 70), "lime": (160, 210, 60), "green": (60, 160, 70),
    "emerald": (30, 150, 100), "teal": (30, 150, 150), "cyan": (60, 200, 210),
    "azure": (90, 170, 230), "blue": (40, 90, 200), "navy": (25, 40, 110),
    "indigo": (70, 50, 150), "violet": (140, 90, 200), "purple": (130, 60, 160),
    "magenta": (200, 50, 150), "pink": (235, 130, 170), "rose": (220, 90, 120),
    "brown": (120, 75, 45), "tan": (200, 170, 120), "beige": (225, 210, 170),
    "grey": (140, 140, 140), "black": (35, 35, 40), "white": (240, 240, 245),
    "silver": (190, 195, 200), "ivory": (240, 235, 215),
    # concept / emotional associations
    "fire": (235, 80, 30), "flame": (240, 100, 30), "ember": (210, 70, 30),
    "spark": (245, 160, 50), "ash": (120, 120, 125), "smoke": (150, 150, 155),
    "sun": (245, 205, 70), "sunlight": (250, 220, 120), "light": (250, 240, 200),
    "dark": (40, 40, 55), "shadow": (60, 60, 75), "night": (20, 30, 70),
    "dawn": (245, 180, 150), "dusk": (130, 90, 140), "sky": (120, 180, 235),
    "cloud": (210, 215, 225), "sea": (30, 110, 170), "ocean": (20, 90, 160),
    "water": (60, 140, 200), "river": (70, 150, 190), "rain": (110, 140, 180),
    "wave": (50, 130, 180), "forest": (30, 110, 60), "tree": (60, 120, 60),
    "leaf": (90, 160, 70), "grass": (110, 180, 80), "wood": (110, 80, 50),
    "earth": (120, 90, 60), "soil": (100, 70, 45), "stone": (130, 130, 125),
    "rock": (120, 120, 115), "iron": (90, 95, 105), "metal": (140, 145, 155),
    "rust": (160, 80, 40), "blood": (140, 20, 30), "wine": (110, 30, 50),
    "snow": (235, 240, 250), "ice": (200, 225, 240), "frost": (210, 230, 240),
    "storm": (90, 95, 110), "mist": (190, 200, 205), "grief": (90, 100, 130),
    "sorrow": (80, 95, 135), "joy": (250, 210, 70), "love": (220, 70, 110),
    "rage": (200, 30, 30), "calm": (90, 170, 170), "peace": (150, 200, 190),
    "fear": (70, 70, 90),
}


def load_nrc(path: str) -> dict:
    """
    Parse the NRC Word-Colour Association Lexicon (word<TAB>colour) into anchors.

    The NRC lexicon gives a colour *name* per word (e.g., "happy\tyellow"). We resolve those names through ANCHORS to get an RGB triple, and skip
    any colour name we don't have an RGB for.

    This scales the anchor set from ~75 hand-curated entries to ~14k crowdsourced entries, which could improve colour predictions
    (more coverage = shorter distances to the nearest anchor = less colour ambiguity).

    Parameters:
     - path : str - Path to the NRC lexicon file. Expected format: word<TAB>colour_name per line.

    Returns:
     - dict - {word: (r, g, b)} with values in 0..255, one entry per resolvable word in the lexicon.
    """
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1].lower() in ANCHORS:
                out[parts[0].lower()] = ANCHORS[parts[1].lower()]
    return out


class LearnedColorMap:
    """
    Learn a mapping from word embeddings to RGB colours using anchor pairs.

    Parameters: 
     - embedder : object - Any embedder with an embed(text) -> (words, vectors) interface. Must produce embeddings where semantic similarity corresponds to
        vector proximity (GloVe, sentence-transformers, WordNet MDS, ...).
     - anchors : dict or None - {word: (r, g, b)} in 0..255. Defaults to the ANCHORS table.
     - method : str - "knn" (default) or "ridge". See module docstring for the maths.
     - k : int - Number of nearest anchors to consider in kNN mode. Default 6.
     - power : float - Exponent for inverse-distance weighting in kNN. Default 4.0.
     - ridge_lambda : float - L2 regularisation strength for ridge mode. Default 1.0.
    """

    def __init__(self, embedder, anchors: dict | None = None, method: str = "knn", k: int = 6, power: float = 4.0, ridge_lambda: float = 1.0):
        self.embedder = embedder
        self.anchors = anchors or ANCHORS
        self.method = method
        self.k = k
        self.power = power
        self.ridge_lambda = ridge_lambda

    def fit_predict(self, words) -> dict:
        """
        Embed anchors + poem words together, learn anchor->colour, predict all.

        Why the union embedding? => Transductive embedders (like WordNet MDS) compute distances *between the words they're given*, so the anchor
        words and poem words must be embedded in the same run to produce comparable vectors. Even for inductive embedders (GloVe, transformers)
        doing them together costs nothing extra and keeps the interface uniform.

        Returns:
         - dict - {word: (r, g, b)} with values in 0..1 for every word in `words`. Words absent from the embedding vocabulary get the mean anchor colour as a neutral fallback.
        """
        anchor_words = list(self.anchors)
        # embed the union of anchor words and poem words together
        union = sorted(set(anchor_words) | set(words))
        emb_words, emb_vecs = self.embedder.embed(" ".join(union))
        vec = {w: v for w, v in zip(emb_words, emb_vecs)}

        # build the anchor matrix: only anchors that got embedded successfully
        A_words = [w for w in anchor_words if w in vec]
        # (A, d) anchor embeddings
        AX = np.array([vec[w] for w in A_words])
        # (A, 3) anchor RGB, normalised to 0..1    
        AY = np.array([self.anchors[w] for w in A_words]) / 255.0 
        # neutral grey-ish colour for OOV words 
        fallback = AY.mean(axis=0)  

        # build the target matrix: poem words that got embedded
        targets = [w for w in words if w in vec]
        TX = np.array([vec[w] for w in targets]) if targets else np.empty((0, AX.shape[1]))

        # predict colours using chosen method
        if self.method == "ridge":
            pred = self._ridge(AX, AY, TX)
        else:
            pred = self._knn(AX, AY, TX)

        # assemble output dict, clipping RGB to [0, 1]
        out = {w: tuple(np.clip(p, 0, 1)) for w, p in zip(targets, pred)}
        for w in words:  # OOV words (e.g., not in GloVe) -> neutral fallback colour
            out.setdefault(w, tuple(fallback))
        return out

    def _knn(self, AX, AY, TX):
        """
        k-nearest-neighbour colour prediction with inverse-distance weighting.

        For each target word embedding t_i in TX:
            1. Compute Euclidean distance to every anchor: d_{i,j} = ||t_i - a_j||_2
            2. Pick the k closest anchors (argsort on d).
            3. Weight each of the k anchors by: w_j = 1 / (d_{i,j} + eps)^power, where eps=1e-6 prevents division by zero. With power=4,
               a 2x-farther anchor gets 1/16th the weight.
            4. Compute weighted average colour: RGB_i = sum(w_j * RGB_j) / sum(w_j)

        Parameters:
        - AX : ndarray (A, d) - Anchor embeddings.
        - AY : ndarray (A, 3) - Anchor colours in [0, 1].
        - TX : ndarray (T, d) - Target (poem word) embeddings.

        Returns:
        - ndarray (T, 3) - Predicted RGB in [0, 1] for each target word.
        """
        if len(TX) == 0:
            return TX
        # d[i, j] = euclidean distance from target i to anchor j
        d = np.linalg.norm(TX[:, None, :] - AX[None, :, :], axis=2)  # (T, A)
        k = min(self.k, AX.shape[0])
        # idx[i] = indices of the k nearest anchors to target i
        idx = np.argsort(d, axis=1)[:, :k]
        out = np.zeros((len(TX), 3))
        for i in range(len(TX)):
            # distances to k nearest
            dk = d[i, idx[i]]
            # inverse-distance weights
            w = 1.0 / (dk + 1e-6) ** self.power
            # weighted avg of the k nearest anchor colours
            out[i] = (w[:, None] * AY[idx[i]]).sum(0) / (w.sum() + 1e-12)
        return out

    def _ridge(self, AX, AY, TX):
        """
        Ridge regression colour prediction.

        Fits a linear model from anchor embeddings to anchor colours with L2 regularisation, then applies it to target embeddings.

        Model:
            Y = X_aug @ W
        where X_aug = [X | 1] (embedding matrix with a bias column of ones), and W is solved in closed form:
            W = (X_aug^T @ X_aug + lambda * I)^{-1} @ X_aug^T @ Y

        This is standard Tikhonov/ridge regression [REMEMBER INCLUDE CITATION]. The lambda term shrinks weights toward zero, preventing overfitting when the number of anchors
        is small relative to the embedding dimensionality.

        Parameters:
        - AX : ndarray (A, d) - Anchor embeddings.
        - AY : ndarray (A, 3) - Anchor colours in [0, 1].
        - TX : ndarray (T, d) - Target embeddings.

        Returns:
        - ndarray (T, 3) - Predicted RGB in [0, 1] for each target.
        """
        if len(TX) == 0:
            return TX
        # append bias column of ones to anchor embeddings
        X = np.hstack([AX, np.ones((len(AX), 1))])  # (A, d+1)
        d = X.shape[1]
        # solve the ridge normal equations: (X^T X + lambda I) W = X^T Y
        W = np.linalg.solve(X.T @ X + self.ridge_lambda * np.eye(d), X.T @ AY)
        # predict: append bias to targets and multiply
        T = np.hstack([TX, np.ones((len(TX), 1))])  # (T, d+1)
        return T @ W