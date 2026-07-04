"""
Poem -> per-word embedding vectors.

The CPPN maps each word's embedding vector to (x, y, r, g, b).

Backends:
WordNetEmbedder        : offline, semantic (path-similarity -> MDS).
HashEmbedder           : offline, deterministic, NON-semantic.
GloVeEmbedder          : real static word vectors (gensim).
SentenceTransformerEmbedder : contextual sub-word embeddings (handles OOV, phrases).

All backends expose:  embed(poem:str) -> (words:list[str], vectors:np.ndarray[N,d])
Vectors are z-scored per dimension so the CPPN sees a well-conditioned input.

Why z-score:
-  CPPN activations (tanh, sigmoid, sin, etc.) are most expressive near zero.
    If the raw embeddings had, for ex., a mean of 50 with std 0.1, the CPPN inputs would all be ~50 and tanh would saturate to 1 for everything. 
    Standardising to mean=0, std=1 puts the inputs in the "interesting" range of the activation sfunctions.

    Formula per dimension j:
        z_{i,j} = (x_{i,j} - mean_j) / std_j
    where mean_j and std_j are computed across all N words.
"""
from __future__ import annotations
import re
import numpy as np

# matches alphabetic words and contractions (e.g., "don't", "it's")
_TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(poem: str, keep_dupes: bool = False) -> list[str]:
    """
    Extract lowercase word tokens from a poem string.

    Parameters:
    - poem : str -Raw poem text (may contain newlines, punctuation, etc.).
    - keep_dupes : bool
        If False (default), deduplicate tokens preserving first-occurrence order.
        If True, keep every occurrence (needed for reading-order rendering where the same word appears multiple times on different lines).

    Returns:
    - list[str] - Lowercase tokens in order of appearance.
    """
    toks = [w.lower() for w in _TOKEN_RE.findall(poem)]
    if keep_dupes:
        return toks
    # deduplicate while preserving order
    seen, out = set(), []
    for w in toks:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _standardize(X: np.ndarray) -> np.ndarray:
    """
    #Z-score normalisation: centre each dimension to mean=0, std=1.

    Formula:
        z_{i,j} = (x_{i,j} - mu_j) / sigma_j

    where:
        mu_j    = (1/N) * sum_i(x_{i,j})        (column mean)
        sigma_j = sqrt( (1/N) * sum_i(x_{i,j} - mu_j)^2 )  (column std)

    If a dimension has near-zero variance (std < 1e-8), its std is set to 1.0 to avoid division by zero (those dimensions become constant at zero).

    Parameters:
    - X : ndarray (N, d) - Raw embedding matrix.

    Returns:
    - ndarray (N, d) - Standardised embedding matrix.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    # prevent division by zero for constant dimensions
    sd[sd < 1e-8] = 1.0 
    return (X - mu) / sd


class HashEmbedder:
    """
    Deterministic character-trigram hashing into d dimensions.

    NON-SEMANTIC: similar words don't get similar vectors. This exists only as a fallback that always works (no downloads, no NLTK, no network) and as a
    smoke test for the pipeline.

    Algorithm for word w:
        1. Pad w with '#': e.g., "cat" -> "#cat#"
        2. Extract all character trigrams: "#ca", "cat", "at#"
        3. For each trigram t:
           - Compute h = hash(t)  (Python's built-in hash)
           - Pick a dimension:     bin = h mod d
           - Pick a sign:          sign = +1 if bit 32 of h is 1, else -1
           - Add sign to v[bin]
        4. L2-normalise the result so all word vectors live on the unit sphere.

    This is a simplified version of feature hashing / the hashing trick.

    Parameters:
    - dim : int - Dimensionality of the output vectors. Default 8.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vec(self, word: str) -> np.ndarray:
        """
        Hash a single word into a d-dimensional vector.
        """
        v = np.zeros(self.dim)
        padded = f"#{word}#"
        for i in range(len(padded) - 2):
            tri = padded[i:i + 3]
            h = hash(tri)
            # assign to a dimension (h mod d) with a sign (bit 32)
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        n = np.linalg.norm(v)
         # L2 normalise
        return v / n if n > 0 else v

    def embed(self, poem: str):
        """
        Embed all unique tokens in the poem.

        Returns:
        - (words, vectors) : (list[str], ndarray (N, d))
        """
        words = tokenize(poem)
        X = np.array([self._vec(w) for w in words])
        return words, _standardize(X)


class WordNetEmbedder:
    """
    Offline semantic embeddings via WordNet path similarity + classical MDS.

    Algorithm:
        1. Tokenize the poem and look up WordNet synsets for each word.
        2. Build an N x N pairwise distance matrix D where:
               D[i,j] = 1 - path_similarity(word_i, word_j)
           path_similarity is based on the shortest path between two synsets in the WordNet hypernym/hyponym taxonomy. It ranges from 0 (no path)
           to 1 (same synset), so 1 - sim gives a distance in [0, 1].
           Words with no synsets (function words, proper nouns) get max distance (1.0).
        3. Apply classical (metric) MDS to recover d-dimensional coordinates from the distance matrix.

    Classical MDS:
    Given an N x N distance matrix D, classical MDS finds coordinates X such that ||x_i - x_j|| approximates D[i,j]. Steps:

        1. Double-centre the squared-distance matrix:
               J = I - (1/n) * 11^T          (centering matrix)
               B = -0.5 * J @ D^2 @ J        (inner-product matrix)
           B is the Gram matrix (X @ X^T) if D came from Euclidean distances.

        2. Eigendecompose B:
               B = V @ diag(lambda) @ V^T

        3. Take the top-d eigenvalues and eigenvectors:
               X = V[:, :d] @ diag(sqrt(lambda[:d]))

    If some eigenvalues are negative (which happens when D is non-Euclidean, as with path_similarity), they're clipped to zero.

    Fallback for OOV words: words without synsets get their MDS coordinates nudged by a HashEmbedder vector scaled by 0.5, so they scatter
    deterministically rather than collapsing to the same point.

    Parameters:
    - dim : int - Number of MDS dimensions. Default 8.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        import nltk
        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from nltk.corpus import wordnet as wn
        self.wn = wn

    def _best_pair_sim(self, a, b) -> float:
        """
        Compute the best WordNet path similarity between two words.

        Each word can map to multiple synsets (senses). We take the maximum path similarity across the top 3 synsets of each word (capped at 3
        for speed, since rare senses rarely help).

        path_similarity(s1, s2) = 1 / (1 + shortest_path_length(s1, s2))
        where shortest_path_length is the number of edges on the shortest path in the WordNet hypernym/hyponym tree.

        Returns None if either word has no synsets.
        """
        sa, sb = self.wn.synsets(a), self.wn.synsets(b)
        if not sa or not sb:
            return None
        best = 0.0
        for x in sa[:3]:       # top 3 synsets of word a
            for y in sb[:3]:   # top 3 synsets of word b
                s = x.path_similarity(y)
                if s and s > best:
                    best = s
        return best

    def embed(self, poem: str):
        """
        Embed the poem via WordNet path similarity + MDS.

        Returns: 
        - (words, vectors) : (list[str], ndarray (N, d)) - Standardised d-dimensional embeddings for each unique token.
        """
        words = tokenize(poem)
        n = len(words)
        has_syn = [bool(self.wn.synsets(w)) for w in words]

        # build pairwise distance matrix
        D = np.ones((n, n)) * 1.0  # default: max distance
        for i in range(n):
            D[i, i] = 0.0
            for j in range(i + 1, n):
                if has_syn[i] and has_syn[j]:
                    s = self._best_pair_sim(words[i], words[j])
                    d = 1.0 - (s if s is not None else 0.0)
                else:
                    d = 1.0  # at least one word has no synset -> max distance
                D[i, j] = D[j, i] = d

        # recover coordinates via classical MDS
        X = self._classical_mds(D, self.dim)

        # nudge OOV words apart so they don't all sit at the origin
        h = HashEmbedder(self.dim)
        for i, ok in enumerate(has_syn):
            if not ok:
                X[i] += 0.5 * h._vec(words[i])

        return words, _standardize(X)

    @staticmethod
    def _classical_mds(D: np.ndarray, dim: int) -> np.ndarray:
        """
        Classical (metric) multidimensional scaling.

        Given distance matrix D (N x N), recover N points in R^dim such that inter-point Euclidean distances approximate D.

        Steps:
            1. Compute the centering matrix:
                   J = I_n - (1/n) * 1_n @ 1_n^T
               This removes the mean from both rows and columns.

            2. Compute the double-centred inner-product (Gram) matrix:
                   B = -0.5 * J @ D^2 @ J
               If D is truly Euclidean, B = X @ X^T for some X. Symmetrise B to fix floating-point drift: B = (B + B^T) / 2.

            3. Eigendecompose B. Sort eigenvalues descending.

            4. Take the top `dim` eigenvalues (clipped to >= 0 since negative eigenvalues indicate non-Euclidean distortion). Coordinates:
                   X = V[:, :dim] @ diag(sqrt(lambda[:dim]))

            5. If fewer than `dim` positive eigenvalues exist, pad with zeros.

        Parameters:
        - D : ndarray (N, N) - Symmetric distance matrix with zero diagonal.
        - dim : int - Target dimensionality.

        Returns:
        - ndarray (N, dim) - Recovered coordinates.
        """
        n = D.shape[0]
        # step 1: centering matrix J = I - (1/n) * 11^T
        J = np.eye(n) - np.ones((n, n)) / n
        # step 2: double-centred Gram matrix B = -0.5 * J * D^2 * J
        B = -0.5 * J @ (D ** 2) @ J
        B = (B + B.T) / 2  # enforce exact symmetry
        # step 3: eigendecomposition
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]  # sort eigenvalues descending
        vals, vecs = vals[idx], vecs[:, idx]
        # step 4: clip negative eigenvalues, take top `dim`
        vals = np.clip(vals[:dim], 0, None)
        # coordinates: X_i = V_i * sqrt(lambda_i)   (+ eps for numerical safety)
        L = vecs[:, :dim] * np.sqrt(vals + 1e-9)
        # step 5: pad if fewer positive eigenvalues than requested dimensions
        if L.shape[1] < dim:
            L = np.hstack([L, np.zeros((n, dim - L.shape[1]))])
        return L


class GloVeEmbedder:
    """
    Real static word vectors via gensim-data.

    Uses pre-trained GloVe vectors (default: 50-dimensional, trained on Wikipedia + Gigaword). Words not in the vocabulary are silently dropped.

    Needs network access on first use to download the model (~70MB for glove-wiki-gigaword-50). Subsequent runs use the cached copy.

    Parameters:
    - name : str - Name of the gensim-data model. Default "glove-wiki-gigaword-50".
    """

    def __init__(self, name: str = "glove-wiki-gigaword-50"):
        import gensim.downloader as api
        self.model = api.load(name)
        self.dim = self.model.vector_size

    def embed(self, poem: str):
        """
        Embed poem words using pre-trained GloVe vectors.

        Only words present in the GloVe vocabulary are returned. Out-of-vocabulary words are silently dropped (caller handles the missing-word case).

        Returns:
        - (words, vectors) : (list[str], ndarray (N, d))
        """
        words = [w for w in tokenize(poem) if w in self.model]
        X = np.array([self.model[w] for w in words])
        return words, _standardize(X)


class SentenceTransformerEmbedder:
    """
    Contextual token embeddings via sentence-transformers.

    Each word is encoded independently as a single-token "sentence". This handles OOV via sub-word tokenisation (BPE/WordPiece) and produces
    the highest-quality embeddings among the backends, at the cost of requiring a transformer model (~80MB for all-MiniLM-L6-v2).

    Parameters:
    - name : str - HuggingFace model name. Default "all-MiniLM-L6-v2" (384 dims).
    """

    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, poem: str):
        """
        Embed each unique word as an independent sentence.

        normalize_embeddings=True L2-normalises each vector to the unit sphere, which makes cosine similarity equivalent to dot product.

        Returns:
        - (words, vectors) : (list[str], ndarray (N, d))
        """
        words = tokenize(poem)
        X = np.array(self.model.encode(words, normalize_embeddings=True))
        return words, _standardize(X)