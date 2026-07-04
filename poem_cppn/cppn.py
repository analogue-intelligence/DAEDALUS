"""
CPPN side: generate a NEAT config for the right input dimensionality, and turn an evolved network into a concrete layout (position + colour per word).

The CPPN here is a function over EMBEDDING space (not pixel space):
    input  = a word's d-dim embedding vector
    output = 6 values -> squashed to (x, y, r, g, b, size)
NEAT evolves topology + per-node activations (sin, gauss, abs, ...), which is what makes the projection structured rather than a plain linear map.

Architecture overview:
A Compositional Pattern-Producing Network (CPPN) is a neural network whose hidden nodes can use different activation functions (sin, gauss, abs, etc.)
rather than being restricted to a single one (like ReLU in standard MLPs). This activation diversity lets the network express symmetry, repetition,
and other geometric regularities compactly.

NEAT (NeuroEvolution of Augmenting Topologies) evolves both the weights and the topology of the CPPN: it starts with a minimal network (no hidden nodes)
and complexifies over generations by adding nodes and connections. This means the mapping from embedding to layout starts simple and gains structure only
as needed by the fitness function.

The flow:
    word -> embedder -> d-dim vector -> CPPN -> (x, y, r, g, b, size) - squashed via tanh (position) and sigmoid (colour, size)
"""
from __future__ import annotations
import math
import numpy as np
import neat

# CPPN-style activation palette.
#
# This is the defining feature of a CPPN vs a regular MLP: each hidden node can independently use any of these activations. NEAT mutates the activation
# choice alongside weights and topology.
#
# Why these specific functions:
#   - sigmoid / tanh : smooth saturation, good for colour & position output
#   - sin            : periodic patterns (repetition, waves)
#   - gauss          : localised bumps (clusters, radial features)
#   - abs            : V-shapes, symmetry
#   - square / cube  : polynomial non-linearities (parabolas, S-curves)
#   - identity       : pass-through (lets some paths stay linear)
#   - relu           : standard half-rectifier

_ACTIVATIONS = "sigmoid tanh sin gauss abs square cube identity relu"

# NEAT configuration template.
#
# This is a Python-format string that gets filled with {num_inputs} (the embedding dimensionality) and {pop} (population size). The resulting .ini
# file is what python-neat reads.
#
# parameters explained:
#   num_inputs            : d (embedding dim). Set dynamically.
#   num_outputs           : 6 -> (x, y, r, g, b, size) raw outputs.
#   num_hidden            : 0 -> NEAT starts with no hidden nodes and adds them.
#   feed_forward          : True -> no recurrence (CPPN is a static function).
#   initial_connection    : full_direct -> every input connects to every output at generation 0 (a linear map to start with).
#   activation_mutate_rate: 0.30 -> 30% chance of changing a node's activation function each generation. High because activation 
#                           diversity is the whole point of CPPNs.
#   conn_add_prob         : 0.5 -> aggressive connection addition (complexify fast).
#   node_add_prob         : 0.3 -> moderate node addition.
#   weight_init_stdev     : 1.5 -> fairly wide initial weight distribution so the initial linear map is non-trivial.
#   compatibility_threshold : 3.0 -> how different two genomes must be to belong to different species. NEAT's speciation protects innovation by letting 
#                                    novel topologies compete only within their niche.
_CONFIG_TEMPLATE = """\
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100000
no_fitness_termination = True
pop_size              = {pop}
reset_on_extinction   = True

[DefaultGenome]
num_inputs            = {num_inputs}
num_outputs           = 6
num_hidden            = 0
feed_forward          = True
initial_connection    = full_direct

activation_default      = tanh
activation_mutate_rate  = 0.30
activation_options      = {activations}

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_max_value        = 5.0
bias_min_value        = -5.0
bias_mutate_power     = 0.5
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.1

response_init_mean    = 1.0
response_init_stdev   = 0.0
response_max_value    = 5.0
response_min_value    = -5.0
response_mutate_power = 0.0
response_mutate_rate  = 0.0
response_replace_rate = 0.0

weight_init_mean      = 0.0
weight_init_stdev     = 1.5
weight_max_value      = 8.0
weight_min_value      = -8.0
weight_mutate_power   = 0.6
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1

enabled_default       = True
enabled_mutate_rate   = 0.02

conn_add_prob         = 0.5
conn_delete_prob      = 0.2
node_add_prob         = 0.3
node_delete_prob      = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


def make_config(num_inputs: int, pop_size: int, path: str) -> neat.Config:
    """
    Write a NEAT config file parameterised on the embedding dimension and load it.

    Parameters:
    - num_inputs : int - Dimensionality of the embedding vectors (= number of CPPN input nodes).
    - pop_size : int - Number of genomes in the NEAT population per generation.
    - path : str - Filesystem path to write the .ini config to.

    Returns:
    - neat.Config - A parsed NEAT configuration object ready to create a Population.
    """
    with open(path, "w") as f:
        f.write(_CONFIG_TEMPLATE.format(num_inputs=num_inputs, pop=pop_size, activations=_ACTIVATIONS))
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, path)


def _sigmoid(x: float) -> float:
    """
    Standard logistic sigmoid: sigma(x) = 1 / (1 + e^{-x}).

    Clamps input to [-30, 30] to prevent floating-point overflow in exp(). Maps any real number to (0, 1), used to squash raw CPPN outputs into
    valid colour channels and size values.
    """
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _square_to_disk(u: float, v: float):
    """
    Shirley-Chiu concentric square-to-disk mapping.

    Maps a point (u, v) in the square [-1, 1]^2 onto the unit disk such that:
      - Area is preserved (equal-area): a uniform distribution over the square
        maps to a uniform distribution over the disk.
      - The mapping is continuous and bijective (no fold-overs).
      - Concentric squares map to concentric circles.

    This is used instead of the naive (r*cos(theta), r*sin(theta)) polar mapping because the polar approach bunches points near the centre
    (non-uniform). Shirley-Chiu keeps the density even.

    Algorithm (two cases based on which axis dominates):
        If |u| > |v|:
            r   = u
            phi = (pi/4) * (v/u)
        Else:
            r   = v
            phi = (pi/2) - (pi/4) * (u/v)

        Then: (x, y) = (r * cos(phi), r * sin(phi))

    Parameters:
    - u, v : float - Coordinates in [-1, 1] (from tanh of CPPN output).

    Returns:
    - (x, y) : tuple of float - Point on the unit disk (radius <= 1).
    """
    if u == 0.0 and v == 0.0:
        return 0.0, 0.0
    if abs(u) > abs(v):
        r, phi = u, (math.pi / 4.0) * (v / u)
    else:
        r, phi = v, (math.pi / 2.0) - (math.pi / 4.0) * (u / v)
    return r * math.cos(phi), r * math.sin(phi)


def layout_from_net(net, vectors: np.ndarray, width: int, height: int, margin: int = 60, shape: str = "circle", radius_range=(7.0, 26.0)):
    """
    Feed every word embedding through the CPPN and extract layout properties.

    For each word vector v_i, the CPPN produces 6 raw outputs:
        [ox, oy, r, g, b, s] = net.activate(v_i)

    These raw outputs are then squashed into usable ranges:

    Position (x, y):
        1. Apply tanh to ox, oy -> (u, w) in [-1, 1]
        2. Depending on `shape`:
           - "circle": Apply Shirley-Chiu mapping (u, w) -> disk point (a, c) in [-1, 1], then scale to canvas:
                       x = cx + a * R,  y = cy + c * R
                       where R = min(inner_width, inner_height) / 2
           - "square": Direct scaling within a centred square:
                       x = cx + u * R,  y = cy + w * R
           - "rect":   Linear mapping to the full inner rectangle:
                       x = margin + (u * 0.5 + 0.5) * inner_width
                       y = margin + (w * 0.5 + 0.5) * inner_height

    Colour (r, g, b):
        Apply sigmoid to each channel -> values in (0, 1). 
        sigmoid(x) = 1 / (1 + e^{-x})

    Size:
        Apply sigmoid to the 6th output, then linearly interpolate:
        size = r_min + sigmoid(s) * (r_max - r_min)

    Parameters:
    - net : neat.nn.FeedForwardNetwork - The evolved CPPN network.
    - vectors : ndarray (N, d) - One d-dimensional embedding vector per word.
    - width, height : int - Canvas dimensions in pixels.
    - margin : int - Border margin in pixels (default 60).
    - shape : str - Composition shape constraint: "circle", "square", or "rect".
    - radius_range : tuple (float, float) - (min_size, max_size) in pixels for the word blobs.

    Returns:
    - positions : ndarray (N, 2) - Pixel coordinates for each word.
    - colors : ndarray (N, 3) - RGB in [0, 1] for each word.
    - sizes : ndarray (N,) - Blob radius in pixels for each word.
    """
    positions, colors, sizes = [], [], []
     # canvas centre
    cx, cy = width / 2.0, height / 2.0 
    # inner dimensions
    iw, ih = width - 2 * margin, height - 2 * margin 
     # radius for circle/square shapes
    R = min(iw, ih) / 2.0  
    rmin, rmax = radius_range

    for v in vectors:
        out = net.activate(v.tolist())
        ox, oy, r, g, b = out[:5]
        s = out[5] if len(out) > 5 else 0.0

        # squash position outputs to [-1, 1] via tanh
        u, w = math.tanh(ox), math.tanh(oy)

        if shape == "circle":
            # Shirley-Chiu: uniform square -> uniform disk, then scale to canvas
            a, c = _square_to_disk(u, w)
            x, y = cx + a * R, cy + c * R
        elif shape == "square":
            # direct linear mapping within centred square
            x, y = cx + u * R, cy + w * R
        else:  # rect
            # map [-1,1] to [margin, width-margin] via (u*0.5+0.5) -> [0,1]
            x = margin + (u * 0.5 + 0.5) * iw
            y = margin + (w * 0.5 + 0.5) * ih

        positions.append((x, y))
        # squash colour channels to [0, 1] via sigmoid
        colors.append((_sigmoid(r), _sigmoid(g), _sigmoid(b)))
        # size: sigmoid maps to [0,1], then linearly interpolate between rmin and rmax
        sizes.append(rmin + _sigmoid(s) * (rmax - rmin))

    return np.array(positions), np.array(colors), np.array(sizes)