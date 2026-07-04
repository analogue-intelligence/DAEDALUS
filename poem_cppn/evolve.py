"""
The evolution loop: embeddings -> NEAT (CPPN) -> corpus-free fitness -> best layout.

This is the top-level coordinator. It wires together:
    - embed.py   : poem -> word embedding vectors
    - cppn.py    : NEAT genome -> CPPN network -> layout (positions, colours, sizes)
    - fitness.py : layout -> scalar fitness score

The NEAT algorithm then breeds, mutates, and selects genomes across generations to maximise the fitness score. The result is a CPPN that maps word embeddings
to an aesthetically pleasing and semantically structured composition.

NEAT loop overview:
    for each generation:
        for each genome in population:
            1. Build a CPPN from the genome
            2. Feed all word embeddings through the CPPN -> positions + colours
            3. Evaluate fitness (semantic fidelity + aesthetic quality)
        Select, crossover, mutate -> next generation
    Return the highest-fitness genome across all generations.
"""
from __future__ import annotations
import os
import numpy as np
import neat

from .cppn import make_config, layout_from_net
from .fitness import Fitness, FitnessWeights


class Evolver:
    """
    Run NEAT evolution to find a CPPN that produces a good visual layout.

    Parameters:
    - words : list[str] - The poem's unique tokens.
    - vectors : ndarray (N, d) - Standardised embedding vectors, one per word.
    - width, height : int - Canvas dimensions in pixels. Default 820x820.
    - pop_size : int - Number of genomes per NEAT generation. Default 150. - Larger populations explore more topologies but run slower.
    - weights : FitnessWeights or None - Fitness term weights (see fitness.py). None uses defaults.
    - config_path : str - Where to write the NEAT .ini config file.
    - seed : int - Random seed for reproducibility. Default 0.
    - shape : str - Composition shape constraint: "circle", "square", or "rect".
    """

    def __init__(self, words, vectors, width=820, height=820, pop_size=150, weights: FitnessWeights | None = None, config_path="config-cppn.ini", seed=0, shape="circle"):
        self.words = words
        self.vectors = vectors
        self.width, self.height = width, height
        self.shape = shape

        # fitness evaluator (pre-computes pairwise embedding distances)
        self.fitness = Fitness(vectors, weights, shape=shape,
                               width=width, height=height)
        # generate and load NEAT config with the correct input dimensionality
        self.config = make_config(vectors.shape[1], pop_size, config_path)
        # storage for intermediate snapshots (generation -> (genome, fitness))
        self.snapshots = {}

        # fix seeds for reproducibility across both numpy and stdlib random
        np.random.seed(seed)
        import random
        random.seed(seed)

    def _layout(self, genome):
        """
        Convert a NEAT genome into a concrete layout.

        Steps:
            1. Build a feed-forward neural network from the genome's topology.
            2. Feed each word's embedding vector through the network.
            3. Squash outputs into positions, colours, and sizes.

        Returns:
        - (positions, colors, sizes) : (ndarray (N,2), ndarray (N,3), ndarray (N,))
        """
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        return layout_from_net(net, self.vectors, self.width, self.height, shape=self.shape)

    def _eval(self, genomes, config):
        """Evaluate fitness for a batch of genomes (called by NEAT each generation).

        For each genome:
            1. Compute the layout (positions + colours)
            2. Score the layout using the multi-objective fitness function
            3. Assign the scalar fitness to genome.fitness

        Parameters:
        - genomes : list of (genome_id, genome) - The current generation's genomes.
        - config : neat.Config - NEAT config (passed by the framework, not used directly here).
        """
        for _, g in genomes:
            pos, col, _sz = self._layout(g)
            g.fitness = self.fitness(pos, col, self.width, self.height)

    def run(self, generations=80, snapshot_at=(0,)):
        """
        Run the full NEAT evolution loop.

        Parameters:
        - generations : int - Number of generations to evolve. Default 80.
        - snapshot_at : tuple of int - Which generation numbers to save a snapshot of the best genome. Useful for visualising how 
            the composition improves over time. The final generation is always snapshotted under key "final".

        Returns:
        - (winner, stats) : (neat.DefaultGenome, neat.StatisticsReporter) - The highest-fitness genome ever found, plus NEAT statistics. The winner's layout can be extracted via layout_for(winner).
        """
        pop = neat.Population(self.config)
        pop.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        def eval_and_snapshot(genomes, config):
            """
            Evaluate fitness + optionally save the best genome at this generation.
            """
            self._eval(genomes, config)
            gen = pop.generation
            if gen in snapshot_at:
                best = max((g for _, g in genomes), key=lambda g: g.fitness)
                self.snapshots[gen] = (best, best.fitness)

        winner = pop.run(eval_and_snapshot, generations)
        self.snapshots["final"] = (winner, winner.fitness)
        return winner, stats

    def layout_for(self, genome):
        """
        Extract the layout for a specific genome (e.g., the winner or a snapshot).

        Returns:
        - (positions, colors, sizes) : (ndarray (N,2), ndarray (N,3), ndarray (N,))
        """
        return self._layout(genome)