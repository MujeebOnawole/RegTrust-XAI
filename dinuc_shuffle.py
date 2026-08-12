"""
Dinucleotide-preserving sequence shuffle, for item 9 rung 1b (motif-shuffled
synthetic controls -- the Nagai et al. 2026 review's own Table 1 "motif
rearrangement" covariate-shift category: same local composition, destroyed
regulatory grammar).

WHY DINUCLEOTIDE-PRESERVING, NOT A PLAIN PER-BASE SHUFFLE. Real genomic DNA
has strong dinucleotide biases (CpG depletion being the best-known example).
A naive per-base shuffle destroys those biases along with motif syntax, so a
model flagging the shuffled sequence as out-of-domain could be responding to
the broken dinucleotide statistics rather than the destroyed motif
organization -- a confound that would make the rung 1b result uninterpretable.
Shuffling at the dinucleotide level (Altschul & Erikson 1985, the same
algorithm MEME's fasta-dinucleotide-shuffle and kundajelab/deeplift's
dinuc_shuffle use for exactly this purpose in regulatory genomics) preserves
every dinucleotide's count exactly, isolating "motif organization destroyed"
from "base-composition statistics changed."

ALGORITHM. A sequence's dinucleotide transitions form a directed multigraph
on {A,C,G,T} (an edge per observed dinucleotide, in sequence order).  Any
reordering of those edges that preserves an Eulerian path from the first to
the last base of the original sequence gives a valid shuffle with identical
dinucleotide counts. Not every random reordering of edges preserves that
path (some choices disconnect the graph), so Wilson's algorithm builds a
random spanning tree rooted at the LAST base first, fixing one "escape edge"
per node that guarantees connectivity to the root; every other edge at each
node is then freely shuffled.
"""
from __future__ import annotations

import numpy as np


def _wilson_last_edges(edges: dict[str, list[str]], root: str, rng: np.random.Generator) -> dict[str, int]:
    """Loop-erased random walk (Wilson's algorithm) to pick, for every node
    with outgoing edges, ONE out-edge index that must be moved to the end of
    that node's shuffled edge list -- this is what guarantees an Eulerian
    path from the sequence's first base to its last base (`root`) survives
    after the rest of each node's edges are shuffled freely."""
    nodes_with_edges = [n for n, e in edges.items() if e]
    in_tree = {root}
    fixed_idx: dict[str, int] = {}
    for start in nodes_with_edges:
        if start in in_tree:
            continue
        path_nodes = [start]
        path_choice: dict[str, int] = {}
        u = start
        while u not in in_tree:
            idx = int(rng.integers(len(edges[u])))
            v = edges[u][idx]
            path_choice[u] = idx
            if v in path_nodes:
                # loop erasure: cut everything from v's earlier occurrence onward
                cut = path_nodes.index(v)
                for stale in path_nodes[cut + 1:]:
                    path_choice.pop(stale, None)
                path_nodes = path_nodes[:cut + 1]
            else:
                path_nodes.append(v)
            u = v
        for node in path_nodes[:-1]:
            in_tree.add(node)
        fixed_idx.update(path_choice)
    return fixed_idx


def dinuc_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Dinucleotide-count-preserving shuffle of seq. Deterministic given rng."""
    seq = seq.upper()
    tokens = list(seq)
    if len(tokens) < 3:
        return seq

    edges: dict[str, list[str]] = {b: [] for b in "ACGT"}
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a in edges and b in edges:
            edges[a].append(b)

    root = tokens[-1]
    fixed_idx = _wilson_last_edges(edges, root, rng)

    shuffled_edges: dict[str, list[str]] = {}
    for node, lst in edges.items():
        lst = lst[:]
        if node in fixed_idx:
            fixed_val = lst.pop(fixed_idx[node])
            rng.shuffle(lst)
            lst.append(fixed_val)
        else:
            rng.shuffle(lst)
        shuffled_edges[node] = lst

    ptrs = {b: 0 for b in "ACGT"}
    out = [tokens[0]]
    cur = tokens[0]
    for _ in range(len(tokens) - 1):
        if cur not in shuffled_edges or ptrs[cur] >= len(shuffled_edges[cur]):
            # cur was a non-ACGT base (N) or ran out of edges (shouldn't happen
            # for a well-formed ACGT-only sequence) -- fail loudly rather than
            # silently emit a wrong-length or wrong-composition sequence.
            raise ValueError(f"dinuc_shuffle: exhausted edges at {cur!r}; "
                              f"input likely contains non-ACGT characters")
        nxt = shuffled_edges[cur][ptrs[cur]]
        ptrs[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


if __name__ == "__main__":
    from collections import Counter

    rng = np.random.default_rng(0)
    alphabet = "ACGT"
    test_seq = "".join(rng.choice(list(alphabet), size=2048, p=[0.3, 0.2, 0.2, 0.3]))

    def dinuc_counts(s):
        return Counter(s[i:i + 2] for i in range(len(s) - 1))

    orig_counts = dinuc_counts(test_seq)
    n_mismatches = 0
    for trial in range(20):
        shuffled = dinuc_shuffle(test_seq, np.random.default_rng(trial))
        assert len(shuffled) == len(test_seq), "shuffle changed sequence length"
        assert Counter(shuffled) == Counter(test_seq), "shuffle changed base composition"
        assert shuffled[0] == test_seq[0], "shuffle changed the first base"
        assert shuffled[-1] == test_seq[-1], "shuffle changed the last base"
        shuf_counts = dinuc_counts(shuffled)
        if shuf_counts != orig_counts:
            n_mismatches += 1
        if trial == 0:
            ident = sum(a == b for a, b in zip(shuffled, test_seq)) / len(test_seq)
            print(f"trial 0: fraction of positions unchanged by shuffle: {ident:.3f} (expect low)")
    print(f"dinucleotide-count mismatches across 20 trials: {n_mismatches}/20 (expect 0)")
    assert n_mismatches == 0, "dinucleotide counts were NOT preserved -- algorithm is broken"
    print("sanity checks passed: length, base composition, first/last base, "
          "and full dinucleotide-count multiset all preserved across 20 random shuffles")
