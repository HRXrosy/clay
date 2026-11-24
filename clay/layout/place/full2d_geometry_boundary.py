"""Join either complete geometry proof to the same final master operands.

Optimization evidence belongs to the optimizing graph. A fixed-candidate proof
retains physical replay obligations, without representing solver alternatives.
"""
from clay.layout.extract.compaction import Full2DCompactionGraph
from clay.layout.extract.fixed_incumbent_geometry import FixedIncumbentGeometryProof


def require_full2d_geometry_proof(graph, blocks, *, overlap_compatible_pairs,
                                layered_footprints_nm, solver_evidence=()):
    """Require an exact proof type and replay its complete final geometry join."""
    if type(graph) not in (Full2DCompactionGraph, FixedIncumbentGeometryProof):
        raise TypeError('final placement has a foreign full-2D compaction graph')
    if type(graph) is FixedIncumbentGeometryProof and solver_evidence:
        raise ValueError('fixed incumbent geometry cannot carry aggregate optimization evidence')
    if not graph.replay(dict(blocks)):
        raise ValueError('final placement full-2D graph does not replay final master blocks')
    if graph.overlap_compatible_pairs != overlap_compatible_pairs:
        raise ValueError('final placement graph overlap compatibility differs from certificate')
    if graph.layered_footprints_nm != layered_footprints_nm:
        raise ValueError('final placement graph layered collision atoms differ from certificate')
