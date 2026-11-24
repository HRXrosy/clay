"""Compile immutable reachability values for one mosaic projection invocation."""
from collections.abc import Callable
from types import MappingProxyType

from clay.layout.spatial import abstract_ir


def compile_dag_queries(dag: abstract_ir.OrthogonalDag) -> Callable[[str, str], bool]:
    """Keep queries local to a captured DAG; reject replacement of its operands."""
    if type(dag) is not abstract_ir.OrthogonalDag:
        raise TypeError('mosaic queries require an exact OrthogonalDag')
    nodes, edges = dag.nodes, dag.edges
    reachable = MappingProxyType(abstract_ir._reachability(nodes, edges))

    def reaches(before: str, after: str) -> bool:
        if dag.nodes is not nodes or dag.edges is not edges:
            raise ValueError('mosaic query DAG operands changed')
        if before not in reachable or after not in reachable:
            raise abstract_ir.SpatialIrSchemaError('reachability query names an unknown node')
        return after in reachable[before]

    return reaches
