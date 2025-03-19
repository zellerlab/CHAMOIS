"""Classes for working with ontological data.
"""

import collections
import typing
from typing import Optional, Iterator, Iterable, Union, Sequence, Dict

import numpy

from ._meta import requires

if typing.TYPE_CHECKING:
    from scipy.sparse import spmatrix


def _compute_semantic_similarity(adjacency_matrix: "AdjacencyMatrix", c=2.0, d=0.5) -> "spmatrix":
    # compute downstream semantic contribution
    semantic_contribution = numpy.eye(len(adjacency_matrix))
    for i in range(len(adjacency_matrix)):
        semantic_contribution[i, i] = 1
        nodes = collections.deque()
        nodes.append(i)
        while nodes:
            j = nodes.popleft()
            for k in adjacency_matrix.parents(j):
                we = 1 / (c + len(adjacency_matrix.children(k))) + d
                semantic_contribution[i, k] = max(semantic_contribution[i, k], we * semantic_contribution[i, j])
                nodes.append(k)

    # compute semantic value of each term
    semantic_value = semantic_contribution.sum(axis=1)

    # compute semantic similarity
    semantic_similarity = numpy.eye(len(adjacency_matrix))
    ancestors = _VariableColumnStorage([
        adjacency_matrix.ancestors(i)
        for i in range(len(adjacency_matrix))
    ])
    for i in range(len(adjacency_matrix)):
        ancestors1 = set(ancestors[i])
        for j in range(i+1, len(adjacency_matrix)):
            ancestors2 = set([j])
            indices = numpy.array(list(ancestors1 & ancestors2))
            if len(indices):
                denominator = semantic_contribution[i, indices].sum() + semantic_contribution[j, indices].sum()
                if denominator > 0:
                    semantic_similarity[i, j] = semantic_similarity[j, i] = denominator / (semantic_value[i] + semantic_value[j])

    return semantic_similarity


class _VariableColumnStorage:
    """A compressed storage for arrays with a variable number of columns.
    """

    def __init__(self, data: Sequence[numpy.ndarray], dtype: Union[numpy.dtype, str, None]=None):
        self.indices = numpy.zeros(len(data)+1, dtype=numpy.int64)
        self.data = numpy.zeros(sum(len(row) for row in data), dtype=dtype)

        for i, row in enumerate(data):
            self.indices[i+1] = self.indices[i] + len(row)
            self.data[self.indices[i]:self.indices[i+1]] = numpy.asarray(row)

    def __len__(self):
        return self.indices.shape[0] - 1

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return type(self)([self[i] for i in range(*index.indices(len(self)))])
        start = self.indices[index]
        end = self.indices[index+1]
        return self.data[start:end]


class AdjacencyMatrix:
    """A tree encoded as an adjacency matrix.
    """

    @requires("scipy.sparse")
    def __init__(self, data: Union[numpy.ndarray, "spmatrix", None] = None) -> None:
        if data is None:
            _data = numpy.array([])
        elif isinstance(data, scipy.sparse.spmatrix):
            _data = data.toarray()
        else:
            _data = numpy.asarray(data, dtype=numpy.int32)

        if _data.size:
            # cache parents and children
            self._parents = _VariableColumnStorage(
                [numpy.nonzero(_data[i, :])[0] for i in range(_data.shape[0])],
                dtype=numpy.int64
            )
            self._children = _VariableColumnStorage(
                [numpy.nonzero(_data[:, i])[0] for i in range(_data.shape[0])],
                dtype=numpy.int64
            )
            # run a BFS walk and store indices
            self._down = self._roots_to_leaves(_data)
            self._up = self._leaves_to_root(_data)
        else:
            self._parents = self._children = _VariableColumnStorage([])
            self._down = self._up = numpy.array([])

        # store input data as a sparse matrix
        self.data = scipy.sparse.csr_matrix(_data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __iter__(self) -> Iterator[int]:
        return iter(self._down)

    def __reversed__(self) -> Iterator[int]:
        return iter(self._up)

    def __getstate__(self) -> Dict[str, object]:
        return dict(data=self.data)

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__init__(state["data"])

    def _roots_to_leaves(self, data: numpy.ndarray) -> numpy.ndarray:
        """Generate a walk order to explore the tree from roots to leaves.
        """
        path = -numpy.ones(data.shape[0], dtype=numpy.int64)
        roots = numpy.where(data.sum(axis=1) == 0)[0]

        n = 0
        todo = collections.deque(roots)
        done = numpy.zeros(data.shape[0], dtype=bool)

        while todo:
            # get next node to visit
            i = todo.popleft()
            # skip if done already
            if done[i]:
                continue
            # delay node visit if we didn't visit some of its children yet
            if not numpy.all(done[self.parents(i)]):
                todo.append(i)
                continue
            # add node parents
            todo.extend(self.children(i))
            # mark node as done
            done[i] = True
            path[n] = i
            n += 1

        assert n == len(path)
        assert (done != 0).sum() == data.shape[0]
        return path

    def _leaves_to_root(self, data: numpy.ndarray) -> numpy.ndarray:
        """Generate a walk order to explore the tree from leaves to roots.
        """
        path = -numpy.ones(data.shape[0], dtype=numpy.int64)
        roots = numpy.where(data.sum(axis=0) == 0)[0]

        n = 0
        todo = collections.deque(roots)
        done = numpy.zeros(data.shape[0], dtype=bool)

        while todo:
            # get next node to visit
            i = todo.popleft()
            # skip if done already
            if done[i]:
                continue
            # delay node visit if we didn't visit some of its children yet
            if not numpy.all(done[self.children(i)]):
                todo.append(i)
                continue
            # add node parents
            todo.extend(self.parents(i))
            # mark node as done
            done[i] = True
            path[n] = i
            n += 1

        assert n == len(path)
        assert (done != 0).sum() == data.shape[0]
        return path

    def parents(self, i: int) -> numpy.ndarray:
        """Get the parents of class ``i``.
        """
        return self._parents[i]

    def children(self, i: int) -> numpy.ndarray:
        """Get the children of class ``i``.
        """
        return self._children[i]

    def neighbors(self, i: int) -> numpy.ndarray:
        """Get the neighbors of class ``i``.
        """
        return numpy.cat([ self.parents(i), self.children(i) ])

    def ancestors(self, i: int) -> numpy.ndarray:
        """Get the entire tree of ancestors of class ``i``.
        """
        ancestors = set()
        nodes = collections.deque()
        nodes.append(i)
        while nodes:
            j = nodes.popleft()
            parents = self.parents(j)
            ancestors.update(parents)
            nodes.extend(parents)
        return numpy.array(list(ancestors))

    def descendants(self, i: int) -> numpy.ndarray:
        """Get the entire tree of descendants of class ``i``.
        """
        descendants = set()
        nodes = collections.deque()
        nodes.append(i)
        while nodes:
            j = nodes.popleft()
            children = self.children(j)
            descendants.update(children)
            nodes.extend(children)
        return numpy.array(list(descendants))


class Ontology:
    """An ontology represented as a Directed Acyclic Graph.

    The ontology is encoded using an adjacency matrix representing the graph
    of subclasses. It supports computing the similarity between terms or
    term groups based on the GOGO algorithm by Zhao and Wang.

    Attributes:
        adjacency_matrix (`AdjacencyMatrix`): The adjacency matrix encoding
            the class hierarchy in the ontology.
        semantic_similarity (`numpy.ndarray`): The semantic similarity of
            the ontology classes, which estimates the shared information
            between a class and its direct subclasses based on the number
            of total subclasses.

    References:
        Zhao, C., Wang, Z. "GOGO: An improved algorithm to measure the
        semantic similarity between gene ontology terms".
        Sci Rep 8, 15107 (2018). :doi:`10.1038/s41598-018-33219-y`

    """

    def __init__(
        self,
        adjacency_matrix: Union["AdjacencyMatrix", numpy.ndarray, "spmatrix"],
    ):
        # convert input to adjacency matrix if needed
        if not isinstance(adjacency_matrix, AdjacencyMatrix):
            adjacency_matrix = AdjacencyMatrix(adjacency_matrix)
        self.adjacency_matrix = adjacency_matrix
        if len(adjacency_matrix) > 1:
            self.semantic_similarity = _compute_semantic_similarity(adjacency_matrix)
        else:
            self.semantic_similarity = numpy.eye(1)

    @requires("scipy.sparse")
    def __getstate__(self) -> Dict[str, object]:
        return {
            "adjacency_matrix": self.adjacency_matrix.__getstate__(),
            "semantic_similarity": scipy.sparse.csr_matrix(self.semantic_similarity),
        }

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.semantic_similarity = state["semantic_similarity"].toarray()
        self.adjacency_matrix = AdjacencyMatrix()
        self.adjacency_matrix.__setstate__(state["adjacency_matrix"])

    def similarity(self, x: Iterable[int], y: Iterable[int]) -> float:
        """Return the semantic similarity between two set of terms.
        """
        _x = numpy.asarray(x)
        _y = numpy.asarray(y)
        sx = self.semantic_similarity[_x][:, _y].max(initial=0, axis=0).sum()
        sy = self.semantic_similarity[_y][:, _x].max(initial=0, axis=0).sum()
        return (sx+sy) / (_x.shape[0] + _y.shape[0])