import collections
import typing
from typing import Optional, Iterator, Union, Sequence, Dict

import numpy
import scipy.sparse


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


class TreeMatrix:
    """A tree encoded as an incidence matrix.
    """

    def __init__(self, data: Optional[numpy.ndarray] = None) -> None:
        if data is None:
            _data = numpy.array([])
        elif isinstance(data, scipy.sparse.spmatrix):
            _data = data.toarray()
        else:
            _data = numpy.asarray(data, dtype=numpy.int32)

        if _data.size:
            # cache parents and children
            self._parents = _VariableColumnStorage(
                [numpy.where(_data[i, :] != 0)[0] for i in range(_data.shape[0])],
                dtype=numpy.int64
            )
            self._children = _VariableColumnStorage(
                [numpy.where(_data[:, i] != 0)[0] for i in range(_data.shape[0])],
                dtype=numpy.int64
            )
            # run a BFS walk and store indices
            self._down = self._roots_to_leaves(_data)
            self._up = self._leaves_to_root(_data)
        else:
            self._parents = self._children = _VariableColumnStorage([])
            self._down = self._up = numpy.array([])

        # store data as a sparse matrix
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
        return self._parents[i]

    def children(self, i: int) -> numpy.ndarray:
        return self._children[i]

    def neighbors(self, i: int) -> numpy.ndarray:
        return numpy.cat([ self.parents(i), self.children(i) ])