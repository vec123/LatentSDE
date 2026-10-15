"""
Feature libraries for sparse drift models: polynomials, polynomials of sines and cosines, polynomials of
neighbouring states, and precomputed features. A sparse_index selects the features that are kept.

Adapted from SVISE, reference/svise/svise/variationalsparsebayes/sparse_glm.py (GPL-3.0). Changes: imports, and
SparsePolynomialFeatures computes all monomials with one gather and one product (SVISE: one product per monomial).
Feature order: all monomials in graded order, as the old JAX MultivariatePolynomialLibrary.
"""
from abc import ABC, abstractmethod
from itertools import chain, combinations_with_replacement, tee
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class SparseFeaturesLibrary(ABC, nn.Module):
    """
    Abstract base class for making sparse feature libraries
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update_basis(self, sparse_index: Tensor) -> None:
        assert (
            self.num_features >= sparse_index.max()
        ), "More sparsity inducing indices than features."
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        pass


class SparsePrecomputedFeatures(SparseFeaturesLibrary):
    def __init__(self, num_features: int, input_labels: np.ndarray = None) -> None:
        super().__init__()
        self.num_features = num_features
        if input_labels is None:
            self.feature_names = np.array(
                ["x_{}".format(i) for i in range(num_features)]
            )
        else:
            assert (
                len(input_labels) == num_features
            ), "Number of labels should be the same as number of inputs."
            self.feature_names = input_labels
        self.sparse_index = torch.arange(self.num_features)

    @property
    def feature_names(self) -> List[str]:
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, value):
        self.__feature_names = value

    @property
    def num_features(self) -> int:
        return self.__num_features

    @num_features.setter
    def num_features(self, value):
        self.__num_features = value

    def update_basis(self, sparse_index: Tensor) -> None:
        """
        Updates the spares_index 

        Args:
            sparse_index (Tensor): sparsity inducing indices 
        """
        super().update_basis(sparse_index)
        self.sparse_index = sparse_index

    def __str__(self) -> str:
        return " + ".join(self.feature_names[self.sparse_index])

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., num_features)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        return x[..., self.sparse_index]


class Polynomial:
    """
    Class for computing polynomial feature of a given degree
    """

    def __init__(self, ind) -> None:
        self.ind = ind

    def __call__(self, x: Tensor) -> Tensor:
        return x[..., self.ind].prod(-1)


class SparsePolynomialFeatures(SparseFeaturesLibrary):
    """
    Polynomial features library for use with sparse models. Draws heavily from scikit learn implementation.
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        include_bias: bool = True,
        input_labels: List[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.degree = degree
        start = int(not include_bias)
        iter_comb = chain.from_iterable(
            combinations_with_replacement(range(dim), i)
            for i in range(start, degree + 1)
        )
        # create two copies of iterable
        iter_comb_1, iter_comb_2 = tee(iter_comb)
        # array of functions
        self.full_polynomial = np.array([Polynomial(ind) for ind in iter_comb_1])
        self.num_features = len(self.full_polynomial)
        self.register_buffer("sparse_index", torch.arange(self.num_features))
        # getting names (taken directly from sklearn)
        if input_labels is None:
            self.input_labels = ["x_{}".format(i) for i in range(dim)]
        else:
            assert len(input_labels) == dim
            self.input_labels = input_labels
        self.feature_names = self._compute_feature_names(iter_comb_2)

    @property
    def feature_names(self) -> List[str]:
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, value):
        self.__feature_names = value

    @property
    def num_features(self) -> int:
        return self.__num_features

    @num_features.setter
    def num_features(self, value):
        self.__num_features = value

    def update_basis(self, sparse_index: Tensor) -> None:
        """
        Updates the spares_index 

        Args:
            sparse_index (Tensor): sparsity inducing indices 
        """
        super().update_basis(sparse_index)
        self.sparse_index = sparse_index

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., dim)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        x_ext = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        return x_ext[..., self._monomial_index(x.device)[self.sparse_index]].prod(-1)

    def _monomial_index(self, device) -> Tensor:
        """
        (num_features, degree) indices into x with a 1 appended (index dim), one row per monomial, so that a
        monomial is one product. Built from full_polynomial on first use, also in models saved without it.
        """
        index = getattr(self, "_monomial_index_cache", None)
        if index is None or index.device != device:
            rows = [p.ind + (self.dim,) * (self.degree - len(p.ind)) for p in self.full_polynomial]
            index = torch.tensor(rows, dtype=torch.long, device=device).reshape(len(rows), self.degree)
            self._monomial_index_cache = index
        return index

    def __str__(self) -> str:
        return " + ".join(self.feature_names[self.sparse_index])

    def _compute_feature_names(self, iter_comb: Iterable) -> np.ndarray:
        """
        Compute a list of feature names. This function is taken directly from the sklearn implementation

        Args:
            iter_comb (Iterable): combination iterable for polynomials

        Returns:
            List[str]: list of strings containing feature names
        """
        powers = np.vstack([np.bincount(c, minlength=self.dim) for c in iter_comb])
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join(
                    "%s^%d" % (self.input_labels[ind], exp)
                    if exp != 1
                    else self.input_labels[ind]
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return np.array(feature_names)


# TODO: unit test this class
class SparsePolynomialSinusoidTfm(SparsePolynomialFeatures):
    """
    Feature library that includes polynomials and sinusoids transform of input features.
    ex: ["x", "y"] -> ["1", "x", "y", "sin(x)", "cos(x)", "sin(y)", "cos(y)", "x^2", "y^2", ... ]

    """

    def __init__(
        self,
        dim: int,
        degree: int,
        include_bias: bool = True,
        input_labels: List[str] = None,
    ):
        # note, I don't think we need a freq + phase argument as we can always scale x?
        if input_labels is None:
            input_labels = ["x_{}".format(i) for i in range(dim)]
        else:
            input_labels = [el for el in input_labels]
        sin_labels = [f"sin({il})" for il in input_labels]
        cos_labels = [f"cos({il})" for il in input_labels]
        input_labels += sin_labels + cos_labels
        super().__init__(dim * 3, degree, include_bias, input_labels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., dim)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        x_in = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        return super().forward(x_in)


# TODO: unit test this class
class SparsePolynomialNeighbour1D(SparsePolynomialFeatures):
    """
    Polynomial feature library for use with sparse models. Assumes that each input is a polynomial function of it's neighbours.

    ex: ["x_0", "x_1", "x_2"] = ["x_0", "x_1", "x_2", "x_0 x_1", "x_1 x_2", "x_0^2", "x_1^2", "x_2^2"]
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        include_bias: bool = True,
        input_labels: List[str] = None,
        num_neighbours: int = 2,
    ):
        assert dim >= 3
        self.num_neighbours = num_neighbours
        inds = [i - self.num_neighbours for i in range(2 * self.num_neighbours + 1)]
        self.inds = inds
        if input_labels is None:
            var_name = ["x"]
        else:
            assert len(input_labels) == 1
            var_name = input_labels[0]
        input_labels = [f"{var_name}_{{k{-i:+}}}" for i in inds]
        super().__init__(2 * num_neighbours + 1, degree, include_bias, input_labels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., dim)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        x_in = torch.stack([x.roll(i, dims=-1) for i in self.inds], dim=-1,)
        return super().forward(x_in)
