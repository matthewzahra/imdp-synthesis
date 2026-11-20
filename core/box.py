import numpy as np


class Box:
    '''
    Axis-aligned box [lower, upper] in R^n, optionally with a name per dimension.

    Benchmarks use boxes for the state space, the input space, and the goal, critical, and other regions. A region
    usually constrains only some dimensions; restrict() copies a box (typically the state space) and narrows the
    dimensions it names:

        state_space = Box([-20, -2, -20, -2], [20, 2, 20, 2], names=['x_pos', 'x_vel', 'y_pos', 'y_vel'])
        goal = state_space.restrict(x_pos=(12, 18), y_pos=(12, 18))   # velocities keep the state-space bounds

    The bounds are read-only float arrays.
    '''

    def __init__(self, lower, upper, names=None):
        '''
        :param lower: Lower bound per dimension
        :param upper: Upper bound per dimension
        :param names: Name per dimension (optional); restrict() refers to dimensions by these names
        '''
        lower = np.array(lower, dtype=float).reshape(-1)
        upper = np.array(upper, dtype=float).reshape(-1)
        if lower.shape != upper.shape:
            raise ValueError(f'Box bounds differ in dimension: lower {lower.shape}, upper {upper.shape}.')
        if np.any(lower > upper):
            raise ValueError(f'Box lower bound exceeds its upper bound: lower {lower}, upper {upper}.')
        if names is not None:
            names = tuple(names)
            if len(names) != len(lower):
                raise ValueError(f'Box has {len(lower)} dimensions but {len(names)} names: {names}.')
        lower.flags.writeable = False
        upper.flags.writeable = False
        self.lower, self.upper, self.names = lower, upper, names

    @property
    def dim(self):
        return len(self.lower)

    @property
    def center(self):
        return (self.lower + self.upper) / 2

    @property
    def width(self):
        return self.upper - self.lower

    def index(self, dim):
        '''Index of a dimension given by its name or index.'''
        if isinstance(dim, str):
            if self.names is None or dim not in self.names:
                raise KeyError(f"Box has no dimension named '{dim}' (names: {self.names}).")
            return self.names.index(dim)
        if not 0 <= dim < self.dim:
            raise IndexError(f'Dimension {dim} out of range for a box of dimension {self.dim}.')
        return dim

    def restrict(self, bounds=None, /, **named_bounds):
        '''
        Copy of this box with new bounds (lower, upper) in some dimensions; the other dimensions keep their bounds.

        :param bounds: Dict from dimension (name or index) to (lower, upper)
        :param named_bounds: Bounds by dimension name, e.g., restrict(x_pos=(12, 18))
        :return: Box with the same names
        '''
        lower, upper = self.lower.copy(), self.upper.copy()
        for dim, (low, high) in {**(bounds or {}), **named_bounds}.items():
            i = self.index(dim)
            lower[i], upper[i] = low, high
        return Box(lower, upper, self.names)

    def contains(self, x):
        '''Whether the point x lies in the box (bounds included).'''
        x = np.asarray(x)
        return bool(np.all((x >= self.lower) & (x <= self.upper)))

    @staticmethod
    def stack(boxes, dim):
        '''
        Bounds of a list of boxes as one array, for vectorized code.

        :param boxes: List of boxes of dimension dim (may be empty)
        :param dim: Dimension of the boxes
        :return: Array of shape [len(boxes), 2, dim], with [i, 0] the lower and [i, 1] the upper bound of box i
        '''
        for box in boxes:
            if box.dim != dim:
                raise ValueError(f'Expected boxes of dimension {dim}, got {box.dim}.')
        return np.array([[box.lower, box.upper] for box in boxes], dtype=float).reshape(-1, 2, dim)

    def __eq__(self, other):
        return (isinstance(other, Box) and np.array_equal(self.lower, other.lower)
                and np.array_equal(self.upper, other.upper) and self.names == other.names)

    __hash__ = None

    def __repr__(self):
        names = '' if self.names is None else f', names={list(self.names)}'
        return f'Box({self.lower.tolist()}, {self.upper.tolist()}{names})'
