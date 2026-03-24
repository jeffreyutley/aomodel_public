import abc
import numpy as np

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

class PredictionWindowStructure(abc.ABC):
    """
    Abstract base class for all prediction window structures.
    """
    @abc.abstractmethod
    def get_mask(self, vector_dimensionality):
        """
        Create the prediction window mask, a boolean numpy 2-D array of shape
        (``vector_dimensionality``, ``vector_dimensionality``).

        Args:
            vector_dimensionality (int): the dimensionality of the vector space to use.

        Returns:
            **mask** (*ndarray*) -- the prediction window mask.
        """
        pass


class FullWindowStructure(PredictionWindowStructure):
    """
    Include all possible prediction window indices for each predicted component.

    Args:
        predicted_components (ndarray, optional): [Default=None] numpy 1-D integer array containing the indices of the
            data vector's components for which the model will compute prediction weights.

            - If set to None, all vector components are predicted.
    """
    def __init__(self, predicted_components=None):
        self.predicted_components = predicted_components

    def get_mask(self, vector_dimensionality):
        mask = np.zeros((vector_dimensionality, vector_dimensionality), dtype=bool)
        mask[self.predicted_components, :] = True
        return mask


class SubspaceWindowStructure(PredictionWindowStructure):
    """
    Restrict prediction to a sub-set of the first ``subspace_dimension`` components.

    Args:
        subspace_dimension (int): the subspace dimension to restrict the prediction window to.
    """
    def __init__(self, subspace_dimension):
        self.subspace_dimension = subspace_dimension

    def get_mask(self, vector_dimensionality):
        assert 0 < self.subspace_dimension <= vector_dimensionality
        mask = np.zeros((vector_dimensionality, vector_dimensionality), dtype=bool)
        mask[:self.subspace_dimension, :self.subspace_dimension] = True
        return mask


class DistanceWindowStructure(PredictionWindowStructure):
    """
    For each predicted component, structure the prediction window to include only the "neighboring" vector components
    with a certain distance.

    Args:
        distance (int): the distance to restrict the prediction window to.

        predicted_components (ndarray, optional): [Default=None] numpy 1-D integer array containing the indices of the
            data vector's components for which the model will compute prediction weights.

            - If set to None, all vector components are predicted.
    """
    def __init__(self, distance, predicted_components=None):
        self.distance = distance
        self.predicted_components = predicted_components

    def get_mask(self, vector_dimensionality):
        # Maximum radius that avoids ambiguity:
        max_valid_distance = (vector_dimensionality // 2) - 1 + (vector_dimensionality % 2)
        if not (1 <= self.distance <= max_valid_distance):
            raise ValueError(
                f"'prediction_window_distance' must be in [1, {max_valid_distance}] "
                f"for vector_dimensionality={vector_dimensionality}; got {self.distance}."
            )
        # If the predicted component indices are not provided, predicts all vector components:
        if self.predicted_components is None:
            self.predicted_components = np.arange(vector_dimensionality)

        mask = np.zeros((vector_dimensionality, vector_dimensionality), dtype=bool)

        all_component_indices = np.arange(vector_dimensionality)[None, :]  # shape (1, vector_dimensionality)

        # Compute distances moving "forward" through the vector components:
        forward_distances = (all_component_indices - self.predicted_components[:, None]) % vector_dimensionality

        # Find the shortest distances between vector components:
        # Returns array of shape (number of predicted components, vector_dimensionality).
        shortest_distances = np.minimum(forward_distances, vector_dimensionality - forward_distances)

        # Restrict the prediction window mask to only include components in the prediction window distance:
        mask[self.predicted_components, :] = (shortest_distances <= self.distance)
        return mask


class ExplicitWindowStructure(PredictionWindowStructure):
    """
    Creates the prediction window structure from an explicit prediction window mask.

    Args:
        mask (ndarray): the prediction window mask.
    """
    def __init__(self, mask):
        self.mask = mask

    def get_mask(self, vector_dimensionality):
        assert (self.mask.shape == (vector_dimensionality, vector_dimensionality))
        assert (self.mask.dtype == bool)
        return self.mask