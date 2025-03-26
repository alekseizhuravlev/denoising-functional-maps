import torch


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p


def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)


def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21


def zoomout(FM_12, evects1, evects2, nit, step, A2=None):
    """
    Perform the ZoomOut algorithm to progressively refine a functional map.

    This iterative algorithm refines the functional map by expanding the basis and
    updating the map with each iteration.

    Args:
        FM_12 (torch.Tensor): Initial functional map of shape [K, K].
        evects1 (torch.Tensor): Eigenvectors of shape 1, of shape [V1, K].
        evects2 (torch.Tensor): Eigenvectors of shape 2, of shape [V2, K].
        nit (int): Number of iterations to perform.
        step (int): Number of new basis functions to add at each iteration.
        A2 (torch.Tensor, optional): Vertex-area matrix of shape [V2] or mass matrix of shape [V2, V2].

    Returns:
        torch.Tensor: Refined functional map of shape [(K + nit * step), (K + nit * step)].
    """

    # Initialize the basis size with the current functional map dimensions
    k = FM_12.shape[0]

    for _ in range(nit):
        # Convert the current functional map to a point-to-point map
        p2p_21 = fmap2pointmap(FM_12, evects1[:, :k], evects2[:, :k])

        # Increase the basis size by the specified step
        k += step

        if A2 is not None:
            # If the vertex-area matrix A2 is provided, use it for area-weighted projection
            if A2.ndim == 1:
                # Handle the case where A2 is a vector (diagonal mass matrix)
                FM_12 = evects2[:, :k].T @ (A2[:, None] * evects1[p2p_21, :k])
            else:
                # Handle the case where A2 is a full mass matrix
                FM_12 = evects2[:, :k].T @ (A2 @ evects1[p2p_21, :k])
        else:
            # If no area matrix is provided, perform a standard least-squares projection
            FM_12 = pointmap2fmap(p2p_21, evects2[:, :k], evects1[:, :k])

    return FM_12
