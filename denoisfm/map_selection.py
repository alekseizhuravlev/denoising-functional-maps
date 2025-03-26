import torch


def select_p2p_map(Pi_21_list, verts_1, L_2, dist_1, num_samples_selection):
    
    """
    Selects the best point-to-point (p2p) map by computing the Dirichlet energy
    and returns a medoid p2p map based on the lowest energy maps.

    Args:
        Pi_21_list (torch.Tensor): Tensor of shape [N, V_2] where N is the number of p2p maps
            and V is the number of vertices. Each row represents a mapping.
        verts_1 (torch.Tensor): Tensor of shape [V_1, 3] representing the vertex positions of mesh 1.
        L_2 (torch.Tensor): Tensor of shape [V_2, V_2] representing the Laplacian matrix of mesh 2.
        dist_1 (torch.Tensor): Tensor of shape [V_1, V_1] representing the geodesic distance matrix of mesh 1.
        num_samples_selection (int): Number of p2p maps with the lowest Dirichlet energy to consider
            for computing the medoid map.

    Returns:
        torch.Tensor: Medoid p2p map of shape [V_2] representing the best mapping.
    """
    
    # dirichlet energy for each p2p map
    dirichlet_energy_list = []
    for n in range(Pi_21_list.shape[0]):
        dirichlet_energy_list.append(
            dirichlet_energy(Pi_21_list[n], verts_1, L_2).item(),
        )
    dirichlet_energy_list = torch.tensor(
        dirichlet_energy_list, device=Pi_21_list.device
    )

    # sort by dirichlet energy, get the arguments
    _, sorted_idx_dirichlet = torch.sort(dirichlet_energy_list)

    # map with the lowest dirichlet energy
    # p2p_dirichlet = Pi_21_list[sorted_idx_dirichlet[0]]

    # medoid p2p map, using e.g. 16 maps with lowest dirichlet energy
    Pi_21_medoid = get_medoid_p2p_map(
        Pi_21_list[sorted_idx_dirichlet[:num_samples_selection]], dist_1
    )

    return Pi_21_medoid


def get_medoid_p2p_map(Pi_21_candidates, dist_1):
    """
    Computes the medoid point-to-point (p2p) map by minimizing the sum of geodesic
    distances between candidate points.

    Args:
        Pi_21_candidates (torch.Tensor): Tensor of shape [N, V_2], where N is the number of maps
            and V is the number of vertices. Each row represents a p2p map.
        dist_1 (torch.Tensor): Tensor of shape [V_1, V_1] representing the geodesic distance matrix
            of mesh 1.

    Returns:
        torch.Tensor: Medoid p2p map of shape [V_2].
    """
    
    assert len(Pi_21_candidates.shape) == 2

    Pi_21_medoid = torch.zeros(Pi_21_candidates.shape[1], dtype=torch.int64)

    for i in range(Pi_21_candidates.shape[1]):
        # get potential matching points in mesh 1, e.g. #10, #15, #20 ...
        vertex_indices = Pi_21_candidates[:, i]

        # get the geodesic distances between these points
        geo_dists_points = dist_1[vertex_indices][:, vertex_indices]

        # find the medoid point,
        # the one with the smallest sum of geodesic distances to all other points
        idx_medoid = vertex_indices[torch.argmin(geo_dists_points.sum(axis=1))]
        Pi_21_medoid[i] = idx_medoid

    return Pi_21_medoid


def dirichlet_energy(Pi_21, verts_1, L_2):
    """
    Computes the Dirichlet energy for a given point-to-point (p2p) map.

    Args:
        Pi_21 (torch.Tensor): Tensor of shape [V_2] representing the p2p map.
        verts_1 (torch.Tensor): Tensor of shape [V_1, 3] representing the vertex positions of mesh 1.
        L_2 (torch.Tensor): Tensor of shape [V_2, V_2] representing the Laplacian matrix of mesh 2.

    Returns:
        torch.Tensor: Scalar value representing the Dirichlet energy of the map.

    Formula:
        E(Pi) = Tr((verts_1[Pi])^T * L_2 * verts_1[Pi])
    """

    assert len(Pi_21.shape) == 1
    assert len(verts_1.shape) == 2
    assert len(L_2.shape) == 2

    mapped_verts = verts_1[Pi_21]

    dirichlet_energy = torch.trace(mapped_verts.transpose(0, 1) @ L_2 @ mapped_verts)

    return dirichlet_energy
