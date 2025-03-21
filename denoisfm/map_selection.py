import torch


def select_p2p_map(p2p_est_zo_sampled, verts_first, L_second, dist_first, num_samples_selection):

    # dirichlet energy for each p2p map
    dirichlet_energy_list = []
    for n in range(p2p_est_zo_sampled.shape[0]):
        dirichlet_energy_list.append(
            dirichlet_energy(p2p_est_zo_sampled[n], verts_first, L_second).item(),
            )
    dirichlet_energy_list = torch.tensor(dirichlet_energy_list, device=p2p_est_zo_sampled.device)

    # sort by dirichlet energy, get the arguments
    _, sorted_idx_dirichlet = torch.sort(dirichlet_energy_list)
    
    # map with the lowest dirichlet energy
    # p2p_dirichlet = p2p_est_zo_sampled[sorted_idx_dirichlet[0]]
    
    # medoid p2p map, using 3 maps with lowest dirichlet energy
    p2p_medoid = get_medoid_p2p_map(
        p2p_est_zo_sampled[
            sorted_idx_dirichlet[:num_samples_selection]
            ],
        dist_first
        )
    
    return p2p_medoid


def get_medoid_p2p_map(p2p_maps, dist_x):
    
    assert len(p2p_maps.shape) == 2, "p2p_maps should be [n, dist_x.shape[0]]"
    
    medoid_p2p_map = torch.zeros(p2p_maps.shape[1], dtype=torch.int64)

    for i in range(p2p_maps.shape[1]):
    
        vertex_indices = p2p_maps[:, i]
        
        geo_dists_points = dist_x[vertex_indices][:, vertex_indices]

        # find the medoid point,
        # the one with the smallest sum of geodesic distances to all other points
        idx_medoid = vertex_indices[
            torch.argmin(geo_dists_points.sum(axis=1))
        ]
        medoid_p2p_map[i] = idx_medoid
        
    return medoid_p2p_map
    
    
def dirichlet_energy(p2p_12, verts_2, L_1):
    """
    p2p_12: point-to-point map from mesh 1 to mesh 2
    verts_2: vertices of mesh 2
    L_1: Laplacian of mesh 1
    """
 
    assert len(p2p_12.shape) == 1
    assert len(verts_2.shape) == 2
    assert len(L_1.shape) == 2
    
    mapped_verts = verts_2[p2p_12]
    
    dirichlet_energy = torch.trace(mapped_verts.transpose(0, 1) @ L_1 @ mapped_verts)
    
    return dirichlet_energy
    