import numpy as np

def lddt_np_dist(
    dmat_predicted, dmat_true, true_points_mask, cutoff=15.0, per_residue=False
):
    """Measure (approximate) lDDT for a batch of coordinates.
    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).
    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.
    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.
    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
        cutoff: Maximum distance for a pair of points to be included
        per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.
    Returns:
        An (approximate, see above) lDDT score in the range 0-1.
    """
    dists_to_score = (
        (dmat_true < cutoff).astype(np.float32)
        * true_points_mask
        * np.transpose(true_points_mask, [0, 2, 1])
        * (1.0 - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )
    # Shift unscored distances to be far away.
    dist_l1 = np.abs(dmat_true - dmat_predicted)
    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * (
        (dist_l1 < 0.5).astype(np.float32)
        + (dist_l1 < 1.0).astype(np.float32)
        + (dist_l1 < 2.0).astype(np.float32)
        + (dist_l1 < 4.0).astype(np.float32)
    )
    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))
    return score

def torsion_angle_difference(
    angles_1,  # [N, 3]
    angles_2,  # [N, 3]
):
    angles_1 = angles_1 % 360
    angles_2 = angles_2 % 360
    diff = angles_1 - angles_2
    diff = np.abs((diff + 180) % 360 - 180)
    mask = np.isnan(diff)
    diff[mask] = 0
    per_residue = diff.sum(-1) / (~mask).sum(-1)
    total = per_residue.mean()
    return per_residue, total