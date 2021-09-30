import pyemma
import numpy as np
import torch
import matplotlib.pyplot as plt



def compute_internal_coordinate_features(baseline_model, dataset):
    # compute distances all of the beads
    baseline_model.cpu()
    if isinstance(dataset, np.ndarray):
        traj = dataset
        n_traj, n_samp, n_beads, _ = traj.shape
        features = []
        for i_traj in range(n_traj):
            _ = baseline_model.geom_feature(torch.from_numpy(traj[i_traj]))
            feat = baseline_model.geom_feature.distances
            features.append(feat)
    else:
        _ = baseline_model.geom_feature(dataset.data.pos.reshape((-1, baseline_model.n_beads, 3)))
        feat = baseline_model.geom_feature.distances

        if 'traj_idx' in dataset.data:
            traj_ids = dataset.data.traj_idx
            n_traj = np.unique(traj_ids).shape[0]
            traj_strides = np.cumsum([0]+(np.bincount(traj_ids)).tolist(), dtype=int)

            features = []
            for i_traj in range(n_traj):
                st, nd = traj_strides[i_traj], traj_strides[i_traj+1]
                features.append(feat[st:nd].numpy())
        else:
            features = feat.numpy()
    return features

def project_tica(features, lag=10, tica=None):
    if tica is None:
        tica = pyemma.coordinates.tica(features, lag=lag, dim=2)
        dimred_features = tica.get_output()
    else:
        dimred_features = tica.transform(features)

    return dimred_features, tica

def plot_tica(tica, dimred_features):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    tica_concatenated = np.concatenate(dimred_features)
    pyemma.plots.plot_feature_histograms(
        tica_concatenated, ['IC {}'.format(i + 1) for i in range(tica.dimension())], ax=axes[0])
    # pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[1], cbar=False, logscale=True)
    pyemma.plots.plot_free_energy(*tica_concatenated[:, :2].T, ax=axes[1], legacy=False)
    for ax in axes.flat[1:]:
        ax.set_xlabel('IC 1')
        ax.set_ylabel('IC 2')
    fig.tight_layout()
    return fig, axes

