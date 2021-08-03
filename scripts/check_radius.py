import argparse
from tqdm import tqdm
from torchmdnet import datasets
from torch_cluster import radius_graph
from torch_geometric.data import DataLoader


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Check number of neighbors inside cutoff')
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='~/data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset argument, e.g. target property for QM9 or molecule for MD17')
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')

    parser.add_argument('--cutoff-upper', type=float, default=15.0, help='Upper cutoff in model')
    parser.add_argument('--batch-size', type=int, default=2048, help='Number of samples per batch')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the radius_graph function on')
    # fmt: on

    return parser.parse_args()


def main():
    args = get_args()

    print("Loading data...")
    if args.dataset == "Custom":
        data = datasets.Custom(
            args.coord_files, args.embed_files, args.energy_files, args.force_files,
        )
    else:
        data = getattr(datasets, args.dataset)(
            args.dataset_root, dataset_arg=args.dataset_arg
        )
    dl = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers)

    errors = 0
    progress = tqdm(dl, desc="checking for errors")
    for batch in progress:
        batch.to(args.device)

        # check with large max_num_neighbors
        edge_index1 = radius_graph(
            batch.pos,
            args.cutoff_upper,
            batch=batch.batch,
            loop=True,
            max_num_neighbors=1000000,
        )
        # check with default max_num_neighbors (32)
        edge_index2 = radius_graph(
            batch.pos, args.cutoff_upper, batch=batch.batch, loop=True,
        )

        errors += edge_index1.shape != edge_index2.shape
        progress.set_postfix(dict(errors=errors))

    print(
        f"{errors} out of {len(dl)} batches ({errors / len(dl):.2%}) contain an error"
    )


if __name__ == "__main__":
    main()
