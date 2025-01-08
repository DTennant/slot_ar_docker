import argparse
import os
import torch
import torch_fidelity
from tqdm import tqdm

def calculate_metrics(root):
    
    fid_statistics_file = f'fid_stats/{os.path.basename(root)}.npz'

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=os.path.join(root, 'inputs'),
        input2=os.path.join(root, 'outputs'),
        fid_statistics_file=fid_statistics_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )

    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']

    print(f"FID: {fid:.4f}")
    print(f"Inception Score: {inception_score:.4f}")

    return fid, inception_score

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fid, is_score = calculate_metrics(args.root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID and Inception Score")
    parser.add_argument("--root", type=str, required=True, help="Path to the root directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    main(args)