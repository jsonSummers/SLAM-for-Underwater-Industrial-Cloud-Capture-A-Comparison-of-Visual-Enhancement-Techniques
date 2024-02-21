import argparse
import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement
import torch
import open3d as o3d
from open3d import geometry
import cupoch as cph

torch.cuda.empty_cache()


def get_dataframe(filepath):
    columns = ['point_id', 'frame', 'x', 'y', 'z']
    dataframe = pd.read_csv(filepath, names=columns)
    return dataframe


def read_in_ply(input_file):
    pointcloud = cph.io.read_point_cloud("../Visualize/" + input_file)
    return pointcloud


def cluster(pointcloud, eps=0.1, min_points=10):
    labels = np.array(pointcloud.cluster_dbscan(eps=eps, min_points=min_points).cpu())
    max_label = labels.max()

    # Count points in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Prepare data for export
    cluster_data = {'ClusterLabel': [], 'NumPoints': []}

    for label, count in zip(unique_labels, counts):
        cluster_data['ClusterLabel'].append(label)
        cluster_data['NumPoints'].append(count)

    # Create a DataFrame for export
    cluster_df = pd.DataFrame(cluster_data)
    return labels, cluster_df


def main():
    parser = argparse.ArgumentParser(description='Analyze files')
    parser.add_argument('input_csv_file', help='Input CSV file')
    parser.add_argument('input_ply_file', help='Input CSV file')
    parser.add_argument('eps', type=float, help='eps - Parameter for DBSCAN')
    parser.add_argument('min_points', type=int, help='min_points - Parameter for DBSCAN')
    parser.add_argument('silhouette', type=bool, help='Generate silhouette score (warning long)')
    parser.add_argument('output_csv_file', help='Output CSV filepath')
    args = parser.parse_args()

    data = get_dataframe(args.input_csv_file)
    pointcloud = read_in_ply(args.input_ply_file)

    labels, cluster_df = cluster(pointcloud, eps=args.eps, min_points=args.min_points)
    data['ClusterLabel'] = labels
    if args.silhouette:
        print("warning")
    else:
        print("no score")
    data.to_csv(args.output_csv_file, index=True)


if __name__ == "__main__":
    main()
