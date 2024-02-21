import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import numpy as np
import cupoch as cph
import argparse
from sklearn.metrics import silhouette_score
import open3d as o3d
from open3d import geometry
from torchclustermetrics import silhouette


'''
number of unique points
number of repeated observations

cluster - amount of valid points inside major cluster
amount of noise
silhouette
'''

columns = ['point_id', 'frame', 'x', 'y', 'z']


def get_file_path(name):
    file_path = '../Results/MapPoints_' + str(name) + '.csv'
    return file_path


def get_dataframe(name):
    dataframe = pd.read_csv(get_file_path(name), names=columns)
    return dataframe


def get_number_of_unique_points(dataframe):
    unique_points = dataframe['point_id'].nunique()
    return unique_points


def read_in_ply(input_file):
    pointcloud = cph.io.read_point_cloud("../Visualize/" + input_file)
    return pointcloud

def get_silhouette(pointcloud, labels):
    original_data = np.array(pointcloud.points.cpu())
    silhouette_avg = silhouette_score(original_data, labels)
    return silhouette_avg

def cluster_old(pointcloud, eps=0.1, min_points=10):
    labels = np.array(pointcloud.cluster_dbscan(eps=eps, min_points=min_points).cpu())
    max_label = labels.max()
    print("Point cloud has %d clusters" % (max_label + 1))

    # Count points in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Prepare data for export
    cluster_data = {'ClusterLabel': [], 'NumPoints': []}

    for label, count in zip(unique_labels, counts):
        cluster_data['ClusterLabel'].append(label)
        cluster_data['NumPoints'].append(count)

        print(f"Cluster {label} has {count} points")

    # Create a DataFrame for export
    cluster_df = pd.DataFrame(cluster_data)
    return labels, cluster_df


def cluster(pointcloud, eps=0.1, min_points=10):
    labels = np.array(pointcloud.cluster_dbscan(eps=eps, min_points=min_points).cpu())
    max_label = labels.max()
    print("Point cloud has %d clusters" % (max_label + 1))

    # Count points in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Prepare data for export
    cluster_data = {'ClusterLabel': [], 'NumPoints': []}

    for label, count in zip(unique_labels, counts):
        cluster_data['ClusterLabel'].append(label)
        cluster_data['NumPoints'].append(count)

        print(f"Cluster {label} has {count} points")

    # Create a DataFrame for export
    cluster_df = pd.DataFrame(cluster_data)

    # Calculate Silhouette Score using the original data
    silhouette_avg = get_silhouette(pointcloud, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Add Silhouette Score to the DataFrame
    cluster_df['SilhouetteScore'] = silhouette_avg

    return labels, cluster_df



def visualize(pointcloud, labels):
    cmap = plt.get_cmap("viridis")
    max_label = labels.max()
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    pointcloud.colors = cph.utility.Vector3fVector(colors[:, :3])
    cph.visualization.draw_geometries([pointcloud])


def data_mining(name):
    df = get_dataframe(name)
    unique_points = get_number_of_unique_points(df)


def save_results(df_data, df_cluster, output_csv_file):
    df_data.to_csv(output_csv_file + '.csv', index=False)

def format_points(df):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse files')
    parser.add_argument('input_csv_file', help='Input CSV file')
    parser.add_argument('input_ply_file', help='Input PLY file')
    parser.add_argument('eps', type=float, help='eps - Parameter for DBSCAN')
    parser.add_argument('min_points', type=int, help='min_points - Parameter for DBSCAN')
    parser.add_argument('output_csv_file', help='Output CSV filepath')

    args = parser.parse_args()

    pointcloud = read_in_ply(args.input_ply_file)
    labels, clusterdf = cluster(pointcloud, eps=args.eps, min_points=args.min_points)
