import pandas as pd
import numpy as np
import cupoch as cph
from cupoch import geometry
from sklearn.metrics import silhouette_score
from torchclustermetrics import silhouette

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

def get_silhouette(dataframe):
    points = dataframe[['x', 'y', 'z']].values
    labels = dataframe['ClusterLabel']
    score = silhouette_score(points, labels)
    return score
