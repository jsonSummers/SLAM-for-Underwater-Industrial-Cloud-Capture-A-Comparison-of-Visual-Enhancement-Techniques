import os
import pandas as pd
from plyfile import PlyData, PlyElement
import argparse
import numpy as np

def csv_to_ply(input_file, output_file):
    # Read CSV file into a DataFrame
    df = pd.read_csv(input_file, header=None, names=['id', 'frame', 'x', 'y', 'z'])

    # Create PLY file
    vertex_data = [(df['x'][i], df['y'][i], df['z'][i]) for i in range(len(df))]
    vertices = [(x, y, z) for x, y, z in vertex_data]

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices_array = np.array(vertices, dtype=vertex_dtype)

    vertex_elements = PlyElement.describe(vertices_array, 'vertex')

    output_file_path = os.path.join(os.getcwd(), output_file)
    PlyData([vertex_elements]).write(output_file_path)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Convert CSV to PLY')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('output_file', help='Output PLY file')

    args = parser.parse_args()

    csv_to_ply(args.input_file, args.output_file)

