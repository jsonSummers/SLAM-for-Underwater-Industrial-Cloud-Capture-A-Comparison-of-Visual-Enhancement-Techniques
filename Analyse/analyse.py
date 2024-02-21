import pandas as pd
import sys

#import cudf

# conda activate ../../mnt/c/Users/Jason/anaconda3/envs/Visualize


def import_files(input_files):
    # Create an empty DataFrame to store concatenated data
    combined_data = pd.DataFrame()

    for input_file in input_files:
        # Extract experiment name and run number from the file path
        _, _, experiment, run, _ = input_file.split("/")
        run = int(run)

        # Read the CSV file into a DataFrame with specified data types, skipping the first row
        columns = ['point_id', 'frame', 'x', 'y', 'z', 'ClusterLabel']
        dtypes = {'point_id': int, 'frame': int, 'x': float, 'y': float, 'z': float, 'ClusterLabel': int}
        df = pd.read_csv(input_file, names=columns, dtype=dtypes, skiprows=1)

        # Add experiment and run columns to the DataFrame
        df['experiment'] = experiment
        df['run'] = run

        # Concatenate the current DataFrame with the combined data
        combined_data = pd.concat([combined_data, df], axis=0, ignore_index=True)
    return combined_data

def data_mine(df):
    # number of unique points that aren't noise
    unique_non_noise_points = df[df['ClusterLabel'] != -1]['point_id'].nunique()

    # number of noise points
    noise_points = df[df['ClusterLabel'] == -1]['point_id'].nunique()

    # number of repeated observations (same point_id) that aren't noise
    repeated_non_noise_observations = df[df['ClusterLabel'] != -1]['point_id'].duplicated().sum()
    return unique_non_noise_points, noise_points, repeated_non_noise_observations

def mine_loop(df):
    # Create an empty dataframe to store results
    results_df = pd.DataFrame(
        columns=['experiment', 'run', 'unique_non_noise_points', 'noise_points', 'repeated_non_noise_observations'])

    # Iterate over unique combinations of 'experiment' and 'run'
    for (experiment, run), group in df.groupby(['experiment', 'run']):
        unique_non_noise_points, noise_points, repeated_non_noise_observations = data_mine(group)

        # Append results to the results dataframe
        results_df = results_df.append({
            'experiment': experiment,
            'run': run,
            'unique_non_noise_points': unique_non_noise_points,
            'noise_points': noise_points,
            'repeated_non_noise_observations': repeated_non_noise_observations
        }, ignore_index=True)

    # Display the results dataframe
    return(results_df)

def averages(df):
    # Group by 'experiment' and calculate the average for each numeric column
    averages_df = df.groupby('experiment').mean().reset_index()

    return averages_df


def calculate_lifespans(df, frame_threshold):
    # Filter out points with ClusterLabel of -1
    df_filtered = df[df['ClusterLabel'] != -1]

    # Sort the dataframe by 'experiment', 'run', 'point_id', and 'frame'
    df_sorted = df_filtered.sort_values(by=['experiment', 'run', 'point_id', 'frame'])

    # Create a new dataframe to store the results
    result_df = pd.DataFrame(columns=['experiment', 'run', 'point_id', 'lifespan'])

    for (experiment, run), group in df_sorted.groupby(['experiment', 'run']):
        current_point_id = None
        first_frame = None
        last_frame_tracked = None

        for _, row in group.iterrows():
            if current_point_id is None or row['point_id'] != current_point_id:
                # Start tracking a new point
                current_point_id = row['point_id']
                first_frame = row['frame']
                last_frame_tracked = row['frame']
            elif row['frame'] - last_frame_tracked <= frame_threshold:
                # Update last_frame_tracked if the frame is within the threshold
                last_frame_tracked = row['frame']
            else:
                # Record the previous point's lifespan and reset tracking for a new point
                result_df = result_df.append({
                    'experiment': experiment,
                    'run': run,
                    'point_id': current_point_id,
                    'lifespan': last_frame_tracked - first_frame
                }, ignore_index=True)

                current_point_id = row['point_id']
                first_frame = row['frame']
                last_frame_tracked = row['frame']

        # Record the final point's lifespan
        result_df = result_df.append({
            'experiment': experiment,
            'run': run,
            'point_id': current_point_id,
            'lifespan': last_frame_tracked - first_frame
        }, ignore_index=True)

    # Calculate the average lifespan for each run
    avg_lifespan_df = result_df.groupby(['experiment', 'run']).agg({'lifespan': 'mean'}).reset_index()
    avg_lifespan_df.rename(columns={'lifespan': 'average_lifespan'}, inplace=True)

    return result_df, avg_lifespan_df


def calculate_lifespans_gpu(df, frame_threshold):
    # Filter out points with ClusterLabel of -1 using cuDF
    df_filtered = df[df['ClusterLabel'] != -1]

    # Sort the cuDF dataframe by 'experiment', 'run', 'point_id', and 'frame'
    df_sorted = df_filtered.sort_values(by=['experiment', 'run', 'point_id', 'frame'])

    # Initialize arrays to store results
    experiment_array = []
    run_array = []
    point_id_array = []
    lifespan_array = []

    current_point_id = None
    first_frame = None
    last_frame_tracked = None

    # Iterate over the rows of the cuDF DataFrame
    for idx in range(len(df_sorted)):
        row = df_sorted.iloc[idx]

        if current_point_id is None or row['point_id'] != current_point_id:
            # Start tracking a new point
            current_point_id = row['point_id']
            first_frame = row['frame']
            last_frame_tracked = row['frame']
        elif row['frame'] - last_frame_tracked <= frame_threshold:
            # Update last_frame_tracked if the frame is within the threshold
            last_frame_tracked = row['frame']
        else:
            # Record the previous point's lifespan and reset tracking for a new point
            experiment_array.append(row['experiment'])
            run_array.append(row['run'])
            point_id_array.append(current_point_id)
            lifespan_array.append(last_frame_tracked - first_frame)

            current_point_id = row['point_id']
            first_frame = row['frame']
            last_frame_tracked = row['frame']

    # Record the final point's lifespan
    experiment_array.append(row['experiment'])
    run_array.append(row['run'])
    point_id_array.append(current_point_id)
    lifespan_array.append(last_frame_tracked - first_frame)

    # Create a new cuDF DataFrame from the arrays
    result_df = pd.DataFrame({
        'experiment': experiment_array,
        'run': run_array,
        'point_id': point_id_array,
        'lifespan': lifespan_array
    })

    # Calculate the average lifespan for each run using cuDF
    avg_lifespan_df = result_df.groupby(['experiment', 'run']).agg({'lifespan': 'mean'}).reset_index()

    return result_df, avg_lifespan_df


if __name__ == "__main__":
    # Get input and output file paths from command line arguments
    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]

    # Call the function to process the files
    data = import_files(input_files)
    mine_result = mine_loop(data)

    close_frame_threshold = 5  # Adjust this threshold as needed
    result_df, avg_lifespan_df = calculate_lifespans_gpu(data, close_frame_threshold)

    combined_df = pd.merge(mine_result, avg_lifespan_df, on=['experiment', 'run'], how='outer')
    print(combined_df)
    combined_df.to_csv(output_file, index=False)
