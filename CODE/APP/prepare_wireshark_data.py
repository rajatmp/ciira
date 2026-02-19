import pandas as pd
import numpy as np

def prepare_wireshark_csv(input_csv_path, output_csv_path):
    """
    Reads a CSV exported from Wireshark, renames columns, fills missing values,
    and saves it in a format suitable for the IIoT Guard application.

    Args:
        input_csv_path (str): Path to the CSV file exported from Wireshark.
        output_csv_path (str): Path where the prepared CSV will be saved.
    """
    try:
        # Read the Wireshark exported CSV
        # Use 'low_memory=False' to prevent dtype warnings with mixed types
        # Removed infer_datetime_format as it's deprecated and not relevant to this issue.
        df = pd.read_csv(input_csv_path, low_memory=False)
        print(f"Successfully loaded {input_csv_path}. Original columns: {df.columns.tolist()}")
        print(f"Original DataFrame head:\n{df.head()}")

        # Normalize column names from Wireshark export (e.g., remove '.', convert to lowercase, replace spaces with underscores)
        # This step converts 'Source Port' to 'source_port', 'TCP Length' to 'tcp_length', etc.
        df.columns = df.columns.str.strip().str.lower().str.replace('.', '_', regex=False).str.replace(' ', '_', regex=False)
        print(f"Normalized DataFrame columns (raw representation): {repr(df.columns.tolist())}") # Use repr to see exact strings
        print(f"Normalized DataFrame columns: {df.columns.tolist()}")


        # Define the mapping from the *normalized Wireshark column names* to the *model's feature names*
        # These keys must now match what `df.columns` will be after normalization.
        column_mapping = {
            'source_port': 'src_port',
            'destination_port': 'dst_port',
            'tcp_length': 'src_bytes', # Wireshark's 'TCP Length' maps to model's 'src_bytes'
            'http_content_length': 'dst_bytes',
            'tcp_sequence': 'duration', # Wireshark's 'TCP Sequence' maps to model's 'duration'
            'tcp_ack_number': 'src_pkts', # Wireshark's 'TCP ACK Number' maps to model's 'src_pkts'
            'tcp_raw_ack_number': 'dst_pkts' # Wireshark's 'TCP Raw ACK Number' maps to model's 'dst_pkts'
        }

        # Create a new DataFrame with only the target feature columns, initialized with 0s
        prepared_df = pd.DataFrame(0, index=df.index, columns=list(column_mapping.values()))

        # Populate prepared_df with data from df, handling type conversion and missing columns
        for ws_col_normalized, model_feature in column_mapping.items():
            if ws_col_normalized in df.columns:
                print(f"\n--- Processing column: '{ws_col_normalized}' (maps to '{model_feature}') ---")
                # Convert column to string type first, then strip whitespace
                cleaned_data = df[ws_col_normalized].astype(str).str.strip()
                print(f"Sample cleaned string data from '{ws_col_normalized}': {cleaned_data.head().tolist()}")

                # Convert to numeric, coercing errors (non-numeric values) to NaN
                numeric_data = pd.to_numeric(cleaned_data, errors='coerce')
                print(f"Sample numeric data (before fillna) for '{model_feature}': {numeric_data.head().tolist()}")

                # Fill NaN values with 0
                prepared_df[model_feature] = numeric_data.fillna(0)
                print(f"Sample final data (after fillna) for '{model_feature}': {prepared_df[model_feature].head().tolist()}")
            else:
                print(f"\n--- Column '{ws_col_normalized}' not found in Wireshark export. '{model_feature}' will be 0. ---")
                # If the Wireshark column isn't present, the model_feature column in prepared_df remains 0 (from initialization)

        # Ensure all required_features are in the final DataFrame and in the correct order
        required_features = list(column_mapping.values())
        prepared_df = prepared_df[required_features] # Reorder if necessary

        # Save the prepared DataFrame to a new CSV file
        prepared_df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Successfully prepared data and saved to: {output_csv_path}")
        print(f"Final Prepared CSV columns: {prepared_df.columns.tolist()}")
        print(f"First 5 rows of final prepared data:\n{prepared_df.head()}")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

if __name__ == "__main__":
    # --- Configuration ---
    WIRESHARK_EXPORT_CSV = 'wireshark_capture.csv' # Name of the CSV you exported from Wireshark
    PREPARED_CSV_FOR_APP = 'prepared_traffic_data.csv' # Name for the output CSV

    # Run the preparation function
    prepare_wireshark_csv(WIRESHARK_EXPORT_CSV, PREPARED_CSV_FOR_APP)
