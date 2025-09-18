folder = r'E:\loss\Utils\all wells'  # folder where you extracted the CSV files
# Define the path to save the model
model_save_path = folder
# Save generated figures into a folder
output_dir = model_save_path

import previs
import sys
import DataProccessing_Prepare as dp
import random
import numpy as np
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from previs import preprocessing
from typing import Iterator
from collections import Counter, defaultdict
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

import os
def load_data(file_names, folder):
    dataframes = {}
    for fname in file_names:
        file_path = os.path.join(folder, fname)
        df = pd.read_csv(file_path)
        df_name = os.path.splitext(fname)[0].replace('-', '_').lower()
        dataframes[f'{df_name}_loss'] = df
    return dataframes


file_names = [
    'JR-07.csv',
    'SPH-09.csv'
]

"""
file_names = [
    'JR-07.csv',
    'JR-09.csv',
    'SPH-06.csv',
    'SPH-08.csv',
    'SPH-09.csv'
]
"""

dataframes = load_data(file_names, folder)
# display the keys of the dictionary to see the dataframe names
print(dataframes.keys())
print(dataframes['jr_07_loss'])




"""# **Data preprocessing**

**0. Drop unwanted columns**
"""

columns_to_drop = [
    'DEPTBITM',
    'BLKPOS',
    'SPM1',
    'SPM2',
    'SPM3',
    'DEXPO',
    'BitSize',
    'Failure Value',
    'MFIA'
]

for df_name, df in dataframes.items():
    df = df.drop(columns=columns_to_drop, errors='ignore')     # Drop the specified columns first
    # Check if 'MDOA' and 'MDIA' exist before calculating 'diff_MD'
    if 'MDOA' in df.columns and 'MDIA' in df.columns:
        df['diff_MD'] = df['MDOA'] - df['MDIA']
        df = df.drop(columns=['MDOA','MDIA'], errors='ignore')
    else:
        print(f"Warning: 'MDOA' or 'MDIA' not found in {df_name} after initial drop. Skipping 'diff_MD' calculation and drop.")

    # Update the dataframe in the dictionary
    dataframes[df_name] = df


# Optional: Display the columns of one dataframe to verify
print("\nColumns in jr_07_loss after dropping:")
print(dataframes['jr_07_loss'].columns)

"""**1. handle missing values**


"""

for df_name, df in dataframes.items():
    print(f"Analyzing dataframe: {df_name}")
    # Count null values in all columns
    print("\nNull values per column:")
    print(df.isnull().sum())
    print("-" * 30) # Separator for clarity between dataframes

# Show non numeric columns
non_numeric_columns = {}
for df_name, df in dataframes.items():
    non_numeric_cols_in_df = [col for col, dtype in df.dtypes.items() if not np.issubdtype(dtype, np.number)]
    non_numeric_columns[df_name] = non_numeric_cols_in_df

print("\nSummary of non-numeric columns across dataframes:")
for df_name, cols in non_numeric_columns.items():
    print(f"{df_name}: {cols}")

# show unique labels in Category & Formation columns
unique_category_labels = set()
unique_formation_labels = set()

for df_name, df in dataframes.items():
    if 'Category' in df.columns:
        unique_category_labels.update(df['Category'].unique())
    if 'Formation' in df.columns:
        unique_formation_labels.update(df['Formation'].unique())

print("Unique labels collected from 'Category' column across all dataframes:")
print(unique_category_labels)

print("\nUnique labels collected from 'Formation' column across all dataframes:")
print(unique_formation_labels)

# Remove variables no longer needed
del unique_category_labels
del unique_formation_labels

"""**2. remove during failre rows from Category column**"""

# remove during failure rows
filtered_dataframes = {}

for df_name, df in dataframes.items():
    filtered_dataframes[df_name] = dp.process_during_failure_events(df, 'During Failure')

# check labels in Category column
for df_name, df in filtered_dataframes.items():
    print(f"Value counts for 'Category' in {df_name}:")
    if 'Category' in df.columns:
        print(df['Category'].value_counts())
    else:
        print("'Category' column not found.")
    print("-" * 30)

"""**3. Encoding Non-numeric Columns**




"""

class MultiColumnCustomLabelEncoder:
    def __init__(self):
        self.inv_maps = {}

    def fit_transform(self, df, column_name, label_map, inplace=False):
        """
        Encode specified column in the dataframe using the provided label_map.

        Args:
            df (pd.DataFrame): Input dataframe.
            column_name (str): The column to encode.
            label_map (dict): Mapping from lowercase labels to integers.
            inplace (bool): If True, replace original column; else return encoded series.

        Returns:
            pd.Series or pd.DataFrame: Encoded series or modified dataframe.
        """
        # Store inverse mapping for decoding later
        self.inv_maps[column_name.lower()] = {v: k for k, v in label_map.items()}

        # Uniform cleaning (lowercase + strip) applied to all columns
        series = df[column_name].astype(str).str.lower().str.strip()

        encoded = series.map(label_map)

        if inplace:
            df[column_name] = encoded
            return df
        else:
            return encoded

    def transform_series(self, series, column_name, label_map):
        """
        Transform a pandas Series using given label_map.

        Args:
            series (pd.Series): Series to encode.
            column_name (str): Name of the column (used for inverse mapping storage).
            label_map (dict): Mapping from lowercase labels to integers.

        Returns:
            pd.Series: Encoded series.
        """
        series_clean = series.astype(str).str.lower().str.strip()
        return series_clean.map(label_map)

    def inverse_transform_series(self, series, column_name):
        """
        Reverse the encoding back to original labels.

        Args:
            series (pd.Series): Encoded series with integers.
            column_name (str): Column name to find inverse mapping.

        Returns:
            pd.Series: Decoded original labels.
        """
        if column_name.lower() not in self.inv_maps:
            raise ValueError(f"No inverse mapping found for column '{column_name}'")
        inv_map = self.inv_maps[column_name.lower()]
        return series.map(inv_map)

# Encode Category and Formation:
category_map = {
    'normal': 0,   # Changed to 0
    'seepage': 1,  # Changed to 1
    'partial': 2,  # Changed to 2
    'severe': 3    # Changed to 3
}

formation_map = {
    'sarvak': 1,
    'gachsaran': 2,
    'gurpi': 3,
    'gadvan': 4,
    'asmari': 5,
    'laffan': 6,
    'mishan': 7,
    'ilam': 8,
    'kazhdumi': 9,
    'tarbur': 10,
    'fahliyan': 11,
    'dariyan': 12,
    'aghajari': 13,
    'pabdeh': 14
}

encoder = MultiColumnCustomLabelEncoder()

for df_name, df in filtered_dataframes.items():
    df['Category_encoded'] = encoder.fit_transform(df, 'Category', category_map)
    if 'Formation' in df.columns:
        df['Formation_encoded'] = encoder.fit_transform(df, 'Formation', formation_map)

# Remove variables no longer needed after encoding
del category_map
del formation_map

# Check if encoding is done properly
for df_name, df in filtered_dataframes.items():
    print(f"Checking encoding for dataframe: {df_name}")
    # Display relevant columns for the first few rows
    cols_to_display = []
    if 'Category' in df.columns and 'Category_encoded' in df.columns:
        cols_to_display.extend(['Category', 'Category_encoded'])
    if 'Formation' in df.columns and 'Formation_encoded' in df.columns:
        cols_to_display.extend(['Formation', 'Formation_encoded'])

    if cols_to_display:
        print(df[cols_to_display].head())
    else:
        print("Relevant encoded columns not found in this dataframe.")
    print("-" * 30)

"""**4. Earrly Prediction: Label Shift**"""

#region 2 ##################################  Earrly prediction (label shifting) ####################################
shifted_dataframes = {}

for df_name, df in filtered_dataframes.items():
    print(f"Applying label shifting to {df_name}...")
    initial_rows = len(df)
    shifted_label, shifted_time = dp.shifted_labels(df.copy()) # Use a copy to avoid modifying the original df in place
    mask = ~np.isnat(shifted_time)

    # Apply the mask to the original dataframe and the shifted results
    df_shifted = df[mask].copy()
    shifted_label = shifted_label[mask]

    df_shifted['Category_shifted'] = shifted_label

    print(f'Dataframe {df_name}: Initial Rows Count: {initial_rows}, Rows Count after shifting and masking: {len(df_shifted)}')
    shifted_dataframes[df_name] = df_shifted

# Remove variables no longer needed after shifting
del filtered_dataframes # This was already done in a separate cell, but including here for completeness of the section
del shifted_label
del shifted_time
del mask
del df_shifted

#endregion #########################################  END Earrly prediction #########################################

# show the five top rows of each data frame
for df_name, df in shifted_dataframes.items():
    print(f"\nFirst 5 rows of {df_name}:")
    print(df.head())

"""**5. Segment Dataframes**"""

#region 3#########################################  Segment #########################################
# Create a dictionary to store segmented dataframes for each original dataframe
all_segmented_dataframes = {}

for df_name, df in shifted_dataframes.items():
    print(f"Segmenting dataframe: {df_name}--------------------------------------------------------------")
    segments = dp.segment_dataframes(df)
    all_segmented_dataframes[df_name] = segments
    print(f"Number of segments for {df_name}: {len(segments)}")

    # Print total rows after segmentation for each dataframe
    total_rows_in_segments = 0
    for segment in segments:
        total_rows_in_segments += len(segment)
    print(f'Total Row after segmentation for {df_name}: {total_rows_in_segments}-----------------------------------------------------')
#endregion #########################################  END Segment #########################################

"""**6. Batching**"""

#region 4#########################################  Batching #########################################
all_batches = {}

for df_name, segments in all_segmented_dataframes.items():
    print(f"Batching dataframe: {df_name}--------------------------------------------------------------")
    batches = dp.combine_batches_fixed_count(segments, 'normal', 100)
    all_batches[df_name] = batches
    print(f'Batches Count for {df_name}: {len(batches)}')
    c = 0
    for batch in batches:
        c += len(batch)
    print(f'Total Row after batching for {df_name}: {c}-----------------------------------------------------')

# Remove variables no longer needed after batching
del shifted_dataframes
del segments
del total_rows_in_segments
del all_segmented_dataframes
del batches
del c
#endregion #########################################  END Batching #########################################

"""**7. Spilit (Train, Test, Validation)**"""

#region ######################################### Spilit (Train, Test, Validation)##############################################################
# 1. Add 'Well Name' to each batch and flatten into a list
all_batches_combined = []
for well_name, batches in all_batches.items():
    well_label = well_name.replace('_loss', '').upper()
    for batch in batches:
        batch = batch.assign(**{'Well Name': well_label})
        all_batches_combined.append(batch)

# 2. Extract batch labels for stratification
batch_labels = []
for batch in all_batches_combined:
    # Get unique Category_shifted values in the batch
    unique_labels = batch['Category_shifted'].unique()
    if len(unique_labels) == 1:
        batch_label = unique_labels[0]
    else:
        # Optionally decide how to handle mixed-label batches
        batch_label = unique_labels[0]  # or use majority label
    batch_labels.append(batch_label)

batch_labels = pd.Series(batch_labels)

# 3. Map string labels to integer classes for sklearn
label_str_to_int = {label: idx for idx, label in enumerate(batch_labels.unique())}
batch_labels_int = batch_labels.map(label_str_to_int)

# 4. Check count per class
print("Batch label counts (int):\n", batch_labels_int.value_counts())

# 5. Now use dp.split_batches on the list of filtered batches
splited_batches_filtered = dp.split_batches(all_batches_combined, normal_label='normal')

train_batches = splited_batches_filtered['train']
validation_batches = splited_batches_filtered['validation']
test_batches = splited_batches_filtered['test']

#endregion ######################################### Spilit (Train, Test, Validation)##############################################################

"""**8. Scaling**"""

#region ######################################### Scaling #########################################
# List the specific columns you want to scale
# Exclude the encoded columns from the list of columns to scale
columns_to_exclude_from_scaling = ['Category_encoded', 'Formation_encoded', 'Category_shifted', 'Well Name', 'TIME', 'delta_t']
columns_to_scale = [col for col in train_batches[1].columns if col not in columns_to_exclude_from_scaling]

import  previs.preprocessing as preprocessing
# Sort batches within each split by 'Well Name' and then 'TIME' before scaling
sorted_train_batches = sorted(train_batches, key=lambda x: (x['Well Name'].iloc[0], x['TIME'].iloc[0]))
sorted_validation_batches = sorted(validation_batches, key=lambda x: (x['Well Name'].iloc[0], x['TIME'].iloc[0]))
sorted_test_batches = sorted(test_batches, key=lambda x: (x['Well Name'].iloc[0], x['TIME'].iloc[0]))

# Separate columns to be scaled from columns to keep
train_cols_to_scale = pd.concat([batch[columns_to_scale] for batch in sorted_train_batches], ignore_index=True)
train_cols_to_keep = pd.concat([batch[columns_to_exclude_from_scaling] for batch in sorted_train_batches], ignore_index=True)

validation_cols_to_scale = pd.concat([batch[columns_to_scale] for batch in sorted_validation_batches], ignore_index=True)
validation_cols_to_keep = pd.concat([batch[columns_to_exclude_from_scaling] for batch in sorted_validation_batches], ignore_index=True)

test_cols_to_scale = pd.concat([batch[columns_to_scale] for batch in sorted_test_batches], ignore_index=True)
test_cols_to_keep = pd.concat([batch[columns_to_exclude_from_scaling] for batch in sorted_test_batches], ignore_index=True)

# Fit and transform only the columns to be scaled
scaled_train_data, scaler = dp.fit_transform_batches([train_cols_to_scale], preprocessing.Normalizer.MIN_MAX)
combined_data_train_scaled = pd.concat(scaled_train_data, ignore_index=True)
combined_data_train_scaled = pd.concat([combined_data_train_scaled, train_cols_to_keep], axis=1)

# Transform validation and test data
scaled_validation_data = scaler.transform(validation_cols_to_scale)
combined_data_validation_scaled = pd.DataFrame(scaled_validation_data, columns=columns_to_scale)
combined_data_validation_scaled = pd.concat([combined_data_validation_scaled, validation_cols_to_keep], axis=1)

scaled_test_data = scaler.transform(test_cols_to_scale)
combined_data_test_scaled = pd.DataFrame(scaled_test_data, columns=columns_to_scale)
combined_data_test_scaled = pd.concat([combined_data_test_scaled, test_cols_to_keep], axis=1)

print("Scaled training data head:")
print(combined_data_train_scaled.head())
print("\nScaled validation data head:")
print(combined_data_validation_scaled.head())
print("\nScaled test data head:")
print(combined_data_test_scaled.head())
#endregion #########################################  END Scaling #########################################

"""**Encode Well name column**"""

# Encode the 'Well Name' column in the combined dataframes
# First, get the unique well names from the combined training data to create the mapping
well_names = combined_data_train_scaled['Well Name'].unique()
well_name_map = {name.lower(): i for i, name in enumerate(well_names)}

# Use the existing encoder object to fit and transform the 'Well Name' column
# encoder = MultiColumnCustomLabelEncoder() # Re-initialize or use the existing one if it's global

combined_data_train_scaled['Well_Name_encoded'] = encoder.fit_transform(
    combined_data_train_scaled,
    'Well Name',
    well_name_map
)

combined_data_validation_scaled['Well_Name_encoded'] = encoder.transform_series(
    combined_data_validation_scaled['Well Name'],
    'Well Name',
    well_name_map
)

combined_data_test_scaled['Well_Name_encoded'] = encoder.transform_series(
    combined_data_test_scaled['Well Name'],
    'Well Name',
    well_name_map
)

print("Well Name encoding applied.")
print(combined_data_train_scaled[['Well Name', 'Well_Name_encoded']].head())
print(combined_data_validation_scaled[['Well Name', 'Well_Name_encoded']].head())
print(combined_data_test_scaled[['Well Name', 'Well_Name_encoded']].head())

# Remove variables no longer needed after encoding
del well_names
del well_name_map
del splited_batches_filtered
del all_batches

# Drop non-numeric columns except 'TIME', 'Category_encoded', 'Formation_encoded', and 'Well_Name_encoded' from the scaled dataframes
columns_to_drop_after_scaling = ['Category_shifted', 'Well Name', 'Formation', 'Category']

combined_data_train_scaled = combined_data_train_scaled.drop(columns=columns_to_drop_after_scaling, errors='ignore')
combined_data_validation_scaled = combined_data_validation_scaled.drop(columns=columns_to_drop_after_scaling, errors='ignore')
combined_data_test_scaled = combined_data_test_scaled.drop(columns=columns_to_drop_after_scaling, errors='ignore')

print("Non-numeric columns (except TIME, Category_encoded, Formation_encoded, Well_Name_encoded) dropped from scaled dataframes.")
print("\nColumns in combined_data_train_scaled after dropping:")
print(combined_data_train_scaled.columns)
print("\nColumns in combined_data_validation_scaled after dropping:")
print(combined_data_validation_scaled.columns)
print("\nColumns in combined_data_test_scaled after dropping:")
print(combined_data_test_scaled.columns)

# Remove variables no longer needed after scaling and encoding
del train_cols_to_scale
del train_cols_to_keep
del validation_cols_to_scale
del validation_cols_to_keep
del test_cols_to_scale
del test_cols_to_keep
del scaled_train_data
del scaler
del scaled_validation_data
del scaled_test_data
del sorted_train_batches
del sorted_validation_batches
del sorted_test_batches
del columns_to_scale
del columns_to_exclude_from_scaling
print("Variables no longer needed after scaling and encoding have been deleted.")

"""***Reshap for 1DCNN***"""

# Assuming all columns except 'TIME', 'Category_encoded', 'Formation_encoded', 'Well_Name_encoded', and 'delta_t' are features
columns_to_exclude_from_features = ['TIME', 'Category_encoded', 'Formation_encoded', 'Well_Name_encoded', 'delta_t']
feature_columns = [col for col in combined_data_train_scaled.columns if col not in columns_to_exclude_from_features]
target_column = 'Category_encoded'
sequence_length = 40
window_length=sequence_length
# 1. Reshape Data for 1D CNN ---
# The 1D CNN expects input data in the format (samples, time_steps, features).
def reshape_for_1DCNN(df_split, feature_columns, target_column, sequence_length):
    #  Extract features, target, and time
    features = df_split[feature_columns]
    target = df_split[target_column].values
    time = df_split['TIME'].values # Extract the time column

    # Trim data to multiple of sequence_length
    num_sample = len(features) // sequence_length
    total_samples = num_sample * sequence_length
    features_trimmed = features[:total_samples]
    target_trimmed = target[:total_samples]
    time_trimmed = time[:total_samples] # Trim the time data

    # Reshape features and targets
    n_features = features_trimmed.shape[1]
    X = features_trimmed.values.reshape(num_sample, sequence_length, n_features)

    # Reshape target to be a 1D array of class labels for sparse_categorical_crossentropy
    # Taking the last time step's label as the sequence label
    y = target_trimmed.reshape(num_sample, sequence_length)[:, -1] # Get the last target of each sequence

    # Reshape time to correspond to the target
    time_reshaped = time_trimmed.reshape(num_sample, sequence_length)[:, -1] # Get the last time of each sequence

    print(f"Sequence length: {sequence_length}")
    print(f"Number of samples: {X.shape[0]}")
    print("\nShape of X (features) after reshaping:", X.shape)
    print("Shape of y (target) after reshaping:", y.shape)
    print("Shape of time after reshaping:", time_reshaped.shape)

    return X, y, time_reshaped # Return time_reshaped

X_train, y_train, time_train = dp.seperate_X_Y_slide_label(combined_data_train_scaled, 
                                                            feature_columns=feature_columns,
                                                            window_length=window_length,
                                                            shift_minutes=0,
                                                            category_col=target_column,
                                                            time_frequency=10.0,
                                                            sort_by='TIME'
                                                            )

X_val, y_val, time_val = dp.seperate_X_Y_slide_label(combined_data_validation_scaled, 
                                                    feature_columns=feature_columns,
                                                    window_length=window_length,
                                                    shift_minutes=0,
                                                    category_col=target_column,
                                                    time_frequency=10.0,
                                                    sort_by='TIME'
                                                    )

X_test, y_test, time_test = dp.seperate_X_Y_slide_label(combined_data_test_scaled, 
                                                    feature_columns=feature_columns,
                                                    window_length=window_length,
                                                    shift_minutes=0,
                                                    category_col=target_column,
                                                    time_frequency=10.0,
                                                    sort_by='TIME'
                                                    )

# Fit scaler on train data
# X_train, y_train, time_train = reshape_for_1DCNN(combined_data_train_scaled, feature_columns, target_column, sequence_length)
# X_val, y_val, time_val = reshape_for_1DCNN(combined_data_validation_scaled, feature_columns, target_column, sequence_length)
# X_test, y_test, time_test = reshape_for_1DCNN(combined_data_test_scaled, feature_columns, target_column, sequence_length)

time_steps = X_train.shape[1]  # 60 (sequence length)
num_features = X_train.shape[2]  # 8 (features per timestep)

# Balance the training data
# Note: The normal_label needs to correspond to the encoded value for 'normal'.
# Based on category_map = {'normal': 1, 'seepage': 2, 'partial': 3, 'severe': 4}
# the normal_label should be 1.
num_classes = 4 # Number of unique categories
normal_encoded_label = 0
X_train_balanced, y_train_balanced, time_train_balanced = dp.balance_classes(
    X_train, y_train, time_train, normal_label=normal_encoded_label
)

# Update variables to use the balanced data for training
X_train = X_train_balanced
y_train = y_train_balanced
time_train = time_train_balanced

print(f"Sequence length: {sequence_length}")
print(f"Number of train samples: {X_train.shape[0]}")
print(f"Number of validation samples: {X_val.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")

print("\nShape of X_train (features) after reshaping and balancing:", X_train.shape)
print("Shape of y_train (target) after reshaping and balancing:", y_train.shape)
print("\nShape of X_val (features) after reshaping:", X_val.shape)
print("Shape of y_val (target) after reshaping:", y_val.shape)
print("\nShape of X_test (features) after reshaping:", X_test.shape)
print("Shape of y_test (target) after reshaping:", y_test.shape)

print(f"\nNumber of classes: {num_classes}")
del X_train_balanced
del y_train_balanced
del time_train_balanced

#region ######################################### Build and TRain Model ###################################
#  Build and train 1D CNN
time_steps = X_train.shape[1]  # 60 (sequence length)
num_features = X_train.shape[2]  # 8 (features per timestep)
# Use the calculated num_classes from the previous cell
# num_classes is already determined in the previous cell.

model = Sequential([
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(time_steps, num_features)),
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    Flatten(), # Add Flatten layer to reduce the output to 1D
    Dense(num_classes, activation='softmax') # Add a Dense layer for classification
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # since labels are integer encoded
    metrics=['accuracy']
)
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

# Ensure target labels are int32 as required by sparse_categorical_crossentropy
y_train = y_train.astype('int32')
y_val = y_val.astype('int32')

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Remove variables no longer needed after training
del early_stopping
#endregion ######################################### Build and TRain Model ###################################

# Save the model
model.save(os.path.join(model_save_path, "1D_CNN_Model.keras"))
print(f"Model saved successfully to {model_save_path}")

#region ######################################### Evaluation ###################################

# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f'\nTest Loss (Sparse Categorical Crossentropy): {loss}')
# print(f'Test Accuracy: {accuracy:.4f}')

# Predict for train, validation, and test sets
train_predictions = model.predict(X_train, verbose=0)
train_pred_classes = np.argmax(train_predictions, axis=-1)
val_predictions = model.predict(X_val, verbose=0)
val_pred_classes = np.argmax(val_predictions, axis=-1)
test_predictions = model.predict(X_test, verbose=0)
test_pred_classes = np.argmax(test_predictions, axis=-1)

# Flatten the arrays for metrics calculation
# Note: y_train, y_val, y_test are already 1D arrays from the reshape_for_1DCNN function's last time step selection
train_true_flat = y_train
train_pred_flat = train_pred_classes
val_true_flat = y_val
val_pred_flat = val_pred_classes
test_true_flat = y_test
test_pred_flat = test_pred_classes

#endregion ######################################### Evaluation ###################################
#region ###################################plot methods#############################################
# Normaliazed confusion matrix
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_normalized_confusion_matrix(y_true, y_pred, labels, ax, title):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    gap = 0.1  # فاصله بین خونه‌ها

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "#b3ffb3" if i == j else "#ffcccc"  # سبز قطر، صورتی بقیه
            rect = FancyBboxPatch(
                (j + gap/2, i + gap/2), 1 - gap, 1 - gap,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="black")

    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", rotation=45)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold", rotation=0)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, cm.shape[1])
    ax.set_ylim(cm.shape[0], 0)

def plot_train_val_confusion_matrices(y_train, train_pred_classes, y_val, val_pred_classes, encoder):
    # Use the category_class_names list generated previously
    category_class_names = [encoder.inv_maps['category'][i] for i in sorted(encoder.inv_maps['category'].keys())]
    labels = category_class_names

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_normalized_confusion_matrix(
        y_train, train_pred_classes, labels, axes[0], "Train Confusion Matrix (Normalized)"
    )
    plot_normalized_confusion_matrix(
        y_val, val_pred_classes, labels, axes[1], "Validation Confusion Matrix (Normalized)"
    )

    plt.tight_layout()
    plt.show()

# Count confusion matrix
def plot_count_confusion_matrix(y_true, y_pred, labels, ax, title):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    gap = 0.1

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "#b3ffb3" if i == j else "#ffcccc"
            rect = FancyBboxPatch(
                (j + gap/2, i + gap/2), 1 - gap, 1 - gap,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha="center", va="center", fontsize=12, fontweight="bold", color="black")

    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", rotation=45)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold", rotation=0)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, cm.shape[1])
    ax.set_ylim(cm.shape[0], 0)

def plot_train_val_count_confusion_matrices(y_train, train_pred_classes, y_val, val_pred_classes, encoder):
    # Use the category_class_names list generated previously
    category_class_names = [encoder.inv_maps['category'][i] for i in sorted(encoder.inv_maps['category'].keys())]
    labels = category_class_names

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_count_confusion_matrix(
        y_train, train_pred_classes, labels, axes[0], "Train Confusion Matrix (Count)"
    )
    plot_count_confusion_matrix(
        y_val, val_pred_classes, labels, axes[1], "Validation Confusion Matrix (Count)"
    )

    plt.tight_layout()
    plt.show()

# plots the true vs predicted Category values against depth for each well.

def plot_category_vs_depth_per_well(combined_df, predicted_classes, encoder, wells_to_plot=None):
    """
    Plots true vs predicted Category values against depth for each well.

    Args:
        combined_df (pd.DataFrame): DataFrame containing 'DEPTMEAS', 'Category_encoded',
                                    'Well Name', and corresponding predicted classes.
        predicted_classes (np.ndarray): Array of predicted class integers.
        encoder (MultiColumnCustomLabelEncoder): The encoder object used for decoding labels.
        wells_to_plot (list, optional): A list of well names to plot. If None, plots all wells.
    """
    # Ensure predicted_classes has the same length as the combined_df
    if len(combined_df) != len(predicted_classes):
        print("Error: Length of combined_df and predicted_classes do not match.")
        return

    # Add predicted classes to the dataframe for easier handling
    df_plot = combined_df.copy()
    df_plot['Predicted_Category_encoded'] = predicted_classes

    # Get the inverse mapping for decoding category labels
    if 'category' not in encoder.inv_maps:
         print("Error: 'category' inverse mapping not found in the encoder.")
         return
    category_inv_map = encoder.inv_maps['category']

    # Get unique well names
    all_wells = df_plot['Well Name'].unique()

    # Determine which wells to plot
    wells_to_process = wells_to_plot if wells_to_plot is not None else all_wells

    for well_name in wells_to_process:
        print(f"Generating plot for Well: {well_name}")
        # Filter data for the current well
        df_well = df_plot[df_plot['Well Name'] == well_name].copy()

        if df_well.empty:
            print(f"No data found for well: {well_name}. Skipping.")
            continue

        # Decode true and predicted labels
        df_well['True_Category_label'] = df_well['Category_encoded'].map(category_inv_map)
        df_well['Predicted_Category_label'] = df_well['Predicted_Category_encoded'].map(category_inv_map)

        # Sort by depth for better visualization
        df_well = df_well.sort_values(by='DEPTMEAS')

        plt.figure(figsize=(10, 8))

        # Plot true categories
        plt.scatter(df_well['True_Category_label'], df_well['DEPTMEAS'], color='blue', label='True', alpha=0.6, s=10)

        # Plot predicted categories
        plt.scatter(df_well['Predicted_Category_label'], df_well['DEPTMEAS'], color='red', label='Predicted', alpha=0.6, s=10)

        plt.xlabel("Category")
        plt.ylabel("Depth (DEPTMEAS)")
        plt.title(f"True vs Predicted Category vs Depth for {well_name}")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis() # Invert y-axis so depth increases downwards
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
        plt.tight_layout()
        plt.show()

# Example Usage (Assuming combined_data_test_scaled and test_pred_classes are available)
# plot_category_vs_depth_per_well(combined_data_test_scaled, test_pred_classes, encoder)
# Example Usage for specific wells
# plot_category_vs_depth_per_well(combined_data_test_scaled, test_pred_classes, encoder, wells_to_plot=['JR_07', 'SPH_09'])
#endregion ###################################plot methods#############################################


#region ###################################plot + save results#############################################
# Save accuracy and loss plot
import os
os.makedirs(output_dir, exist_ok=True)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_loss_plot.png'))
plt.close() # Close the plot to avoid displaying it again

# Save normalized confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
category_class_names = [encoder.inv_maps['category'][i] for i in sorted(encoder.inv_maps['category'].keys())]
category_class_names_val=['normal', 'seepage', 'partial']
plot_normalized_confusion_matrix(
    y_train, train_pred_classes, category_class_names, axes[0], "Train Confusion Matrix (Normalized)"
)

plot_normalized_confusion_matrix(
    y_val, val_pred_classes, category_class_names_val, axes[1], "Validation Confusion Matrix (Normalized)"
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'))
plt.close()

# Save count confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_count_confusion_matrix(
    y_train, train_pred_classes, category_class_names, axes[0], "Train Confusion Matrix (Count)"
)
plot_count_confusion_matrix(
    y_val, val_pred_classes, category_class_names_val, axes[1], "Validation Confusion Matrix (Count)"
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'count_confusion_matrix.png'))
plt.close()


# Save evaluation table as text
evaluation_results_path = os.path.join(output_dir, 'evaluation_report.txt')
with open(evaluation_results_path, 'w') as f:
    f.write("Evaluation Results:\n\n")
    #f.write(f'Test Loss (Sparse Categorical Crossentropy): {loss}\n')
    #f.write(f'Test Accuracy: {accuracy:.4f}\n\n')
    f.write("Train Classification Report:\n")
    f.write(classification_report(train_true_flat, train_pred_flat, target_names=category_class_names))
    f.write("\nValidation Classification Report:\n")
    f.write(classification_report(val_true_flat, val_pred_flat, target_names=category_class_names_val))
    #f.write("\nTest Classification Report:\n")
    #f.write(classification_report(test_true_flat, test_pred_flat, target_names=category_class_names))

print(f"Evaluation results saved to '{output_dir}' folder.")
#endregion ###################################plot + save results#############################################
# Save the data used for figures and evaluations
import os
import numpy as np
import joblib

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define file paths for saving
train_data_path = os.path.join(output_dir, 'train_evaluation_data.npz')
val_data_path = os.path.join(output_dir, 'validation_evaluation_data.npz')
test_data_path = os.path.join(output_dir, 'test_evaluation_data.npz')
encoder_path = os.path.join(output_dir, 'label_encoder.joblib')

# Save the training data
np.savez(train_data_path, X_train=X_train, y_train=y_train, time_train=time_train, train_pred_classes=train_pred_classes)
print(f"Training evaluation data saved to: {train_data_path}")

# Save the validation data
np.savez(val_data_path, X_val=X_val, y_val=y_val, time_val=time_val, val_pred_classes=val_pred_classes)
print(f"Validation evaluation data saved to: {val_data_path}")

# Save the test data
np.savez(test_data_path, X_test=X_test, y_test=y_test, time_test=time_test, test_pred_classes=test_pred_classes)
print(f"Test evaluation data saved to: {test_data_path}")


