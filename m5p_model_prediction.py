import pandas as pd
import numpy as np
import pickle

# Step 1: Function to load the model, encoder, and scaler from a .pkl file
def load_model_and_preprocessing(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        model_data = pickle.load(file)
    loaded_model = model_data['model']
    loaded_encoder = model_data['encoder']
    loaded_scaler = model_data['scaler']
    return loaded_model, loaded_encoder, loaded_scaler


# Function to preprocess the input data and generate predictions for both Promo and Base cases
def preprocess_and_predict(filtered_df, model, encoder, scaler):
    """
    This function takes the filtered DataFrame and processes it for prediction using the loaded model.
    It performs encoding and scaling before generating predictions for both Promo and Base cases.
    Includes additional debug steps to ensure data integrity.
    """
    # Define the columns for categorical and numerical features with their original case
    categorical_features = ['Item', 'Week_Type', 'Promo_Type', 'Merch', 'Customer']
    numerical_features = ['Base_Price', 'Price']

    # Create a mapping of lowercase column names to their original case
    column_map = {
        'item': 'Item',
        'week_type': 'Week_Type',
        'promo_type': 'Promo_Type',
        'merch': 'Merch',
        'customer': 'Customer',
        'base_price': 'Base_Price',
        'price': 'Price'
    }

    # Convert all column names in the DataFrame to lowercase for case-insensitive matching
    filtered_df.columns = [col.lower() for col in filtered_df.columns]
    
    # Map the columns back to their original case
    filtered_df.rename(columns=column_map, inplace=True)

    # Check if required columns are present in the DataFrame
    missing_columns = [col for col in categorical_features + numerical_features if col not in filtered_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")

    # Ensure the data types are consistent
    dtype_dict = {
        'Base_Price': 'float',
        'Price': 'float',
        'Customer': 'str',
        'Item': 'str',
        'Week_Type': 'str',
        'Promo_Type': 'str',
        'Merch': 'str'
    }
    filtered_df = filtered_df.astype(dtype_dict)

    if filtered_df.empty:
        print("Filtered DataFrame is empty. No predictions to make.")
        return filtered_df


    # --- Predicting Promo Units ---
    # Prepare input features for promo case
    X = filtered_df[categorical_features + numerical_features]

    # Encode categorical variables using the loaded encoder
    X_encoded = encoder.transform(X[categorical_features])

    # Scale numerical features using the loaded scaler
    X_scaled = scaler.transform(X[numerical_features])



    # Combine processed numerical and categorical data
    X_processed = np.hstack((X_scaled, X_encoded.toarray()))


    # Generate predictions using the loaded model
    predictions = model.predict(X_processed)



    # Add predictions to the original filtered DataFrame
    filtered_df['Predicted_Units'] = predictions

    # --- Predicting Base Units ---
    # Create a copy of the filtered DataFrame to modify for base units calculation
    base_df = filtered_df.copy()

    # Set base conditions: Week_Type = "Base", Promo_Type = "No_Promo", Merch = "No_Promo", Price = Base_Price
    base_df['Week_Type'] = 'Base'
    base_df['Promo_Type'] = 'No_Promo'
    base_df['Merch'] = 'No_Promo'
    base_df['Price'] = base_df['Base_Price']  # Set Price equal to Base_Price

    # Prepare input features for base case
    X_base = base_df[categorical_features + numerical_features]

    # Encode categorical variables for base case using the loaded encoder
    X_encoded_base = encoder.transform(X_base[categorical_features])

    # Scale numerical features for base case using the loaded scaler
    X_scaled_base = scaler.transform(X_base[numerical_features])


    # Combine processed numerical and categorical data for base case
    X_processed_base = np.hstack((X_scaled_base, X_encoded_base.toarray()))

    # Generate predictions for base case using the loaded model
    base_predictions = model.predict(X_processed_base)



    # Add base predictions to the original DataFrame as 'Predicted_Base_Units'
    filtered_df['Predicted_Base_Units'] = base_predictions

    return filtered_df

# Main function to generate predictions from filtered DataFrame
def predict_units_from_filtered_df(filtered_df, model_file):
    """
    This function loads the model, encoder, and scaler, and uses the filtered DataFrame to generate predictions.
    Ensures that all column names and values are handled in a case-insensitive manner.
    """

    # Load the predictive model and preprocessing objects
    loaded_model, loaded_encoder, loaded_scaler = load_model_and_preprocessing(model_file)

    # Preprocess the filtered data and generate predictions for both promo and base cases
    predicted_df = preprocess_and_predict(filtered_df, loaded_model, loaded_encoder, loaded_scaler)

    return predicted_df

# Save function to export predictions to a CSV
def save_predictions_to_csv(predicted_df, filename="predictions_with_base_units.csv"):
    predicted_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
