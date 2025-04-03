from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from model_creation import create_dfnn
from data_processing import DataProcessor

def train_model(climate_data, soil_data, crop_data):
    # Initialize data processor
    processor = DataProcessor()
    
    # Preprocess data
    X_climate = np.array([processor.process_climate(c) for c in climate_data])
    X_soil = np.array([processor.process_soil(s) for s in soil_data])
    X_crop, y = processor.process_crop(crop_data)
    
    # Train-test split
    (X_clim_train, X_clim_test,
     X_soil_train, X_soil_test,
     X_crop_train, X_crop_test,
     y_train, y_test) = train_test_split(X_climate, X_soil, X_crop, y, test_size=0.2)

    # Create the model using imported function
    input_shapes = {'climate': 8, 'soil': 6, 'crop': 4}
    model = create_dfnn(input_shapes)  # Use the imported function
    
    # Train the model
    history = model.fit(
        [X_clim_train, X_soil_train, X_crop_train],
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=15),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Save the trained model and preprocessors
    model.save('models/crop_yield_model.keras')
    processor.save_preprocessors()
    
    return history

# Example usage:
# train_model(climate_data=<list>, soil_data=<list>, crop_data=<DataFrame>)
