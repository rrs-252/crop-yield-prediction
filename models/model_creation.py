from tensorflow.keras import layers, models

def build_model():
    """Define Deep Fusion Neural Network (DFNN) architecture."""
    # Soil branch (3 features: OC, pH, Clay)
    soil_input = layers.Input(shape=(3,), name="soil")
    x = layers.Dense(128, activation='swish')(soil_input)
    x = layers.BatchNormalization()(x)
    
    # Climate branch (3 features: temp, rain, humidity)
    climate_input = layers.Input(shape=(3,), name="climate")
    y = layers.Dense(64, activation='swish')(climate_input)
    y = layers.Dropout(0.3)(y)
    
    # Crop embedding (4 crops: wheat, rice, maize, soy)
    crop_input = layers.Input(shape=(1,), name="crop")
    z = layers.Embedding(input_dim=4, output_dim=8)(crop_input)
    z = layers.Flatten()(z)
    
    # Year input (normalized years: 2010-2020)
    year_input = layers.Input(shape=(1,), name="year")
    
    # Feature fusion
    merged = layers.Concatenate()([x, y, z, year_input])
    merged = layers.Dense(256, activation='swish')(merged)
    merged = layers.Dropout(0.5)(merged)
    
    # Output layer: Yield prediction (kg/ha)
    output = layers.Dense(1, activation='linear')(merged)
    
    return models.Model(
        inputs=[soil_input, climate_input, crop_input, year_input],
        outputs=output
    )

