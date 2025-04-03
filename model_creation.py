from keras import layers, models

def create_dfnn(input_shapes):
    climate_input = layers.Input(shape=(input_shapes['climate'],), name="climate")
    x = layers.Dense(64, activation='swish')(climate_input)
    
    soil_input = layers.Input(shape=(input_shapes['soil'],), name="soil")
    y = layers.Dense(32, activation='swish')(soil_input)
    
    crop_input = layers.Input(shape=(input_shapes['crop'],), name="crop_geo")
    z = layers.Embedding(input_dim=10000, output_dim=8)(crop_input)
    z = layers.Flatten()(z)
    
    merged = layers.Concatenate()([x, y, z])
    merged = layers.Dense(128, activation='swish')(merged)
    merged = layers.Dropout(0.3)(merged)
    output = layers.Dense(1, activation='linear')(merged)
    
    model = models.Model(
        inputs=[climate_input, soil_input, crop_input],
        outputs=output
    )
    
    model.compile(
        optimizer='adamw',
        loss='huber',
        metrics=['mae']
    )
    return model
