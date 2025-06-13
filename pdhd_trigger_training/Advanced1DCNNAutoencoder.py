class Advanced1DCNNAutoencoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        input_layer = keras.Input(shape=self.input_shape)
        # Encoder
        x = layers.Conv1D(40, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        x = layers.Conv1D(10, kernel_size=3, activation='relu', padding='same')(x)
        latent = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        # Decoder
        x = layers.Conv1D(10, kernel_size=3, activation='relu', padding='same')(latent)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(40, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(size=2)(x)
        output_layer = layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(x)
        
        autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def get_model(self):
        return self.model