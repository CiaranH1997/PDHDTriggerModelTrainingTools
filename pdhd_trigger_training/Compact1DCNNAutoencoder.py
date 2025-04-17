from tensorflow import keras
from tensorflow.keras import layers

class Compact1DCNNAutoencoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        input_layer = keras.Input(shape=self.input_shape)
        # Encoder
        x = layers.Conv1D(20, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        latent = layers.Conv1D(10, kernel_size=3, activation='relu', padding='same')(x)
        # Decoder
        x = layers.Conv1D(10, kernel_size=3, activation='relu', padding='same')(latent)
        x = layers.UpSampling1D(size=2)(x)
        output_layer = layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(x)
        
        autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def get_model(self):
        return self.model
    
    def save_as_savedmodel(self, export_dir):
        """
        Saves the model in TensorFlow's SavedModel format.
        """
        self.model.save(export_dir)
        print(f"Model saved in SavedModel format at: {export_dir}")

    def export_frozen_graph(self, export_dir):
        """
        Converts the SavedModel to a frozen graph and saves it as a .pb file.
        """
        # Load the SavedModel
        loaded = tf.saved_model.load(export_dir)
        infer = loaded.signatures['serving_default']

        # Convert variables to constants
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        frozen_func = convert_variables_to_constants_v2(infer)
        frozen_graph_def = frozen_func.graph.as_graph_def()

        # Save the frozen graph
        frozen_graph_dir = os.path.join(export_dir, 'frozen_graph')
        os.makedirs(frozen_graph_dir, exist_ok=True)
        tf.io.write_graph(graph_or_graph_def=frozen_graph_def,
                          logdir=frozen_graph_dir,
                          name='frozen_graph.pb',
                          as_text=False)
        print(f"Frozen graph saved at: {os.path.join(frozen_graph_dir, 'frozen_graph.pb')}")
