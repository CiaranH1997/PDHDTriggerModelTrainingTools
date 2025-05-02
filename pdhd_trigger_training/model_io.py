import tensorflow as tf
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def save_model(model, export_dir):
    """
    Saves a Keras model in the TensorFlow SavedModel format.

    Parameters:
    - model: tf.keras.Model instance to be saved.
    - export_dir: Directory path where the model will be saved.
    """
    os.makedirs(export_dir, exist_ok=True)
    model.save(export_dir, save_format='tf')

def load_model(export_dir):
    """
    Loads a Keras model from the TensorFlow SavedModel format.

    Parameters:
    - export_dir: Directory path from where the model will be loaded.

    Returns:
    - A tf.keras.Model instance.
    """
    return tf.keras.models.load_model(export_dir)

def save_frozen_graph(model, export_dir, model_name="frozen_graph.pb"):
    """
    Converts and saves a Keras model as a frozen graph (.pb file).

    Parameters:
    - model: tf.keras.Model instance to be converted.
    - export_dir: Directory path where the frozen graph will be saved.
    - model_name: Name of the output .pb file.
    """
    os.makedirs(export_dir, exist_ok=True)

    # Create a concrete function from the Keras model
    full_model = tf.function(lambda x: model(x))
    input_shape = model.inputs[0].shape
    input_dtype = model.inputs[0].dtype
    concrete_func = full_model.get_concrete_function(tf.TensorSpec(input_shape, input_dtype))

    # Convert variables to constants
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_graph_def = frozen_func.graph.as_graph_def()

    # Save the frozen graph
    tf.io.write_graph(frozen_graph_def, export_dir, model_name, as_text=False)

def load_frozen_graph(pb_file_path):
    """
    Loads a frozen graph from a .pb file.

    Parameters:
    - pb_file_path: File path to the .pb file.

    Returns:
    - A TensorFlow Graph object.
    """
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph definition into a new graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph
