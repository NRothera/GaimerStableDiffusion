import base64
import io
import os
from PIL import Image
from rembg import remove, new_session
import numpy as np

def remove_background(image: Image.Image) -> Image.Image:
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the isnet-anime model
    model_path = os.path.join(project_root, "models/rembg/isnet-anime.onnx")
    
    # Initialize a new session with the model
    session = new_session(model_path)

    # Convert the PIL Image to a NumPy array
    image_np = np.array(image)
    
    # Remove the background using rembg with the specified session
    result_np = remove(image_np, session=session)
    
    # Convert the result back to a PIL Image
    result_image = Image.fromarray(result_np)
    
    return result_image