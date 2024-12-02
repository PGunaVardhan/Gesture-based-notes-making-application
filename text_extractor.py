import ollama
from io import BytesIO
import base64
from PIL import Image
import numpy as np

# Define the LLAVA model name and prompt
MODEL_NAME = "llava"
EXTRACTION_PROMPT = "Extract individual alphabets from this image. Give only alphabets. Nothing else."

def extract_text_from_image(image):
    """
    Extracts text from an image using the LLAVA model from Ollama.

    Args:
        image (PIL.Image, numpy.ndarray, or file-like object): The image file from which to extract text.

    Returns:
        str: Extracted text or an error message if extraction fails.
    """
    try:
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert image to base64 string for sending to Ollama
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Send the base64-encoded image and prompt to LLAVA for text extraction
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': EXTRACTION_PROMPT,
                    'images': [image_base64]
                }
            ]
        )
        # Return the content of the response
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"
