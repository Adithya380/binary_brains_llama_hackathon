import requests, base64

def predict_crop_disease(image_path):
    """
    Predicts the crop disease based on an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Description of the crop disease.
    """

    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/meta/llama-3.2-90b-instruct"
    stream = False

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # assert len(image_b64) < 180_000, \
    #   "To upload larger images, use the assets API (see docs)"

    headers = {
        "Authorization": "Bearer nvapi-'provide your key here'",
        "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Try to identify the anomalous patterns and possible disease infestation in the given crop image. <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 50,
        "stream": False,
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    response_1 = response.json()

    result = response_1['choices'][0]['message']['content']

    return result

# Example usage:
# image_path = "LlamaHack\Image farm 1.jpeg"
# disease_description = predict_crop_disease(image_path)
# print(disease_description)