import json
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os

# Initialize the vision-language model
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

def generate_caption(image_path):
    """
    Generate a caption for an image.
    
    Args:
    - image_path (str): Path to the image
    
    Returns:
    - str: Generated caption
    """
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(image, return_tensors="pt")
        
        # Generate caption
        output_ids = model.generate(
            **inputs, 
            max_length=50,
            num_beams=4,
            early_stopping=True
        )[0]
        
        # Decode the caption
        caption = processor.decode(output_ids, skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error processing image: {e}"

def process_images_and_generate_captions(input_json_path, image_directory):
    """
    Process images and generate captions based on image IDs.
    
    Args:
    - input_json_path (str): Path to the input JSON file with image IDs and questions
    - image_directory (str): Directory containing the images
    """
    # Load input JSON
    with open(input_json_path, "r") as file:
        data = json.load(file)
    
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a dictionary.")

    output_data = {}

    for key, entry in data.items():
        try:
            img_id = entry["imageId"]
            image_path = f"{image_directory}/{img_id}.jpg"  # Adjust extension if needed

            # Generate caption
            caption = generate_caption(image_path)

            # Append results with imageId and generated caption (description)
            output_data[key] = {
                "imageId": img_id,
                "description": caption
            }
        except KeyError as e:
            print(f"Skipping entry due to missing key: {e}")
        except Exception as e:
            print(f"Error processing entry: {e}")

    # Define the output file path (saving to root directory)
    output_json_path = "captions.json"

    # Save output JSON to the root directory
    with open(output_json_path, "w") as file:
        json.dump(output_data, file, indent=2)

    print(f"Captions saved to {output_json_path}")

# Example usage
if __name__ == "__main__":
    input_json_path = "questions.json"  # Path to your input JSON file
    image_directory = "./images"         # Directory containing images

    process_images_and_generate_captions(input_json_path, image_directory)
