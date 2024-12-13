import torch
from model import create_model
from utils import model_name_for_prediction, save_path, image_paths, class_names
from transform import transform
import sys
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import torch
from torchvision import transforms

def predict_single_image(model, image_path, class_names, device='cpu'):
    """
    Predict the class of a single image.
    
    Args:
        model: The trained model in .eval mode.
        image_path: Path to the image to predict.
        class_names: List of class names corresponding to indices.
        device: 'cuda' or 'cpu'.

    Returns:
        Predicted class label.
    """
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Ensure it's RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    print('step 3 done')
    # Move image to the same device as the model
    image = image.to(device)
    
    # Model inference
    model = model.to(device)
    model.eval()
    print('step 4 done')
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)  # Get index of the highest score
    
    # Map index to class label
    predicted_class = class_names[predicted_idx.item()]
    print('step 5 done')
    return predicted_class


def main(ca):
    model = model_name_for_prediction
    model = create_model(model_name=model_name_for_prediction)
    model.load_state_dict(torch.load(save_path+"/"+model_name_for_prediction, weights_only=True))
    model.eval()
    if ca<1 or ca>5:
            image_path = image_paths[1]
    else:
            image_path = image_paths[ca]
    # print('step 1')
    predicted_class = predict_single_image(model, image_path, class_names)
    print(f"Predicted Class: {predicted_class}")

if __name__=="__main__":
    ca = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(ca)