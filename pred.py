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

def prediction(ca=None, model_name_for_prediction=model_name_for_prediction, img_path= None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_name_for_prediction
    model = create_model(model_name=model_name_for_prediction)
    # model.load_state_dict(torch.load(save_path+"/"+model_name_for_prediction+'.pt', weights_only=True))
    model.load_state_dict(torch.load(os.path.join(save_path, model_name_for_prediction + '.pt'), map_location=device))
    model.eval()

    if ca is not None:
        if  ca<1 or ca>5:
            image_path = image_paths[1]
    if img_path is not None:
          image_path = img_path
    else:
            image_path = image_paths[ca]
    # print('step 1')
    image = Image.open(image_path).convert('RGB')  # Ensure it's RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    # Move image to the same device as the model
    image = image.to(device)
    
    # Model inference
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)  # Get index of the highest score
    
    # Map index to class label
    predicted_class = class_names[predicted_idx.item()]
    return predicted_class
#     print(f"Predicted Class: {predicted_class}")
