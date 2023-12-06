import json
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

def create_pretrained_model():
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def model_fn(model_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=create_pretrained_model()
    model.to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        checkpoint = torch.load(f, map_location =device)
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def input_fn(request_body, content_type='image/jpeg'):
    if content_type == 'image/jpeg': 
        return Image.open(io.BytesIO(request_body))
    
    if content_type == 'application/json':
        request = json.loads(request_body)
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    input_object = test_transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction


def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)