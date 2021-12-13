import io
import os
import json
import base64

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.conf import settings
from django.shortcuts import render

from .forms import ImageUploadForm
import pretrainedmodels

from .utils import mappings

device = torch.device('cpu')

# load mapping of ImageNet index to human-readable label (from staticfiles directory)
# run "python manage.py collectstatic" to ensure all static files are copied to the STATICFILES_DIRS
# json_path = os.path.join(settings.STATIC_ROOT, 'imagenet_class_index.json')
# imagenet_mapping = json.load(open(json_path))

mapping_age, mapping_gender, mapping_race = mappings()


def transform_image(image_bytes):
    """
    Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
    Return the corresponding tensor.
    """
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


class Net(nn.Module):
    def __init__(self, pretrained):
        super(Net, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["densenet161"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["densenet161"](pretrained=None)
        self.fc1 = nn.Linear(2208, 9)  # For age class
        self.fc2 = nn.Linear(2208, 2)  # For gender class
        self.fc3 = nn.Linear(2208, 7)  # For race class

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label2 = torch.sigmoid(self.fc2(x))
        label3 = self.fc3(x)
        return {'label1': label1, 'label2': label2, 'label3': label3}

# load pretrained DenseNet and go straight to evaluation mode for inference
# load as global variable here, to avoid expensive reloads with each request


# model = models.densenet121(pretrained=False)
model = Net(pretrained=False)
model.to(device)
model.load_state_dict(torch.load('face_classification/model.pt', map_location=device))
model.eval()


def get_prediction(image_bytes):
    """Fir given image bytes, predict the label using the pretrained DenseNet"""
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    age = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs['label1'])))
    gender = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs['label2'])))
    race = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs['label3'])))
    res = []
    res.append(mapping_age[age.item()])
    res.append(mapping_gender[gender.item()])
    res.append(mapping_race[race.item()])
    return res

def index(request):
    image_uri = None
    predicted_label = None

    if request.method == 'POST':
        # in case of POST: get the uploaded image from the form and process it
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # retrieve the uploaded image and convert it to bytes (for PyTorch)
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            # convert and pass the image as base64 string to avoid storing it to DB or filesystem
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            # get predicted label with previously implemented PyTorch function
            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply the empty form for uploading images
        form = ImageUploadForm()

    # pass the form, image URI and predicted label to the template to be rendered
    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }
    return render(request, 'face_classification/index.html', context)