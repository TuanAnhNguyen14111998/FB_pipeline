import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2
import yaml
from PIL import Image
from lib.Chest_cls.models.chest_notchest import CNCNet
from collections import namedtuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess(image, cfg):
    cfg = namedtuple('cfg', cfg.keys())(*cfg.values())
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    trans = transforms.Compose(
        [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
         transforms.ToTensor(), normalize])

    return trans(img)


def load_model(m_params, trained_model_file, cfg):
    model = CNCNet(cfg=m_params)
    model = model.to(device)
    state_dict = torch.load(trained_model_file, map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


class Chest_cls_inference:
    def __init__(self, model_path = None, config = None, device = device):
        with open(config) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        cfg['model_params']['pretrained'] = False
        self.m_params = cfg['model_params']
        self.t_params = cfg['train_params']
        model = load_model(self.m_params,model_path,cfg)
        self.model = model.to(device)
        self.chest_types = ['Not chest', 'Frontal chest', 'Lateral chest', 'Chest of children']
        self.device = device
    
    def __call__(self, image):
        image = preprocess(image, self.m_params)
        image = image.unsqueeze(0)
        logit = self.model(image.to(self.device)).detach().cpu().squeeze()
        label_prob = F.softmax(logit)
        label_index = torch.argmax(label_prob).item()
        if label_index == 1 or label_index == 3:
            return True
        else:
            return False
