# model_manager.py
import torch
from resnet.resnet import ResNet, ResNetBackbone, Bottleneck

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIVE_MODEL = None
ACTIVE_MODEL_NAME = None
ACTIVE_MODEL_PATH = None

def load_model(model_path: str):
    global ACTIVE_MODEL, ACTIVE_MODEL_PATH
    ACTIVE_MODEL = model
    ACTIVE_MODEL_PATH = model_path
    model = ResNetBackbone(
        Bottleneck, [3, 8, 36, 3],
        embedding_size=512
    )

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    ACTIVE_MODEL = model
    ACTIVE_MODEL_NAME = model_path

    return model
