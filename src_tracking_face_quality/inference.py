import json
from tqdm import tqdm
from PIL import Image
import os
import yaml
import random
import importlib
from typing import Tuple
from torch import nn
import cv2
import numpy as np



def assess_sharpness(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()  # Variance of the Laplacian

def assess_blurriness_fft(gray_image):
    """Assess blurriness using FFT."""
    # Perform FFT and get the magnitude spectrum
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Compute the mean frequency; a higher mean frequency indicates a sharper image
    return np.mean(magnitude_spectrum)


import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
class ImageDataset():

    def __init__(self, 
                 movie_tracker_output, 
                 trans: Compose) -> None:
        """ Helper class that loads images from a movie_tracker_output.

        Args:
            image_loc (str): The location of the directory containing the desired images.
            trans (Compose): Transformations used on loaded images.
        """

        self.trans = trans

        self.items = movie_tracker_output

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, x):
        face_data = self.items[x]

        n_track_id = str(face_data["n"]) + "_" + str(face_data["track_id"])
        # Convert the image from BGR to RGB

        face = face_data["face"].copy() 
        rgb_img = Image.fromarray(np.uint8(face)).convert('RGB')
        return (n_track_id, self.trans(rgb_img))

ediffiqaL_config_yaml = """
base_model:
  module: "model.iresnet.iresnet100"
  weights: "weights/r100.pth"
  transformations:
    trans_1:
      module: "torchvision.transforms.Resize"
      params:
        size: [112, 112]
    trans_2:
      module: "torchvision.transforms.ToTensor"
    trans_3:
      module: "torchvision.transforms.Normalize"
      params: 
        mean: [.5, .5, .5]
        std: [.5, .5, .5]

mlp:
  module: "model.mlp.MLP"
  params:
      in_dim: 512
      hidden_dim: 1024
      out_dim: 1

ediffiqa:
  module: "model.ediffiqa.eDifFIQA"
  params:
    return_feat: 0
"""

def seed_all(seed: int) -> None:
    """ Seeds python.random, torch and numpy,

    Args:
        seed (int): Desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forard_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out        

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forard_imlp, x)
        else:
            return self.forard_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model

def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)

class MLP(torch.nn.Module):

    def __init__(self, 
                 in_dim :     int = 512, 
                 hidden_dim : int = 1024, 
                 out_dim :    int = 1):
        super().__init__()

        self.l1 = torch.nn.Linear(in_dim, hidden_dim)
        self.ac = torch.nn.GELU()
        self.l2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.ac(self.l1(x))
        return self.l2(x)

class eDifFIQA(torch.nn.Module):
    """ eDifFIQA model consisting of a pretrained FR backbone (CosFace in the original implementation) and a quality regression MLP head.

    Args:
        base_model (torch.nn.Module): FR backbone used for feature extraction.
        mlp (torch.nn.Module): MLP used as a quality regression head.
        return_feat (bool): Flag for returning features, if set to True the model returns (features, qualities) otherwise only the qualities.
    """

    def __init__(self, 
                 backbone_model : torch.nn.Module,
                 quality_head   : torch.nn.Module,
                 return_feat    : bool = True):
        super().__init__()

        self.base_model = backbone_model
        self.mlp = quality_head
        self.return_feat = return_feat

    def forward(self, x):
        feat = self.base_model(x)
        pred = self.mlp(feat)
        if self.return_feat:
            return (feat, pred)
        return pred

class Arguments:
    """ Empty class definition for constructing any type of arguments from configs.
    """
    pass


def convert_single_layer(config: dict, arguments_obj: Arguments):
    """ Converts a single layer (level) of a config dictionary into the Arguments class

    Args:
        config (dict): Dictionary of provided configuration.
        arguments_obj (Arguments): Argument class of the parent configuration.

    Returns:
        Arguments: Argument class of current configuration.
    """

    for key, value in config.items():
        if isinstance(value, dict):
            setattr(arguments_obj, key, convert_single_layer(value, Arguments()))
        else:
            setattr(arguments_obj, key, value)
    
    return arguments_obj

def construct_transformation(transformation_arguments: Arguments) -> Compose:
    """ Constructs a composition of transformations given by the transformation arguments.

    Args:
        transformation_arguments (Arguments): Arguments of the transformation.

    Returns:
        Compose: Torchvision Compose object of given transformations.
    """

    transforms_list = []
    idx = 1
    while True:
        try:
            trans_args = getattr(transformation_arguments, f"trans_{idx}")
            module_name, function_name = trans_args.module.rsplit(".", 1)
            module = importlib.import_module(module_name)
            trans_function = getattr(module, function_name)
            if hasattr(trans_args, "params"):
                transforms_list.append(trans_function(**vars(getattr(trans_args, "params"))))
            else:
                transforms_list.append(trans_function())
        except AttributeError:
            break
        idx += 1

    return Compose(transforms_list)

def construct_full_model_ad() -> Tuple[torch.nn.Module, Compose]:
    """ Construct a torch.nn.Module model and its transform from the config provided in config_loc.

    Args:
        config_loc (str): Location of the config file containing information about the model.

    Returns:
        Tuple[torch.nn.Module, Compose]: Returns the constructed model and given transformation.
    """
    config = yaml.safe_load(ediffiqaL_config_yaml)
    args = convert_single_layer(config, Arguments())

    base_model_args = args.base_model
    base_model = iresnet100()
    trans = construct_transformation(base_model_args.transformations)
 
    model_head = MLP(in_dim = 512, hidden_dim = 1024, out_dim = 1)
    
    ediffiqa_function = eDifFIQA
    ediffiqa_model = ediffiqa_function(
        backbone_model=base_model,
        quality_head=model_head,
        return_feat = False
        )
    return ediffiqa_model, trans

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model_name =  'ediffiqaL.pth'
    weight_location = f"model/{model_name}"
    
    model, trans = construct_full_model_ad()
    model.load_state_dict(torch.load(weight_location))
    
    device = torch.device("cuda")
    model = model.to(device).eval()
    return {"model": model, "trans": trans}

def input_fn(request_body, request_content_type):
    """
    Preprocess the incoming request.

    Parameters:
    request_body (str): The body of the incoming request.
    request_content_type (str): The content type of the incoming request.

    Returns:
    np.array: The preprocessed image.
    """
    # Check the content type
    if request_content_type == 'application/json':
        # Parse the JSON object
        request_data = json.loads(request_body)
        return request_data

    raise ValueError(f"Unsupported content type: {request_content_type}")
    
@torch.no_grad()
def predict_fn(input_data, model_trans):
    movie_tracker_output = input_data["movie_tracker_output"].copy()
    nufs = input_data["nufs"]
    
    n_names = len(movie_tracker_output)
    
    trans = model_trans["trans"]
    model = model_trans["model"]
    
    # Construct the Image Dataloader 
    dataset = ImageDataset(movie_tracker_output, trans)
    dataloader = DataLoader(dataset, batch_size = 64)
    
    device = torch.device("cuda")
    
    # Predict quality scores 
    quality_scores = {}

    for (name_batch, img_batch) in tqdm(dataloader, 
                                        desc=" Inference ", 
                                        disable=False):

         img_batch = img_batch.to(device)
         preds = model(img_batch).detach().squeeze().cpu().numpy()
         if  n_names == 1:
             preds = [preds]

         if len(name_batch) == 1: 
            preds = [preds]
         quality_scores.update(dict(zip(name_batch, preds)))

    del model   
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return {"movie_tracker_output": movie_tracker_output, "quality_scores": quality_scores, "nufs": nufs}
        
def output_fn(prediction_output, content_type):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super(NumpyEncoder, self).default(obj)

    quality_scores = prediction_output["quality_scores"].copy()
    output = []
    for elem in prediction_output["movie_tracker_output"]:
        n_track_id = str(elem["n"]) + "_" + str(elem["track_id"])
        elem["face_quality"] = quality_scores[n_track_id]
          
        ###############################################
        gray_image = cv2.cvtColor(elem["face"], cv2.COLOR_BGR2GRAY)
        sharp_score = assess_sharpness(gray_image)
        blur_score = assess_blurriness_fft(gray_image)
        
        elem["sharp_score"] = sharp_score
        elem["blur_score" ] = blur_score
        ############################################### 
        
        output.append(elem)

    processed_input_data = {"movie_tracker_output": output, "nufs": prediction_output["nufs"]}
    return json.dumps(processed_input_data, cls = NumpyEncoder)

def NEW_output_fn(prediction_output):
    quality_scores = prediction_output["quality_scores"].copy()
    output = []
    for elem in prediction_output["movie_tracker_output"]:
        n_track_id = str(elem["n"]) + "_" + str(elem["track_id"])
        elem["face_quality"] = quality_scores[n_track_id]
        ###############################################
        gray_image = cv2.cvtColor(elem["face"], cv2.COLOR_BGR2GRAY)
        sharp_score = assess_sharpness(gray_image)
        blur_score = assess_blurriness_fft(gray_image)
        
        elem["sharp_score"] = sharp_score
        elem["blur_score" ] = blur_score
        ############################################### 

        output.append(elem)

    processed_input_data = {"movie_tracker_output": output, "nufs": prediction_output["nufs"]}
    return processed_input_data
