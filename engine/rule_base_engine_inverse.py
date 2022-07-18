from torch.utils import data
from collections import namedtuple
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lib.Lung_segment.inference import LungSegmentation
from lib.bone_segmentation.inference import BoneSegmentation
from lib.Chest_cls.inference import Chest_cls_inference
from utils.utils import *
from utils.config import Cfg
from lib.image_quality_check.rule_base import *
from utils.utils import read_image_url


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def default_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    data_origin = [item[0] for item in batch]
    data_image_id = [item[1] for item in batch]
    data_image_url = [item[2] for item in batch]
    data_doctor_border = [item[3] for item in batch]
    data_chest_cls = torch.stack([item[4] for item in batch])
    data_lung_segment = torch.stack([item[5] for item in batch])

    return data_origin, data_image_id, data_image_url, \
        data_doctor_border, data_chest_cls, data_lung_segment


class Dataset(data.Dataset):
    def __init__(self, df_info, disease_names, img_size, crop_size, round_two):
        self.df_info = df_info
        self.img_size = img_size
        self.crop_size = crop_size
        self.disease_names = disease_names
        self.round_two = round_two
        
    def __len__(self):
        return len(self.df_info)

    def __getitem__(self, index):
        try:
            image_url = self.df_info.iloc[index]["image_url"]
            image_id = self.df_info.iloc[index]["image_id"]
            image_origin = read_image_url(image_url)
            if self.round_two:
                image_origin = cv2.bitwise_not(image_origin)
            mask_border = {k: self.df_info.iloc[index][f"{k}_border"] for k in self.disease_names}
            image_chest_cls = preprocess_chest_cls(image_origin, self.img_size, self.crop_size)
            image_lung_segment = preprocess_lung_segmentation(image_origin)
        except:
            return None
        
        return image_origin, image_id, image_url, \
            mask_border, image_chest_cls, image_lung_segment


class RuleBaseEngine():
    def __init__(self, config_path=None):
        config = Cfg.load_config_from_file("config/config.yml")

        self.Chest_cls = Chest_cls_inference(
            model_path = config["weight_chest_cls"],
            config = config["yml_config_chest_cls"]
        )
        self.segments = LungSegmentation(
            checkpoint_path = config["weight_lung_segmentation"], 
            device=device, threshold = 0.1
        )
        self.clav_seg = BoneSegmentation(
            weight_path = config["weight_bone_segmentation"],
            img_size=512, device=device
        )
        self.df = pd.read_csv(config["path_dataset"])
        self.dictionary_process = {
            "lung_opacity": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "lung_lesion": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "atelectasis": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "consolidation": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "pneumonia": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "edema": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "cavitation": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "pulmonary_scar": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "widening_mediastinum": {
                "rule_function": mediastinum_process,
                "model_other_segmentation": None
            },
            "pleural_other": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "medical_device": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "other_findings": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "covid_19": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "tuberculosis": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "nodule": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "mass": {
                "rule_function": inside_lung_process,
                "model_other_segmentation": None
            },
            "cardiomegaly": {
                "rule_function": cardiomegaly_process,
                "model_other_segmentation": None
            },
            "fracture": {
                "rule_function": fracture_process,
                "model_other_segmentation": self.clav_seg
            },
            "enlarged_cardiomediastinum": {
                "rule_function": mediastinum_process,
                "model_other_segmentation": None
            },
            "pneumothorax": {
                "rule_function": pneumothorax_process,
                "model_other_segmentation": None
            },
            "pleural_effusion": {
                "rule_function": pleural_effusion_process,
                "model_other_segmentation": None
            }
        }
        self.disease_names = list(self.dictionary_process.keys())
        self.dictionary_result = {k: [] for k in ["image_id", "image_url"] + self.disease_names}
        self.params = {
            'batch_size': 132,
            'shuffle': False,
            'num_workers': 6
        }
        self.cfg = namedtuple('cfg', self.Chest_cls.m_params.keys())(*self.Chest_cls.m_params.values())
        self.path_save_result = config["path_save_result"]
        self.round_two = False
    
    
    def run_rule_base(self, round_two=False):
        if round_two:
            df_result = pd.read_csv(self.path_save_result)
            self.dictionary_result = df_result[df_result["lung_opacity"] != -2].to_dict('list')
            df_result = df_result[df_result["lung_opacity"] == -2]
            image_id = df_result.image_id.tolist()
            self.df = self.df[self.df['image_id'].isin(image_id)]
            self.round_two = round_two
        
        data_loader = Dataset(
            df_info=self.df, 
            img_size=self.cfg.img_size, 
            crop_size=self.cfg.crop_size, 
            disease_names=self.disease_names,
            round_two=self.round_two
        )
        data_generator = data.DataLoader(data_loader, collate_fn=default_collate, **self.params)

        with torch.no_grad():
            with tqdm(data_generator, unit="batch") as tepoch:
                for local_image_origin, local_data_image_id, \
                    local_data_image_url, local_data_doctor_border, \
                    local_image_chest_cls, local_image_lung_segment in tepoch:

                    local_image_chest_cls = local_image_chest_cls.to(device)
                    local_image_lung_segment = local_image_lung_segment.to(device)

                    logit = self.Chest_cls.model(local_image_chest_cls)
                    label_prob = F.softmax(logit)
                    label_index = torch.argmax(label_prob, dim=1)

                    org_h = [item.shape[0] for item in local_image_origin]
                    org_w = [item.shape[1] for item in local_image_origin]
                    
                    outputs = self.segments.predict_batch(
                        org_h=org_h, org_w=org_w,
                        img_tensor=local_image_lung_segment,
                        threshold=0.1
                    )
                    for index, image_origin in enumerate(local_image_origin):
                        self.dictionary_result["image_id"].append(local_data_image_id[index])
                        self.dictionary_result["image_url"].append(local_data_image_url[index])
                        try:
                            if (label_index[index] == 1) or (label_index[index] == 3):
                                height, width = image_origin.shape[:2]
                                lung_segmentation_nested = lung_border(lung_raw=outputs[index])
                                # lung_segmentation_nested = outputs[index]
                                for disease_name in self.disease_names:
                                    if lung_segmentation_nested is not None:
                                        mask_doctor = local_data_doctor_border[index][f"{disease_name}"]
                                        if not pd.isnull(mask_doctor) and mask_doctor != '[{"polygon":[]}]' \
                                            and mask_doctor != '[]':
                                            mask_doctor = np.array([mask_doctor])
                                            dr_border = mask_image(mask_doctor[0], height, width).reshape((height,width))

                                            model_other_segmentation = self.dictionary_process[disease_name]["model_other_segmentation"]
                                            status = int(self.dictionary_process[disease_name]["rule_function"](
                                                    mask_doctor=dr_border,
                                                    lung_segmentation=lung_segmentation_nested,
                                                    other_segmentation=get_other_segmentation(
                                                        image=image_origin,
                                                        model=model_other_segmentation
                                                    )
                                                )
                                            )
                                            self.dictionary_result[disease_name].append(status)
                                        else:
                                            self.dictionary_result[disease_name].append(-1)
                                    else:
                                        self.dictionary_result[disease_name].append(-3)
                            else:
                                for disease_name in self.disease_names:
                                    self.dictionary_result[disease_name].append(-2)
                        except:
                            for disease_name in self.disease_names:
                                self.dictionary_result[disease_name].append(-4)
                    
                    pd.DataFrame(dict([(k,pd.Series(v)) for k,v in self.dictionary_result.items()])).to_csv(self.path_save_result, index = False)

    def __call__(self):
        self.run_rule_base(round_two=False)
        self.run_rule_base(round_two=True)

        return None