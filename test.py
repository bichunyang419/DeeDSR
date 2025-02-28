from share import *

import pytorch_lightning as pl
from MySDSR.dataload import *
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import time

# Configs
weight_path = ''


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('').cpu()
model.load_state_dict(load_state_dict(weight_path, location='cpu'),strict=False)


with open('./MySDSR/config_test.yaml', encoding='utf-8')as file:
    content = file.read()
    data = yaml.load(content, Loader=yaml.FullLoader)
    # print(data)
    # print(type(data))
opt = data['data']['params']['train']['params']
# Misc
test_set = TestDataset(opt)
test_loader = DataLoader(dataset=test_set, num_workers=data['data']['params']['num_workers'], batch_size=1, shuffle=False)
#logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=[0], precision=32)


# Train!
trainer.test(model, test_loader)
