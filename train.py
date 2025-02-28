from share import *

import pytorch_lightning as pl
from MySDSR.dataload import *
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint

batch_size = 2
logger_freq = 300
learning_rate = 1e-5
deg_locked = True

checkpoint_callback = ModelCheckpoint(
    filename='model_{epoch:02d}_{iteration:04d}_{train_loss:.4f}',  # Filename pattern
    save_top_k=-1,  # Save all models
    every_n_val_epochs=1,  # Save every 10 validation epochs
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/DeeDSR.yaml').cpu()
model.load_state_dict(load_state_dict('./degradSD630.ckpt', location='cpu'))  #todo
model.learning_rate = learning_rate
model.deg_locked = deg_locked

with open('./MySDSR/config.yaml', encoding='utf-8')as file:
    content = file.read()
    data = yaml.load(content, Loader=yaml.FullLoader)

opt = data['data']['params']['train']['params']
# Misc
train_set = RealESRGANDataset(opt)
train_loader = DataLoader(dataset=train_set, num_workers=data['data']['params']['num_workers'], batch_size=batch_size, shuffle=True, drop_last=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=[0,1,2,3], precision=32, callbacks=[logger, checkpoint_callback], accelerator='ddp', accumulate_grad_batches=16) 


# Train!
trainer.fit(model, train_loader)
