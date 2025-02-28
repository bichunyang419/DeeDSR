import sys
import os

#assert len(sys.argv) == 3, 'Args are wrong.'

#input_path = sys.argv[1]
#output_path = sys.argv[2]

#assert os.path.exists(input_path), 'Input model does not exist.'
#assert not os.path.exists(output_path), 'Output filename already exists.'
#assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from moco_degrad_sd.Blocks import ResNet,BasicBlock


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = ResNet(BasicBlock, [2, 2, 2, 2])


cnn_pretrained_weights = torch.load('checkpoint_0267.pth.tar')
if 'state_dict' in cnn_pretrained_weights:
    cnn_pretrained_weights = cnn_pretrained_weights['state_dict']

scratch_dict = model.state_dict()


target_dict = {}

for k in scratch_dict.keys():
    copy_k = 'module.encoder_q.' + k
    target_dict[k] = cnn_pretrained_weights[copy_k].clone()
    print(f'These weights are newly added: {k}')
model.load_state_dict(target_dict)
torch.save(model.state_dict(), '300.ckpt')
print('Done.')
