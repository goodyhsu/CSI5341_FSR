import torch
from .BaseModel import BaseModelDNN

class SeparateClassifier(BaseModelDNN):
    # Model for transfer learning, the backbone & FSR module might be trained on different datasets
    def __init__(self, args, num_classes, image_size, net, device):
        super(BaseModelDNN).__init__()
        self.net = net(tau=args.tau, num_classes=num_classes, image_size=image_size).to(device)
        self.set_requires_grad([self.net], False)
        self.device = device
        
    def load(self, args):
        print(f'Loading model with backbone: {args.backbone_weights} and FSR: {args.fsr_weights}')
        backbone_checkpoint = torch.load(args.backbone_weights, map_location=self.device)
        self.net.load_state_dict(backbone_checkpoint)
        
        # Replace the FSR module
        fsr_checkpoint = torch.load(args.fsr_weights, map_location=self.device)
        fsr_state_dict = {
            k.replace('fsr.', ''): v
            for k, v in fsr_checkpoint.items()
            if k.startswith('fsr.')
        }

        self.net.fsr.load_state_dict(fsr_state_dict)

    def predict(self, x, is_eval=True):
        return self.net(x, is_eval=is_eval)
