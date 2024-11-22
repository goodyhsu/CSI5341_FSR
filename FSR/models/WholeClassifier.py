import torch
from .BaseModel import BaseModelDNN

class WholeClassifier(BaseModelDNN):
    def __init__(self, args, num_classes, image_size, net, device) -> None:
        super(BaseModelDNN).__init__()
        self.device = device
        self.net = net(tau=args.tau, num_classes=num_classes, image_size=image_size).to(device)
        self.set_requires_grad([self.net], False)

    def load(self, args):
        checkpoint = torch.load('./weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.load_name), map_location=self.device)
        self.net.load_state_dict(checkpoint)
    
    def predict(self, x, is_eval=True):
        return self.net(x, is_eval=is_eval)
