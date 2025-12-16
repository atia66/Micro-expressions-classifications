import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.features = None
        
        self.target_layer = self._find_last_conv()

        self._register_hooks()

    def _find_last_conv(self):
        last_conv = None
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                last_conv = layer
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.features = out  # the output of layer

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0] # the graidents of backwawrd through the layer 

        self.target_layer.register_forward_hook(forward_hook)
        
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor):
        
        self.model.zero_grad()

        output = self.model(img_tensor)
        class_idx = output.argmax(1).item()

        loss = output[0, class_idx]
        
        loss.backward()

        grads = torch.mean(self.gradients, dim=[0, 2, 3])  
        cam = self.features[0]

        for i in range(len(grads)):
            cam[i] *= grads[i]

        cam = cam.detach().cpu().numpy()
        cam=np.maximum(cam,0)
        cam = np.mean(cam, axis=0) # per patch 
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)   # avoid division by zero
        cam = (cam * 255).astype(np.uint8)

        cam = cv2.resize(cam, (48, 48))
        return cam
