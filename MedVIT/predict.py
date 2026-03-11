import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Internal Imports
from model import OptimizedMedViT
from data_loader import get_loaders

class GradCAM:
    """Generates heatmaps to visualize where the model is 'looking'."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, meta_tensor, class_idx):
        # We need gradients for CAM, so we don't use torch.no_grad() here
        self.model.zero_grad()
        
        # Ensure input requires grad
        input_tensor.requires_grad = True 
        
        output = self.model(input_tensor, meta_tensor)
        loss = output[0, class_idx]
        loss.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations. Check target_layer.")
            
        # Global Average Pooling of the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU to focus on features that have a positive influence on the class
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)
        return heatmap.detach().cpu().numpy()

def run_inference(img_path, age, gender, view, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    model = OptimizedMedViT(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 2. Prepare Image and Metadata
    raw_image = Image.open(img_path).convert('L')
    input_tensor = data_transforms(raw_image).unsqueeze(0).to(device)
    
    view_enc = 1 if view == 'AP' else 0
    gender_enc = 1 if gender == 'M' else 0
    meta_tensor = torch.tensor([[view_enc, age/100.0, gender_enc]], dtype=torch.float32).to(device)

    # 3. Setup Grad-CAM
    # Target the last convolutional layer of the MaxViT backbone
    target_layer = model.backbone.stages[-1].blocks[-1]
    cam = GradCAM(model, target_layer)

    # 4. Predict
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor, meta_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # 5. Visualize Top Class
    top_idx = np.argmax(probs)
    heatmap = cam.generate_heatmap(input_tensor, meta_tensor, top_idx)
    
    # Overlay logic
    img_np = np.array(raw_image.resize((448, 448)))
    heatmap_resized = cv2.resize(heatmap, (448, 448))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR), 0.6, heatmap_color, 0.4, 0)

    # 6. Plot and Save
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title(f"Original X-ray\nAge: {age}, View: {view}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title(f"Grad-CAM: {class_names[top_idx]}\nConf: {probs[top_idx]:.2f}")
    
    # Generate a filename based on the class and age
    save_filename = f"gradcam_{class_names[top_idx]}_{age}.png"
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    print(f"Result saved as {save_filename}")
    plt.show()

# Example Usage:
# run_inference('test_xray.png', 45, 'M', 'PA', 'best_medvit_model.pth')