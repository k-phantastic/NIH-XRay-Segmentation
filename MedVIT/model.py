import torch
import torch.nn as nn
import timm

class MetaGatedInference(nn.Module):
    """
    SOTA Gating: Metadata dynamically scales image features.
    """
    def __init__(self, img_feat_dim, meta_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(meta_dim, img_feat_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feats, meta_feats):
        # Element-wise scaling
        return img_feats * self.gate(meta_feats)

class OptimizedMedViT(nn.Module):
    def __init__(self, num_classes=15, metadata_dim=3):
        super().__init__()
        
        # 1. Image Backbone (MaxViT)
        # Using maxvit_tiny_tf_224. It handles flexible resolutions 
        # (like 320) due to relative position biases.
        self.backbone = timm.create_model('maxvit_tiny_tf_224', pretrained=True, in_chans=1)
        self.img_dim = self.backbone.num_features
        self.backbone.reset_classifier(0) 

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 2. Metadata Encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Gating logic
        self.gating_unit = MetaGatedInference(self.img_dim, 64)
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.img_dim, num_classes)
        )

    def forward(self, x, meta_data):
        
        # Get backbone features
        img_feats = self.backbone.forward_features(x) 
        
        # If the backbone returns a spatial map [B, C, H, W], flatten it
        if len(img_feats.shape) == 4:
            img_feats = self.global_pool(img_feats).flatten(1)
        elif len(img_feats.shape) == 3: # Some ViTs return [B, L, C]
            img_feats = img_feats.mean(dim=1)
            
        meta_feats = self.meta_encoder(meta_data)
        
        # Apply gating
        fused_feats = self.gating_unit(img_feats, meta_feats)
        
        return self.classifier(fused_feats)