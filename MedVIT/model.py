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
        # The metadata learned mask scales the image features element-wise
        return img_feats * self.gate(meta_feats)

class OptimizedMedViT(nn.Module):
    def __init__(self, num_classes=15, metadata_dim=3):
        super().__init__()
        
        # 1. Image Backbone (MaxViT)
        # in_chans=1 because NIH images are grayscale
        self.backbone = timm.create_model('maxvit_tiny_tf_224', pretrained=True, in_chans=1)
        self.img_dim = self.backbone.num_features
        self.backbone.reset_classifier(0) 

        # 2. Metadata Encoder
        # Using LayerNorm instead of BatchNorm1d for stability with small batches
        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Dropout(0.1) # Added slight dropout for regularization
        )
        
        # 3. Gating logic (Late Fusion)
        self.gating_unit = MetaGatedInference(self.img_dim, 64)
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.img_dim, num_classes)
        )

    def forward(self, x, meta_data):
        # x shape: [batch, 1, 224, 224]
        # meta_data shape: [batch, 3]
        
        img_feats = self.backbone(x)     # [batch, img_dim]
        meta_feats = self.meta_encoder(meta_data) # [batch, 64]
        
        # Apply gating: metadata weights the importance of image features
        fused_feats = self.gating_unit(img_feats, meta_feats)
        
        return self.classifier(fused_feats)