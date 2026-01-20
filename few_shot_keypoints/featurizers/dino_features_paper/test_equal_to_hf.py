"""
Test script to verify that DINOViTPaperFeaturizer and DinoV2SmallFeaturizer 
produce identical outputs when configured correctly.
"""
import torch
from few_shot_keypoints.featurizers.dino_vit_paper.featurizer import ViTPaperFeaturizer
from few_shot_keypoints.featurizers.ViT_featurizer import DinoV2SmallFeaturizer

def test_featurizers_equal():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # DINOv2-S has 12 layers (0-11), patch_size=14
    # Paper featurizer: layer=11 is the last layer (0-indexed)
    # HF featurizer: layers=[-1] is the last layer
    
    # Both should use:
    # - Same model: dinov2_vits14 / facebook/dinov2-small
    # - Same layer: last layer (11 or -1)
    # - facet='token' for paper (to match HF hidden states)
    # - stride=14 (equal to patch_size, no overlap)
    
    print("Creating featurizers...")
    featurizer_paper = ViTPaperFeaturizer(
        model_type='dinov2_vits14',
        stride=14,  # Must equal patch_size for fair comparison
        device=device,
        layer=11,   # Last layer (0-indexed)
        facet='token',  # Use token to match HF hidden_states
        use_bin_features=False
    )
    
    featurizer_hf = DinoV2SmallFeaturizer(
        layers=[-1],  # Last layer
        device=device
    )
    
    # Create test image (must be divisible by patch_size=14)
    # Using 504x504 = 36x36 patches
    print("Creating test image...")
    torch.manual_seed(2026)
    test_image = torch.rand(1, 3, 504, 504)
    
    print(f"Test image shape: {test_image.shape}")
    
    # Extract features
    print("Extracting features with paper featurizer...")
    with torch.no_grad():
        features_paper = featurizer_paper.extract_features(test_image).cpu()
    
    print("Extracting features with HF featurizer...")
    with torch.no_grad():
        features_hf = featurizer_hf.extract_features(test_image).cpu()
    
    print(f"\nPaper featurizer output shape: {features_paper.shape}")
    print(f"HF featurizer output shape: {features_hf.shape}")
    
    # Check shapes match
    if features_paper.shape != features_hf.shape:
        print(f"\n❌ SHAPES DON'T MATCH!")
        print(f"   Paper: {features_paper.shape}")
        print(f"   HF: {features_hf.shape}")
        return False
    
    print(f"\n✓ Shapes match: {features_paper.shape}")
    
    # Check if values are close
    are_close = torch.allclose(features_paper, features_hf, rtol=1e-4, atol=1e-4)
    
    if are_close:
        print("✓ Values are close (rtol=1e-4, atol=1e-4)")
    else:
        # Compute differences
        abs_diff = torch.abs(features_paper - features_hf)
        rel_diff = abs_diff / (torch.abs(features_hf) + 1e-8)
        
        print(f"\n❌ VALUES DON'T MATCH!")
        print(f"   Max absolute difference: {abs_diff.max().item():.6f}")
        print(f"   Mean absolute difference: {abs_diff.mean().item():.6f}")
        print(f"   Max relative difference: {rel_diff.max().item():.6f}")
        print(f"   Mean relative difference: {rel_diff.mean().item():.6f}")
        
        # Check correlation
        flat_paper = features_paper.flatten()
        flat_hf = features_hf.flatten()
        correlation = torch.corrcoef(torch.stack([flat_paper, flat_hf]))[0, 1]
        print(f"   Correlation: {correlation.item():.6f}")
        
        # Sample some values
        print(f"\n   Sample values (first 5 at position [0, :5, 0, 0]):")
        print(f"   Paper: {features_paper[0, :5, 0, 0].tolist()}")
        print(f"   HF:    {features_hf[0, :5, 0, 0].tolist()}")
    
    return are_close

if __name__ == "__main__":
    result = test_featurizers_equal()
    print(f"\n{'='*50}")
    if result:
        print("SUCCESS: Both featurizers produce identical outputs!")
    else:
        print("FAILED: Featurizers produce different outputs.")

