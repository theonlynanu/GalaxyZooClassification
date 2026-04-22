"""
Danyal Ahmed - April 2026

models/basic_cnn.py
A basic from-scratch CNN for the 4-class GZ2 morphology classification task

Running this file as main (`python models/basic_cnn.py`) will run a basic sanity check

Architecture Summary:
    Stem:       7x7 conv    stride=2    32 channels         (3, 224, 244)   ->  (32, 112, 112)
    Block1:     
                3x3 conv    stride=1    64 channels         (32, 112, 112)  ->  (64, 112, 112)
                3x3 conv    stride=2    64 channels         (64, 112, 112)  ->  (64, 56, 56)
    Block2: 
                3x3 conv    stride=1    128 channels        (64, 56, 56)    ->  (128, 56, 56) 
                3x3 conv    stride=2    128 channels        (128, 56, 56)   ->  (128, 28, 28)
    Block3:
                3x3 conv    stride=1    192 channels        (128, 28, 28)    ->  (192, 28, 28) 
                3x3 conv    stride=2    192 channels        (192, 28, 28)   ->  (192, 14, 14)
    Block4:
                3x3 conv    stride=1    256 channels        (192, 14, 14)    ->  (256, 14, 14) 
                3x3 conv    stride=2    256 channels        (256, 14, 14)   ->  (256, 7, 7)
    Head (classifier):
                GAP -> Dropout (0.4) -> Linear(256 -> 4)
                
Design Notes:
- BatchNorm after every conv layer
- ReLU activation throughout
- Stride-2 convolutions instead of maxpool for downsampling so that downsampling is learnable (may undo)
- padding=same for all 3x3 convolutions
- GAP + single Linear to keep parameter count low and allow for Grad-CAM later
- Bias terms omitted during convolution layers since BatchNorm has its own bias term that essentially 
  does the same thing (I think)
- ~1.4M parameters
"""
import torch
import torch.nn as nn

######################## CLASSES ########################

####                Convolution Block               ####
class ConvBlock(nn.Module):
    """
    Single convolutional block for the network.
    
    Applies:
        conv 3x3, stride=1, pad=1 -> BatchNorm -> ReLU
        conv 3x3, stride=2, pad=1 -> BatchNorm -> ReLU
        
    First layer refines features at input resolution, second downsamples by factor of 2
    """
    
    def __init__(self, in_ch: int, out_ch: int):    
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.relu = nn.ReLU(inplace=True)       # inplace=True saves me so much time here wow
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x
    
    
####                Main Model              ####

class BasicCNN(nn.Module):
    """
    Basic CNN for 4-class classification task
    
    Notes:
        - First layer uses a 7x7 stride=2 layer to jumpstart the receptive field
        - Designed with resized (3, 224, 224) images in mind (from dataset.py)
    """
    
    
    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
    
        # Channel counts
        self.STEM_CHANNELS = 32
        self.BLOCK_CHANNELs = [64, 128, 192, 256]
        
        # Stem is a 7x7 conv w/ stride 2 + BatchNorm + ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.STEM_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.STEM_CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        c1, c2, c3, c4 = self.BLOCK_CHANNELs
        self.block1 = ConvBlock(self.STEM_CHANNELS, c1)
        self.block2 = ConvBlock(c1, c2)
        self.block3 = ConvBlock(c2, c3)
        self.block4 = ConvBlock(c3, c4)
        
        # classifier heda
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c4, num_classes)
        
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run all conv layers and return the final feature map WITHOUT 
        running through the classification head.
        
        Mostly so I can use it in Grad-CAM for visualizations.
        
        (B, 3, H, W) -> (B, 256, 7, 7) for a 224x224 input
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass"""
        x = self.gap(self.forward_features(x))
        x = torch.flatten(x, 1)
        x = self.classifier(self.dropout(x))
        return x
    
    
######################## UTILITY/TESTING ########################

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters

    Args:
        model (nn.Module): model to look at
        trainable_only (bool, optional): whether to only count params that
            require grad. Defaults to True.

    Returns:
        int: parameter count
    """
    # Initially did all the multiplications by hand, then discovered that
    # .numel() just does it for you
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return sum(p.numel() for p in model.parameters())


def summarize(
    model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 3, 224, 224)
) -> None:
    """Print a layer-by-layer summary of output shapes and paramete counts

    Args:
        model (nn.Module): model being summarized
        input_shape (tuple[int, int, int, int], optional): input sample batch shape. Defaults to (1, 3, 224, 224).
    """
    model.eval()
    
    # Plan is to create hooks rather than alter the network's forward pass (not
    # even sure how I'd do that)
    shapes: dict[str, tuple] = {}
    hooks = []
    
    # Closure factory for creating a hook into the model's `.forward()` method
    # Gives a dedicated hook function with the name as part of the closure's scope
    def make_hook(name: str):
        def hook(_module, _input, output):
            if isinstance(output, torch.Tensor):
                shapes[name] = tuple(output.shape)
        return hook
    
    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(make_hook(name)))
        
    # The original pattern for this was giving me a bug, as each hook was pointing
    # to the same name variable, leading to them all being named 'classifier'
    # for name, module in model.named_children():
    #     def hook(_module, _input, output):
    #         shapes[name] = tuple(output.shape)
    #     hooks.append(module.register_forward_hook(hook))
        
    with torch.no_grad():
        dummy = torch.zeros(input_shape)
        _ = model(dummy)
        
    for h in hooks:
        h.remove()
        
    print(f"\n{'Layer':<14} {'Output shape':<26} {'Params':>12}")
    print("-"*54)
    
    total_params = 0
    for name, module in model.named_children():
        shape = shapes.get(name, "n/a")
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"{name:<14} {str(shape):<26} {params:>12,}")
        
    print("-" * 54)
    print(f"{'Total':<14} {'':<26} {total_params:>12,}")
    
    
######################## SMOKE TESTING ########################

if __name__ == "__main__":
    print("BasicCNN smoke test")
    print("=" * 54)
    
    model = BasicCNN(num_classes=4, dropout=0.4)
    
    summarize(model, input_shape=(1, 3, 224, 224))
    
    print("\nForward pass check (batch size = 8)")
    # Just using random values here - using actual images is more work than needed
    x = torch.randn(8, 3, 224, 224)
    logits = model(x)
    features = model.forward_features(x)
    
    print(f"    input shape     = {tuple(x.shape)}")
    print(f"    features shape  = {tuple(features.shape)}")
    print(f"    logits shape    = {tuple(logits.shape)}")
    print(f"    input range     = [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # Check that loss backprops
    print("\nGradient flow check (all should be nonzero):")
    hard_labels = torch.randint(0, 4, (8,))
    loss = nn.functional.cross_entropy(logits, hard_labels)
    loss.backward()
    
    with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() >0)
    total = sum(1 for p in model.parameters())
    print(f"    {with_grad}/{total} parameter tensors received with nonzero gradients")
    print(f"    loss = {loss.item():.4f}")
    
    print("\nDone")
    