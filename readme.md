# Tensor Operations Library

A comprehensive PyTorch-based library for advanced tensor operations, merging, and interpolation techniques. This library provides a flexible framework for performing complex mathematical operations on tensors with built-in caching and operation chaining capabilities.

## Features

- 🔗 **Operation Chaining**: Build complex tensor operation graphs with automatic dependency resolution
- 💾 **Smart Caching**: LRU cache system to optimize memory usage and computation speed
- 🎯 **Advanced Merging**: Multiple sophisticated tensor merging algorithms including power-up, interpolation, and similarity-based methods
- 🧠 **ML-Focused**: Designed specifically for machine learning model weight manipulation and merging
- 📊 **Statistical Operations**: Built-in smoothing, difference calculations, and similarity measurements

## Installation

### Requirements

```bash
pip install torch numpy scipy
```

### Quick Start

```python
from operators import *

# Load your tensors into checkpoints
checkpoint_a = {"layer1.weight": torch.randn(128, 256)}
checkpoint_b = {"layer1.weight": torch.randn(128, 256)}

load_checkpoint("model_a", checkpoint_a)
load_checkpoint("model_b", checkpoint_b)

# Create and execute operations
load_a = LoadTensor("layer1.weight", "model_a")
load_b = LoadTensor("layer1.weight", "model_b")
add_op = Add("result", load_a, load_b)

result = add_op.merge()
```

## Core Components

### Operation Base Class

All operations inherit from the `Operation` base class, which provides:
- Automatic hashing and equality for caching
- Operation chaining through source dependencies
- Configurable merge functions with caching support

### Configuration

The library uses a global `Config` class to manage:
- Device selection (CPU/CUDA)
- Data type (default: float32)
- Cache size limits
- Loaded checkpoint storage

## Available Operations

### Basic Operations

#### LoadTensor
Loads tensors from pre-loaded checkpoints.

```python
load_op = LoadTensor("layer1.weight", "checkpoint_name")
tensor = load_op.merge()
```

#### Multiply
Multiplies a tensor by a scalar value.

```python
multiply_op = Multiply("key", 0.5, source_operation)
```

#### Add / Sub
Performs element-wise addition or subtraction between tensors.

```python
add_op = Add("key", source_a, source_b)
sub_op = Sub("key", source_a, source_b)
```

### Advanced Operations

#### Smooth
Applies median and Gaussian filtering for tensor smoothing.

```python
smooth_op = Smooth("key", source_operation)
```

#### TrainDiff
Calculates training differences with distance-based scaling.

```python
train_diff_op = TrainDiff("key", source_a, source_b, source_c)
```

#### Extract
Performs cosine similarity-based extraction with configurable parameters.

```python
extract_op = Extract("key", alpha=0.5, beta=0.3, gamma=2.0, base_op, source_a, source_b)
```

#### PowerUp
Implements dropout-based weight merging using Bernoulli masks.

```python
powerup_op = PowerUp("key", alpha=0.3, seed=42, source_a, source_b)
```

#### InterpolateDifference
Advanced interpolation based on tensor differences with stochastic masking.

```python
interp_op = InterpolateDifference("key", alpha=0.8, beta=1.0, gamma=0.5, seed=123, source_a, source_b)
```

#### Enhanced Interpolation Variants

**ManualEnhancedInterpolateDifference**: Manual threshold control
```python
manual_op = ManualEnhancedInterpolateDifference("key", alpha=0.7, beta=0.1, gamma=0.9, delta=0.3, seed=456, source_a, source_b)
```

**AutoEnhancedInterpolateDifference**: Automatic threshold adaptation
```python
auto_op = AutoEnhancedInterpolateDifference("key", alpha=0.6, beta=0.2, gamma=0.4, seed=789, source_a, source_b)
```

#### WeightSumCutoff
Conditional interpolation based on statistical thresholds.

```python
cutoff_op = WeightSumCutoff("key", alpha=0.5, beta=0.3, gamma=0.7, source_a, source_b)
```

## Parameter Descriptions

### Common Parameters

- **alpha**: Primary interpolation/scaling factor (typically 0.0-1.0)
- **beta**: Secondary threshold or scaling parameter
- **gamma**: Tertiary parameter, often for power/smoothing functions
- **delta**: Quaternary parameter for advanced operations
- **seed**: Random seed for reproducible stochastic operations

### Operation-Specific Parameters

| Operation | Alpha | Beta | Gamma | Delta | Seed |
|-----------|-------|------|-------|--------|------|
| Extract | Interpolation strength | Similarity weight | Cosine power | - | - |
| PowerUp | Dropout rate | - | - | - | Random seed |
| InterpolateDifference | Power factor | Mode selector | Interpolation weight | - | Random seed |
| ManualEnhanced | Power factor | Lower threshold | Upper threshold | Smoothness | Random seed |
| AutoEnhanced | Power factor | Threshold adjustment | Smoothness | - | Random seed |

## Caching System

The library includes an intelligent caching system:

```python
# Enable caching for operations
operation.cache()

# Configure cache size (in MB)
opts.cache_size = 1024  # 1GB cache
```

### Cache Features

- **LRU Eviction**: Least recently used tensors are removed first
- **Memory Management**: Automatic size tracking and cleanup
- **GPU-Aware**: Moves cached tensors to CPU to save GPU memory
- **Transparent**: Operations work identically with or without caching

## Example: Model Weight Merging

```python
# Load two different model checkpoints
model_a_weights = torch.load("model_a.pt")
model_b_weights = torch.load("model_b.pt")

load_checkpoint("model_a", model_a_weights)
load_checkpoint("model_b", model_b_weights)

# Create merging operations for each layer
merged_weights = {}
for layer_name in model_a_weights.keys():
    load_a = LoadTensor(layer_name, "model_a")
    load_b = LoadTensor(layer_name, "model_b")
    
    # Use PowerUp merging with 30% dropout
    merge_op = PowerUp(layer_name, alpha=0.3, seed=42, load_a, load_b)
    merge_op.cache()  # Enable caching for this operation
    
    merged_weights[layer_name] = merge_op.merge()

# Save merged model
torch.save(merged_weights, "merged_model.pt")
```

## Example: Complex Operation Chain

```python
# Create a complex operation chain
load_base = LoadTensor("layer.weight", "base_model")
load_a = LoadTensor("layer.weight", "model_a")  
load_b = LoadTensor("layer.weight", "model_b")

# First, extract features using cosine similarity
extract_op = Extract("extracted", 0.7, 0.3, 2.0, load_base, load_a, load_b)

# Then smooth the result
smooth_op = Smooth("smoothed", extract_op)

# Finally, interpolate with original
final_op = InterpolateDifference("final", 0.8, 1.0, 0.5, 123, load_base, smooth_op)

# Execute the entire chain
result = final_op.merge()
```

## Performance Tips

1. **Enable Caching**: Use `.cache()` on operations that will be reused
2. **Batch Operations**: Group related operations to maximize cache hits  
3. **Device Management**: Keep tensors on appropriate device (GPU/CPU)
4. **Memory Monitoring**: Adjust cache size based on available memory

## Configuration Options

```python
# Global configuration
opts.cache_size = 2048  # Cache size in MB
opts.device = torch.device('cuda:0')  # Force specific device
opts.dtype = torch.float16  # Use half precision
```

## Advanced Usage

### Custom Operations

Extend the `Operation` base class to create custom tensor operations:

```python
class MyCustomOperation(Operation):
    def __init__(self, key, custom_param, *sources):
        super().__init__(key, *sources)
        self.custom_param = custom_param
    
    def oper(self, *tensors):
        # Implement your custom logic here
        return some_function(*tensors, self.custom_param)
```

### Operation Introspection

```python
# Check operation dependencies
print(f"Operation sources: {operation.sources}")
print(f"Operation parameters: α={operation.alpha}, β={operation.beta}")

# Verify cache status
print(f"Cached: {operation in weights_cache.mapping}")
```

## Contributing

This library is designed to be extensible. When adding new operations:

1. Inherit from the `Operation` base class
2. Implement the `oper()` method with your tensor logic
3. Use appropriate parameter names (alpha, beta, gamma, delta)
4. Include proper error handling and assertions
5. Add comprehensive docstrings and examples

## License

[Specify your license here]

## References

- Based on techniques from [sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger)
- PowerUp implementation follows [arXiv:2311.03099](https://arxiv.org/pdf/2311.03099.pdf)
- Additional merging techniques from [MergeLM](https://github.com/yule-BUAA/MergeLM/tree/main/model_merging_methods)
