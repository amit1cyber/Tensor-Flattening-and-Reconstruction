# Conclusion

The assignment successfully implemented the transformation

```text
B × C × H × W → B × (CHW) → B × C × H × W
```

using explicit indexing and arithmetic address generation rather than direct library shape-conversion functions.

The flattening index was generated using:

```text
k = c × H × W + h × W + w
```

For reconstruction, the channel, height, and width indices were recovered using integer division and modulo operations.

The implementation was tested on an actual MNIST sample, an actual CIFAR-10 sample, and synthetic feature maps with channel counts of 8, 16, 32, 64, 128, 256, and 500.

All experiments produced:

```text
Maximum Absolute Error = 0
Mean Absolute Error    = 0
```

Therefore, the conversion was lossless for all tested tensor configurations. The results demonstrate that a multidimensional tensor can be mapped to a linear representation and reconstructed correctly using deterministic address-generation equations. This type of indexing is relevant to AI accelerator data movement, where feature maps may need to be flattened for buffering and processing before being reconstructed or interpreted in their original tensor layout.

The experiment also verified that the same indexing mechanism works across both small image tensors and larger multi-channel feature-map representations.
