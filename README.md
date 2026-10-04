
## Tensor Flattening and Reconstruction

---

## 1. Objective

The objective of this assignment is to implement and understand the mapping of a four-dimensional image/feature-map tensor in **BCHW (NCHW)** format into a flattened two-dimensional representation:

```text
B × C × H × W  →  B × (C × H × W)
```

and reconstruct it back:

```text
B × (C × H × W) → B × C × H × W
```

The implementation uses explicit loops, indexing, integer division, and modulo operations. Direct shape-conversion functions such as `reshape()`, `view()`, `flatten()`, and `ravel()` are not used in the required algorithm.

---

## 2. Repository Contents

```text
assign3_tensor.py
README.md
conclusion.md
```

### `assign3_tensor.py`
Main Python implementation containing:

- Manual BCHW → B×(CHW) flattening
- Manual B×(CHW) → BCHW reconstruction
- Reconstruction error calculation
- Actual MNIST experiment
- Actual CIFAR-10 experiment
- Synthetic multi-channel feature-map experiments
- Manual indexing demonstration

### `README.md`
Project setup, implementation details, execution instructions, and experimental summary.

### `conclusion.md`
Concise conclusion and observations from the experiments.

---

## 3. Tensor Index Mapping

For an element:

```text
I[b, c, h, w]
```

the corresponding position in the flattened tensor is:

```text
k = c × H × W + h × W + w
```

Therefore:

```text
F[b, k] = I[b, c, h, w]
```

For reconstruction, the original indices are recovered from `k`:

```text
c = k // (H × W)

remainder = k % (H × W)

h = remainder // W

w = remainder % W
```

and:

```text
I_hat[b, c, h, w] = F[b, k]
```

---

## 4. Data Used

The experiments use two real image datasets and synthetic feature maps.

### MNIST

An actual MNIST training sample was loaded using `torchvision.datasets.MNIST`.

Observed experiment:

```text
B = 1
C = 1
H = 28
W = 28
```

The sample used during the recorded run had label **5**.

### CIFAR-10

An actual CIFAR-10 training sample was loaded using `torchvision.datasets.CIFAR10`.

Observed experiment:

```text
B = 1
C = 3
H = 32
W = 32
```

The sample used during the recorded run had label **6**.

### Synthetic Feature Maps

Synthetic tensors were generated using `torch.rand()` to represent intermediate DNN feature maps with larger channel counts:

```text
C = 8, 16, 32, 64, 128, 256, 500
```

The assignment permits synthetic multi-channel tensors for testing larger feature-map channel dimensions.

---

## 5. Software Environment

The implementation was executed in a Python virtual environment.

Recorded package versions:

```text
Python environment: .venv
PyTorch:            2.14.0
torchvision:        0.29.0
```

The implementation was executed successfully on a Mac.

---

## 6. Installation

Activate the existing virtual environment:

```bash
source "/Users/cybertron/M Tech Sem-I/LNS Computation/.venv/bin/activate"
```

If PyTorch and torchvision are not installed:

```bash
pip install torch torchvision
```

In the recorded environment they were already installed.

---

## 7. Running the Program

From the directory containing the Python file:

```bash
python assign3_tensor.py
```

The first execution downloads MNIST and CIFAR-10 automatically.

The program then performs:

1. Manual indexing demonstration.
2. MNIST loading and conversion.
3. CIFAR-10 loading and conversion.
4. Synthetic feature-map tests.
5. Reconstruction-error calculation.
6. Final results-table display.

---

## 8. Assignment Compliance

The required flattening and reconstruction algorithms do **not** use:

```text
reshape()
view()
flatten()
ravel()
```

or equivalent library operations.

The actual layout conversion is implemented using:

```text
for loops
explicit tensor indexing
integer division
modulo
arithmetic index generation
```

PyTorch is used only for permitted operations such as dataset loading, tensor creation, shape inspection, and individual element access.

---

## 9. Manual Example

For:

```text
B = 1
C = 2
H = 2
W = 3
```

the tensor values were:

```text
Channel 0:
1  2  3
4  5  6

Channel 1:
7  8  9
10 11 12
```

The flattened representation is:

```text
1 2 3 4 5 6 7 8 9 10 11 12
```

For example:

```text
I[0,1,1,2]
```

maps to:

```text
k = 1×2×3 + 1×3 + 2
  = 11
```

Therefore:

```text
I[0,1,1,2] → F[0,11]
```

---

## 10. Experimental Results

| Dataset/Input | B | C | H | W | Emax | MAE |
|---|---:|---:|---:|---:|---:|---:|
| MNIST / actual sample | 1 | 1 | 28 | 28 | 0 | 0 |
| CIFAR-10 / actual sample | 1 | 3 | 32 | 32 | 0 | 0 |
| Synthetic fmap 1 | 1 | 8 | 16 | 16 | 0 | 0 |
| Synthetic fmap 2 | 1 | 16 | 16 | 16 | 0 | 0 |
| Synthetic fmap 3 | 1 | 32 | 16 | 16 | 0 | 0 |
| Synthetic fmap 4 | 1 | 64 | 16 | 16 | 0 | 0 |
| Synthetic fmap 5 | 1 | 128 | 8 | 8 | 0 | 0 |
| Synthetic fmap 6 | 1 | 256 | 8 | 8 | 0 | 0 |
| Synthetic fmap 7 | 1 | 500 | 4 | 4 | 0 | 0 |

All experiments produced:

```text
Maximum Absolute Error = 0.0000000000
Mean Absolute Error    = 0.0000000000
```

---

## 11. Error Metrics

Element-wise reconstruction error:

```text
E[b,c,h,w] = I[b,c,h,w] - I_hat[b,c,h,w]
```

Maximum absolute error:

```text
Emax = max |I[b,c,h,w] - I_hat[b,c,h,w]|
```

Mean absolute error:

```text
MAE = (1 / BCHW) × Σ |I[b,c,h,w] - I_hat[b,c,h,w]|
```

Because the conversion only changes the representation and does not modify values, the measured reconstruction error was zero in every experiment.

---

## 12. Expected Output

A successful run should show:

```text
MNIST:
Original shape     : [1, 1, 28, 28]
Flattened shape    : [1, 784]
Reconstructed shape: [1, 1, 28, 28]

CIFAR-10:
Original shape     : [1, 3, 32, 32]
Flattened shape    : [1, 3072]
Reconstructed shape: [1, 3, 32, 32]
```

and zero maximum absolute error and zero MAE for all tested configurations.

---

## 13. AI Accelerator Relevance

The experiment demonstrates explicit address generation for tensor data movement. A multidimensional feature map can be represented as a linear sequence for buffering and feeding processing elements, while the original BCHW coordinates can be recovered using deterministic arithmetic.

This makes the indexing equations useful for understanding address-generation logic in AI accelerator data paths.

---

## 14. Conclusion

The implementation successfully performs lossless BCHW-to-B×(CHW) flattening and the corresponding reconstruction using explicit indexing. The algorithm was validated using real MNIST and CIFAR-10 samples as well as synthetic feature maps with channel counts up to 500.

All experiments produced zero maximum absolute error and zero mean absolute error, confirming that the implemented address mapping and reverse mapping preserve every tensor element.

---

```
