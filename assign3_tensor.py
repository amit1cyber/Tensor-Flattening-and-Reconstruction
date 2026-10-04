import torch
from torchvision import datasets, transforms


# ============================================================
# 1. BCHW -> B x (CHW)
#    Manual flattening
# ============================================================

def flatten_bchw(x):
    """
    Convert:
        B x C x H x W
    into:
        B x (C*H*W)

    IMPORTANT:
    No reshape(), view(), flatten(), ravel(), etc.
    are used here.
    """

    B, C, H, W = x.shape

    flat_size = C * H * W

    flat = torch.empty(
        B,
        flat_size,
        dtype=x.dtype
    )

    for b in range(B):

        for c in range(C):

            for h in range(H):

                for w in range(W):

                    # BCHW -> linear index
                    k = c * H * W + h * W + w

                    flat[b, k] = x[b, c, h, w]

    return flat


# ============================================================
# 2. B x (CHW) -> BCHW
#    Manual reconstruction
# ============================================================

def reconstruct_bchw(flat, B, C, H, W):
    """
    Convert:
        B x (C*H*W)
    into:
        B x C x H x W

    IMPORTANT:
    No reshape(), view(), flatten(), ravel(), etc.
    are used here.
    """

    reconstructed = torch.empty(
        B,
        C,
        H,
        W,
        dtype=flat.dtype
    )

    for b in range(B):

        for k in range(C * H * W):

            # Recover channel
            c = k // (H * W)

            # Remaining index after removing channel
            remainder = k % (H * W)

            # Recover height
            h = remainder // W

            # Recover width
            w = remainder % W

            reconstructed[b, c, h, w] = flat[b, k]

    return reconstructed


# ============================================================
# 3. Error calculation
# ============================================================

def calculate_error(original, reconstructed):

    B, C, H, W = original.shape

    max_error = 0.0
    total_error = 0.0
    total_elements = 0

    for b in range(B):

        for c in range(C):

            for h in range(H):

                for w in range(W):

                    error = abs(
                        original[b, c, h, w].item()
                        - reconstructed[b, c, h, w].item()
                    )

                    if error > max_error:
                        max_error = error

                    total_error += error
                    total_elements += 1

    mae = total_error / total_elements

    return max_error, mae


# ============================================================
# 4. Run one experiment
# ============================================================

def run_experiment(name, tensor):

    B, C, H, W = tensor.shape

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"B = {B}")
    print(f"C = {C}")
    print(f"H = {H}")
    print(f"W = {W}")

    print(f"Original shape    : {tensor.shape}")

    # --------------------------------------------------------
    # BCHW -> B x (CHW)
    # --------------------------------------------------------

    flattened = flatten_bchw(tensor)

    print(f"Flattened shape   : {flattened.shape}")

    # --------------------------------------------------------
    # B x (CHW) -> BCHW
    # --------------------------------------------------------

    reconstructed = reconstruct_bchw(
        flattened,
        B,
        C,
        H,
        W
    )

    print(f"Reconstructed shape: {reconstructed.shape}")

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    max_error, mae = calculate_error(
        tensor,
        reconstructed
    )

    print(f"Maximum absolute error : {max_error:.10f}")
    print(f"Mean absolute error    : {mae:.10f}")

    return (
        name,
        B,
        C,
        H,
        W,
        max_error,
        mae
    )


# ============================================================
# 5. Load actual MNIST sample
# ============================================================

def load_mnist():

    print("\nDownloading/loading MNIST...")

    transform = transforms.ToTensor()

    mnist = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    image, label = mnist[0]

    # MNIST image is already:
    # C x H x W = 1 x 28 x 28
    #
    # We need BCHW.
    #
    # Instead of using reshape/view,
    # explicitly create B dimension.

    B = 1
    C = image.shape[0]
    H = image.shape[1]
    W = image.shape[2]

    tensor = torch.empty(
        B,
        C,
        H,
        W,
        dtype=image.dtype
    )

    for c in range(C):

        for h in range(H):

            for w in range(W):

                tensor[0, c, h, w] = image[c, h, w]

    print(f"MNIST label = {label}")

    return tensor


# ============================================================
# 6. Load actual CIFAR-10 sample
# ============================================================

def load_cifar10():

    print("\nDownloading/loading CIFAR-10...")

    transform = transforms.ToTensor()

    cifar10 = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    image, label = cifar10[0]

    # CIFAR-10:
    # C x H x W = 3 x 32 x 32
    #
    # Explicitly add batch dimension.

    B = 1
    C = image.shape[0]
    H = image.shape[1]
    W = image.shape[2]

    tensor = torch.empty(
        B,
        C,
        H,
        W,
        dtype=image.dtype
    )

    for c in range(C):

        for h in range(H):

            for w in range(W):

                tensor[0, c, h, w] = image[c, h, w]

    print(f"CIFAR-10 label = {label}")

    return tensor


# ============================================================
# 7. Generate synthetic feature map
# ============================================================

def generate_synthetic_feature_map(B, C, H, W):

    # Random tensor representing an intermediate
    # DNN feature map.

    tensor = torch.rand(B, C, H, W)

    return tensor


# ============================================================
# 8. Manual indexing demonstration
# ============================================================

def manual_indexing_example():

    print("\n")
    print("=" * 60)
    print("MANUAL INDEXING EXAMPLE")
    print("=" * 60)

    B = 1
    C = 2
    H = 2
    W = 3

    # Create tensor
    tensor = torch.empty(B, C, H, W)

    value = 1

    for b in range(B):

        for c in range(C):

            for h in range(H):

                for w in range(W):

                    tensor[b, c, h, w] = value

                    value += 1

    print("\nOriginal BCHW tensor:")
    print(tensor)

    # Flatten
    flat = flatten_bchw(tensor)

    print("\nFlattened B x (CHW) tensor:")
    print(flat)

    print("\nMapping:")

    for c in range(C):

        for h in range(H):

            for w in range(W):

                k = c * H * W + h * W + w

                print(
                    f"I[0,{c},{h},{w}]"
                    f" -> F[0,{k}]"
                    f" = {tensor[0,c,h,w].item():.0f}"
                )


# ============================================================
# 9. Main program
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Manual indexing example required by assignment
    # --------------------------------------------------------

    manual_indexing_example()

    results = []

    # --------------------------------------------------------
    # Experiment 1: ACTUAL MNIST
    # B=1, C=1, H=28, W=28
    # --------------------------------------------------------

    mnist_tensor = load_mnist()

    result = run_experiment(
        "MNIST / actual sample",
        mnist_tensor
    )

    results.append(result)

    # --------------------------------------------------------
    # Experiment 2: ACTUAL CIFAR-10
    # B=1, C=3, H=32, W=32
    # --------------------------------------------------------

    cifar_tensor = load_cifar10()

    result = run_experiment(
        "CIFAR-10 / actual sample",
        cifar_tensor
    )

    results.append(result)

    # --------------------------------------------------------
    # Synthetic feature maps
    # --------------------------------------------------------

    synthetic_configs = [

        # name, B, C, H, W
        ("Synthetic fmap 1", 1, 8,   16, 16),
        ("Synthetic fmap 2", 1, 16,  16, 16),
        ("Synthetic fmap 3", 1, 32,  16, 16),
        ("Synthetic fmap 4", 1, 64,  16, 16),
        ("Synthetic fmap 5", 1, 128, 8,  8),
        ("Synthetic fmap 6", 1, 256, 8,  8),
        ("Synthetic fmap 7", 1, 500, 4,  4),
    ]

    for name, B, C, H, W in synthetic_configs:

        tensor = generate_synthetic_feature_map(
            B, C, H, W
        )

        result = run_experiment(
            name,
            tensor
        )

        results.append(result)

    # --------------------------------------------------------
    # Final results table
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 100)
    print("FINAL EXPERIMENTAL RESULTS")
    print("=" * 100)

    print(
        f"{'Dataset/Input':25s}"
        f"{'B':>5s}"
        f"{'C':>6s}"
        f"{'H':>6s}"
        f"{'W':>6s}"
        f"{'Emax':>18s}"
        f"{'MAE':>18s}"
    )

    print("-" * 100)

    for result in results:

        name, B, C, H, W, max_error, mae = result

        print(
            f"{name:25s}"
            f"{B:5d}"
            f"{C:6d}"
            f"{H:6d}"
            f"{W:6d}"
            f"{max_error:18.10f}"
            f"{mae:18.10f}"
        )