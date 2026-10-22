# Why Correlation Power Analysis (CPA) Works on AES-128

## 1. The Core Idea in Simple Words

A full 128-bit AES key has $2^{128}$ possible values—far too many to guess all at once. However, during the very first round of AES-128, the chip processes the 16 bytes of the key **one byte at a time**:

$$v_i = \text{S-box}(p_i \oplus k_i) \quad \text{for byte } i \in \{0, 1, \dots, 15\}$$

Here:
- $p_i$ is a known plaintext byte (0 to 255),
- $k_i$ is one secret key byte (only 256 possible values!),
- $\text{S-box}(\cdot)$ is a fixed, publicly known lookup table.

Because CMOS transistors draw a tiny burst of current whenever wires flip between `0` and `1`, the instantaneous power consumed when storing or computing $v_i$ is roughly proportional to its **Hamming weight** $\text{HW}(v_i)$—the number of `1` bits in that byte (between 0 and 8, with an average of 4.0)—plus random electronic noise $\mathcal{N}(0, \sigma^2)$:

$$L_i = \text{HW}(\text{S-box}(p_i \oplus k_i)) + \mathcal{N}(0, \sigma^2)$$

## 2. Divide-and-Conquer with Pearson Correlation

Instead of guessing all 16 bytes together ($2^{128}$ possibilities), CPA attacks each byte independently ($16 \times 256 = 4{,}096$ total guesses):

1. Capture $N$ power traces using $N$ known random plaintexts.
2. For byte $i$ and a candidate guess $g \in \{0, \dots, 255\}$, compute the predicted Hamming weight $H_{n,g} = \text{HW}(\text{S-box}(p_{n,i} \oplus g))$ for every trace $n$.
3. Compute the **Pearson correlation coefficient** $r$ between the prediction vector $H_{\cdot, g}$ and the measured power trace $T_{\cdot, t}$.

Why does only the right guess stand out?
- **Correct guess ($g = k_i$):** The predicted Hamming weights match the true hardware activity on every single trace. The only difference is random Gaussian noise, which averages out as $N$ grows. Theoretically, the expected correlation is:
  $$r_{\text{true}} \approx \frac{1}{\sqrt{1 + \sigma^2 / \text{Var}(\text{HW})}} = \frac{1}{\sqrt{1 + \sigma^2 / 2}}$$
  since $\text{Var}(\text{HW}) = 8 \times 0.25 = 2.0$ for a uniform random byte.
- **Wrong guess ($g \neq k_i$):** The AES S-box is specifically designed to be highly non-linear. Even if $g$ differs from $k_i$ by just 1 bit, $\text{S-box}(p_i \oplus g)$ jumps to a completely unrelated byte across different plaintexts $p_i$. Its correlation with the real power trace stays near $0$ (within random sampling noise of roughly $\pm 1/\sqrt{N}$).

## 3. Key Numbers from Our Experiments

Using the FIPS-197 cipher key `2b7e151628aed2a6abf7158809cf4f3c` and fixed seed `20261022`:

### Single-Byte Demonstration (Byte 0, $k_0 = \text{0x2b}$, $N = 100$ traces, $\sigma = 1.0$)
- **True key guess (`0x2b`):** $|r| = 0.8454$ (Rank #1 out of 256; theoretical expectation $1/\sqrt{1 + 1/2} = 0.8165$).
- **Highest wrong guess (out of 255 incorrect values):** $|r| = 0.4164$.
- **All 16 bytes:** 16/16 bytes recovered with Rank #1, and the correlation across the 24-sample time series peaks sharply at the exact leaking time index ($t = 4$ for Byte 0) while staying flat across the 8 non-leaking padding samples ($t \in \{0..3, 20..23\}$).

### Traces Required for $\ge 90\%$ Full-Key Success (All 16 Bytes Correct, 20 Runs per $\sigma$)

| Noise Std. Dev. ($\sigma$) | Theoretical SNR ($2/\sigma^2$) | Expected True Correlation ($r_{\text{true}}$) | Traces Needed for $\ge 90\%$ Full-Key Recovery |
|---:|---:|---:|---:|
| **0.5** | 8.000 | 0.943 | **15 traces** |
| **1.0** | 2.000 | 0.816 | **30 traces** |
| **2.0** | 0.500 | 0.577 | **80 traces** |
| **4.0** | 0.125 | 0.333 | **275 traces** |

Each doubling of the noise standard deviation $\sigma$ reduces the Signal-to-Noise Ratio (SNR) by $4\times$, requiring roughly $3\times\text{–}3.5\times$ more traces to recover all 16 key bytes with $\ge 90\%$ reliability.
