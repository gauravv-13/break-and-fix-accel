| option | fresh randomness per image? | cycles, 1 image | cycles, 360 images | random bits, 1 image | random bits, 360 images | extra hardware / storage | clean acc |
|---|---|---|---|---|---|---|---|
| no defense | - | 920 (+0.0%) | 15,280 (+0.0%) | 0 | 0 | - | 96.94% |
| shuffle, one order per batch | no (same order for all images in a batch) | 920 (+0.0%) | 15,280 (+0.0%) | 604 | 604 | 544 bits permutation table | 96.94% |
| shuffle, new order per image | yes (weights reload per image) | 920 (+0.0%) | 331,200 (+2067.5%) | 604 | 217,440 | 544 bits permutation table | 96.94% |
| mask (2 multipliers per PE), masks per batch | no (same masks for all images in a batch) | 1,000 (+8.7%) | 15,360 (+0.5%) | 151,552 | 151,552 | +5632 bits in the array, 2 x (8x32) instead of 1 x (8x8); weight memory x2 (+18944 bits of shares) | 96.94% |
| mask (2 multipliers per PE), new masks per image | yes (weights reload per image) | 1,000 (+8.7%) | 360,000 (+2256.0%) | 151,552 | 54,558,720 | +5632 bits in the array, 2 x (8x32) instead of 1 x (8x8); weight memory x2 (+18944 bits of shares) | 96.94% |
| mask (1 widened multiplier, 2 passes), masks per batch | no (same masks for all images in a batch) | 1,920 (+108.7%) | 30,640 (+100.5%) | 151,552 | 151,552 | +24 bits weight register per PE (8 -> 32), 8x32 multiplier; weight memory x2 (+18944 bits of shares) | 96.94% |
| mask (1 widened multiplier, 2 passes), new masks per image | yes (weights reload per image) | 1,920 (+108.7%) | 691,200 (+4423.6%) | 151,552 | 54,558,720 | +24 bits weight register per PE (8 -> 32), 8x32 multiplier; weight memory x2 (+18944 bits of shares) | 96.94% |
| mask (2 multipliers per PE), in-PE refresh per image [formula] | yes (no reload; needs a refresh adder per share per PE) | 1,000 (+8.7%) | 15,360 (+0.5%) | 151,552 | 27,355,136 | +5632 bits in the array, 2 x (8x32) instead of 1 x (8x8); weight memory x2 (+18944 bits of shares); + refresh adders | 96.94% |

Shuffling random bits per inference: 604 (Fisher-Yates, one exact draw per step; minimum possible log2(n!) = 553.1). If the permutation is NOT generated in the background: +138 cycles per inference.
Masking: 64 random bits per weight per load (8 refresh + 24 upper-bit mask + 32 for Goubin), 2 cycles converter latency per tile (assumed).
