## Introduction

This is from the official implementation: https://github.com/palmtreemodel/PalmTree

## Requirements:
- cuda >= 10.1
- pytorch >= 1.3.1
- Binary Ninja

## Usage in Bin2Summary

- Step 1: Use script to ananlyze the binaries and generate CFG & DFG data (need binaryninja)
  ```
  python gen_asm.py [--FSE]
  ```

- Step 2: Use script to train assembly language model with CFG & DFG data
  ```
  [CUDA_VISIBLE_DEVICES=0] python train.py -d train
  ```

- Step 3: Use script to generate the function embeddings with the pre-trained assembly language model
  ```
  [CUDA_VISIBLE_DEVICES=0] python gen_emb.py [--FSE]
  ```



## Citation

Please consider citing the paper:

```
@inproceedings{li2021palmtree,
  title={Palmtree: learning an assembly language model for instruction embedding},
  author={Li, Xuezixiang and Qu, Yu and Yin, Heng},
  booktitle={Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security},
  pages={3236--3251},
  year={2021}
}
```
