
## Introduction

This is from the unofficial implementation: https://github.com/Lancern/asm2vec

## Requirements:

* python
* numpy


## Usage in Bin2Summary

- Step 1: Run Deepbindiff `binary_analysis.py` to prepare .instruction files (need Radare2 disassembler)
- Step 2: Use script to ananlyze the binaries and generate CFG & DFG data (very slow)
  ```
  python gen_emb.py [--FSE]
  ```



## Citation

Please consider citing the paper:

```
@inproceedings{ding2019asm2vec,
  title={Asm2vec: Boosting static representation robustness for binary clone search against code obfuscation and compiler optimization},
  author={Ding, Steven HH and Fung, Benjamin CM and Charland, Philippe},
  booktitle={2019 IEEE Symposium on Security and Privacy (SP)},
  pages={472--489},
  year={2019},
  organization={IEEE}
}
```