## Introduction

This is from the official implementation: https://github.com/yueduan/DeepBinDiff/tree/master

## Requirements:

* tensorflow (2.0 > tensorflow version >= 1.14.0)
* gensim
* angr
* networkx
* lapjv
* r2pipe (API for Radare2)


## Usage in Bin2Summary

- Step 1: Use script to ananlyze the binaries and generate CFG & DFG data (need Radare2)
  ```
  python binary_analysis.py
  ```

- Step 2: If you want to enable the FSE module, use
  ```
  python gen_fse_emb.py --Filepath ../../data/
  ```

  If you want to disable the FSE module (with basic DeepBinDiff), use

  ```
  python gen_emb.py --Filepath ../../data/
  ```

## Citation

Please consider citing the paper:

```
@inproceedings{duan2020deepbindiff,
  title={Deepbindiff: Learning program-wide code representations for binary diffing},
  author={Duan, Yue and Li, Xuezixiang and Wang, Jinghan and Yin, Heng},
  booktitle={Network and distributed system security symposium},
  year={2020}
}
```
