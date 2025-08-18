# README 
This is the official implementation of TimeCapsule presented in the paper "TimeCapsule: Solving the Jigsaw Puzzle of Long-Term Time Series
Forecasting with Compressed Predictive Representations" <https://arxiv.org/abs/2504.12721>.<img width="36" height="50" alt="Dream_Timer_Ball_Sprite" src="https://github.com/user-attachments/assets/faaac99e-eba9-433e-a2af-aed6b648bcf4" />


🎉🎉**Our work has been accepted by KDD 2025.**
## Introduction
TL;DR
![image](https://github.com/user-attachments/assets/1d45428d-f14d-4674-8883-4a3c15f06756)

## Get Started
1. Please download the dataset you need at [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr).

2. Note the position you place the dataset package and modify the path in `scripts` files accordingly.

3. Change to the correct path, then use bash to run the specific script file, e.g., `bash ./scripts/traffic.sh`.

## Acknowledgment
> 1. We recognize that this is an imperfect work. Please let us know if you have any questions or suggestions about the code or specific techniques in the paper.
> 2. It is recommended to **increase the length of compression parameters** when using ***short look-back windows***, other hyperparameters should also be adjusted accordingly.  
> 3. Please consider starring this repository and citing our research if you find the paper interesting or this repo helpful.

```bibtex
@article{lu2025timecapsule,
  title={TimeCapsule: Solving the Jigsaw Puzzle of Long-Term Time Series Forecasting with Compressed Predictive Representations},
  author={Lu, Yihang and Xu, Yangyang and Qing, Qitao and Meng, Xianwei},
  journal={arXiv preprint arXiv:2504.12721},
  year={2025}
}
```
We appreciate the following repos for their valuable code and efforts.
- PatchTST
- iTransformer
