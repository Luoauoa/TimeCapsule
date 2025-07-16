# README
This is the official implementation of TimeCapsule presented in the paper "TimeCapsule: Solving the Jigsaw Puzzle of Long-Term Time Series
Forecasting with Compressed Predictive Representations" <https://arxiv.org/abs/2504.12721>.

🎉🎉**We are delighted to hear that our work has been accepted by KDD 2025**
## Introduction
TL;DR
![image](https://github.com/user-attachments/assets/1d45428d-f14d-4674-8883-4a3c15f06756)

From the overview of its architecture, it is likely to be a generative model (May also be applicable for time series generation). This model can be interpreted in multiple ways. To understand one of the motivations and ideas for developing such an architecture, we paraphrase parts of our rebuttal in the conference as a detailed explanation here:

>The core components of TimeCapsule—3D tensor modeling, encoder compression, and decoder recovery—are theoretically grounded in (inspired by) modern lossy compression techniques (e.g., bits-back coding) and low-rank tensor recovery principles (related to matrix recovery and compressive sensing). The theoretical foundations for these design choices can be found in the cited references. Specifically, achieving a forecaster with satisfactory generality requires an implicit self-supervised learning strategy capable of efficiently and effectively recognizing the specific properties of each time series before making predictions. Therefore, we chose this compression and recovery framework, rooted in specific theorems. Through this framework, we hope the neural network not only purifies redundant information but also learns to decompose, analyze, and recreate the underlying mechanisms of different time series (as well as improving the capability of long-range information utilization).

>To implement this, we need an encoder-decoder structure. It is true that Transformer- and MLP-based structures are well-explored in this domain. However, as explained in the manuscript (Section 1, lines 123-141), previous works have often relied heavily on one or the other, and their definitive roles in Long-Term Time Series Forecasting (LTSF) have not yet been fully established. From this perspective, we developed an asymmetric structure to leverage both the representation learning capability and scalability of Transformers and the forecasting modeling (data-dependent basis mapping) functionality of MLPs. Meanwhile, this asymmetry helps achieve flexibility in managing computational cost, as analyzed in Section 5.3.

>Then, to align with and better serve the LTSF task, we found it necessary to inject specific time series modeling methods and strategies into this learning framework. Thus, we revisited and organized core ideas from recent works, abstracting concepts such as multi-level processing (encompassing time series decomposition and multi-scale analysis), various attention mechanisms, and redundancy reduction, as illustrated in Section 2. To streamline these techniques in our framework, we use mode production to implement the so-called "MoMSA". It should be noted that these techniques have demonstrated effectiveness in time series modeling and are motivated by practical utility. In addition, we introduced JEPA for two main reasons: first, to help monitor and learn the compressed predictive representation; second, and very importantly (as explained in lines 483-493), we expect the inner prediction facilitated by JEPA can help the model inherently adapt to the non-stationarity often present in time series, a crucial challenge in practical LTSF. The effects of this are evaluated in the experiments section. To avoid complexity from additional gradient flow and reduce training difficulty, Exponential Moving Average (EMA) is used to update the Y-encoder parameters (as in I-JEPA and V-JEPA). Its role is to help generate an alignment loss for intermediate representations and can be seen as a form of pre-training that provides an auxiliary supervision task for learning predictive representations...

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
