<div align="center">

# Benchmarking LLMs via Uncertainty Quantification

![Question Answering](https://img.shields.io/badge/Task-Question_Answering-red) 
![RC](https://img.shields.io/badge/Task-Reading_Comprehension-red) 
![CI](https://img.shields.io/badge/Task-Commonsense_Inference-red) 
![DRS](https://img.shields.io/badge/Task-Dialogue_Response_Selection-red)
![DS](https://img.shields.io/badge/Task-Document_Summarization-red)  
![Llama-2](https://img.shields.io/badge/Model-Llama--2-21C2A4) 
![Mistral](https://img.shields.io/badge/Model-Mistral-21C2A4) 
![Falcon](https://img.shields.io/badge/Model-Falcon-21C2A4) 
![MPT](https://img.shields.io/badge/Model-MPT-21C2A4)
![Yi](https://img.shields.io/badge/Model-Yi-21C2A4)
![Qwen](https://img.shields.io/badge/Model-Qwen-21C2A4)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek-21C2A4)
![InternLM](https://img.shields.io/badge/Model-InternLM-21C2A4)

[Paper](https://arxiv.org/abs/2401.12794)

</div>

## Introduction
The proliferation of open-source Large Language Models (LLMs) from various institutions has highlighted the urgent need for comprehensive evaluation methods. However, current evaluation platforms, such as the widely recognized HuggingFace open LLM leaderboard, neglect a crucial aspect -- **uncertainty**, which is vital for thoroughly assessing LLMs. 

<p align="center">
  <img src="images/intro_exp.jpg" width="45%" />
  <p align="center">Two LLMs can achieve the same accuracy score but demonstrate different levels of uncertainty.</p>
</p>

To bridge this gap, we introduce a new benchmarking approach for LLMs that integrates uncertainty quantification. Our examination involves eight LLMs (LLM series) spanning five representative natural language processing tasks. Additionally, we introduce an uncertainty-aware evaluation metric, UAcc, which takes into account both prediction accuracy and prediction uncertainty. Our findings reveal that: 

* **LLMs with higher accuracy may exhibit lower certainty**;
* **Larger-scale LLMs may display greater uncertainty compared to their smaller counterparts**;
* **Instruction-finetuning tends to increase the uncertainty of LLMs**.
  
By taking uncertainty into account, our new UAcc metric can either amplify or diminish the relative improvement of one LLM over another and may even change the relative ranking of two LLMs, thus underscoring the significance of incorporating uncertainty in the evaluation of LLMs.


## Uncertainty Quantification
We propose the utilization of [conformal prediction](https://arxiv.org/abs/2107.07511) for uncertainty quantification in LLMs. Compared to other methods, conformal prediction offers multiple advantages including ease of implementation, high efficiency, and a statistically rigorous estimation of uncertainty rather than a heuristic approximation.

<p align="center">
  <img src="images/diagram.png" width="85%" />
  <p align="center">The overall process of applying conformal prediction for uncertainty quantification in LLMs.</p>
</p>


## Citation

```bibtex
@article{ye2024llm_uq,
  title={Benchmarking LLMs via Uncertainty Quantification},
  author={Ye, Fanghua and Yang MingMing and Pang, Jianhui and Wang, Longyue and Wong, Derek F and Yilmaz Emine and Shi, Shuming and Tu, Zhaopeng},
  journal={arXiv preprint arXiv:2401.12794},
  year={2024}
  }
```
