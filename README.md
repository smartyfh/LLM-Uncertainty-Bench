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

## 1. Introduction
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


## 2. Uncertainty Quantification
We propose the utilization of [conformal prediction](https://arxiv.org/abs/2107.07511) for uncertainty quantification in LLMs. Compared to other methods, conformal prediction offers multiple advantages including ease of implementation, high efficiency, and a statistically **rigorous** estimation of uncertainty rather than a heuristic approximation.

<p align="center">
  <img src="images/diagram.png" width="90%" />
  <p align="center">The overall process of applying conformal prediction for uncertainty quantification in LLMs.</p>
</p>

### Evaluation Tasks and Datasets
In order to evaluate the performance of LLMs comprehensively, we consider five typical NLP tasks and prepare a dataset with **10,000** instances for each task.

* **Question Answering (QA):** QA is applied to evaluate an LLM's proficiency in utilizing its extensive world knowledge to provide accurate answers to a diverse range of questions. For this task, we construct the evaluation dataset based on [MMLU](https://arxiv.org/abs/2009.03300).
* **Reading Comprehension (RC):** RC is used for testing an LLM's ability to understand and analyze a given context, and answer questions based on the information presented in the context. For this task, we construct the evaluation dataset based on [CosmosQA](https://arxiv.org/abs/1909.00277).
* **Commonsense Inference (CI):** CI is leveraged to evaluate the ability of LLMs to understand and reason about the relationships between concepts and events based on commonsense and background knowledge. For this task, we construct the evaluation dataset based on [HellaSwag](https://arxiv.org/abs/1905.07830).
* **Dialogue Response Selection (DRS):** DRS is adopted for assessing the ability of LLMs to comprehend the meaning of a given dialogue and select an appropriate response from a set of possible responses. For this task, we construct the evaluation dataset based on [HaluEval](https://arxiv.org/abs/2305.11747).
* **Document Summarization (DS):** DS is taken to evaluate the proficiency of LLMs in comprehending the substance and context of a given document, and in producing a succinct and cohesive summary that effectively conveys the crucial information and main ideas of the document. For this task, we construct the evaluation dataset based on [HaluEval](https://arxiv.org/abs/2305.11747).

### Evaluated LLMs

## Citation

```bibtex
@article{ye2024llm_uq,
  title={Benchmarking LLMs via Uncertainty Quantification},
  author={Ye, Fanghua and Yang MingMing and Pang, Jianhui and Wang, Longyue and Wong, Derek F and Yilmaz Emine and Shi, Shuming and Tu, Zhaopeng},
  journal={arXiv preprint arXiv:2401.12794},
  year={2024}
  }
```
