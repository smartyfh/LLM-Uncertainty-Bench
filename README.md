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


## 3. Evaluation Tasks and Datasets
In order to evaluate the performance of LLMs comprehensively, we consider five typical NLP tasks and prepare a dataset with **10,000** instances for each task.

* **Question Answering (QA):** QA is applied to evaluate an LLM's proficiency in utilizing its extensive world knowledge to provide accurate answers to a diverse range of questions. For this task, we construct the evaluation dataset based on [MMLU](https://arxiv.org/abs/2009.03300).
* **Reading Comprehension (RC):** RC is used for testing an LLM's ability to understand and analyze a given context, and answer questions based on the information presented in the context. For this task, we construct the evaluation dataset based on [CosmosQA](https://arxiv.org/abs/1909.00277).
* **Commonsense Inference (CI):** CI is leveraged to evaluate the ability of LLMs to understand and reason about the relationships between concepts and events based on commonsense and background knowledge. For this task, we construct the evaluation dataset based on [HellaSwag](https://arxiv.org/abs/1905.07830).
* **Dialogue Response Selection (DRS):** DRS is adopted for assessing the ability of LLMs to comprehend the meaning of a given dialogue and select an appropriate response from a set of possible responses. For this task, we construct the evaluation dataset based on [HaluEval](https://arxiv.org/abs/2305.11747).
* **Document Summarization (DS):** DS is taken to evaluate the proficiency of LLMs in comprehending the substance and context of a given document, and in producing a succinct and cohesive summary that effectively conveys the crucial information and main ideas of the document. For this task, we construct the evaluation dataset based on [HaluEval](https://arxiv.org/abs/2305.11747).

We formulate each task as a multiple-choice question answering (MCQA) task and the objective is to select the _only_ correct answer out of possible options.


## 4. Evaluation Results

### Pretrained Base LLMs

We first compare the performance of various LLMs in terms of prediction accuracy (**Acc**), which measures the proportion of test instances whose true label has the highest predicted probability.

| LLMs | QA | RC | CI | DRS | DS | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| Qwen-14B | 64.25 | 91.52 | 91.00 | 73.90 | 49.33 | 74.00 |
| Yi-6B | 57.57 | 85.99 | 76.50 | 58.72 | 66.06 | 68.97 |
| Mistral-7B | 60.44 | 81.94 | 62.93 | 53.21 | 62.16 | 64.14 |
| Llama-2-13B | 52.52 | 77.23 | 59.66 | 52.65 | 60.05 | 60.42 |
| Qwen-7B | 55.21 | 83.89 | 63.70 | 64.04 | 32.53 | 59.87 |
| InternLM-7B | 48.37 | 73.86 | 46.21 | 43.72 | 34.38 | 49.31 |
| Llama-2-7B | 45.60 | 65.79 | 43.05 | 32.61 | 45.60 | 46.53 |
| DeepSeek-7B | 45.65 | 65.39 | 42.66 | 33.50 | 42.15 | 45.87 |
| MPT-7B | 29.49 | 31.69 | 25.50 | 24.38 | 24.86 | 27.18 |
| Falcon-7B | 23.75 | 24.98 | 24.91 | 25.86 | 24.69 | 24.84 |

We then compare the performance of various LLMs in terms of prediction uncertainty, which is meaured as the average size of prediction sets of all test instances (**SS**). Note that a larger set size indicates higher uncertainty.

| LLMs | QA | RC | CI | DRS | DS | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| Qwen-14B | 2.80 | 1.74 | 2.02 | 1.94 | 2.37 | 2.17 |
| Yi-6B | 3.20 | 1.92 | 1.88 | 2.85 | 1.96 | 2.36 |
| Mistral-7B | 2.80 | 1.75 | 2.48 | 2.71 | 2.40 | 2.43 |
| Llama-2-13B | 3.06 | 2.24 | 2.72 | 2.55 | 2.24 | 2.56 |
| Qwen-7B | 3.26 | 2.15 | 2.28 | 2.51 | 2.92 | 2.63 |
| InternLM-7B | 3.49 | 2.19 | 3.28 | 3.63 | 4.47 | 3.41 |
| Llama-2-7B | 3.20 | 2.39 | 3.27 | 3.26 | 3.30 | 3.09 |
| DeepSeek-7B | 3.34 | 2.77 | 3.06 | 3.40 | 3.08 | 3.13 |
| MPT-7B | 3.53 | 3.46 | 3.60 | 3.59 | 3.66 | 3.57 |
| Falcon-7B | 3.90 | 3.60 | 3.66 | 3.64 | 3.92 | 3.75 |






## Citation

```bibtex
@article{ye2024llm_uq,
  title={Benchmarking LLMs via Uncertainty Quantification},
  author={Ye, Fanghua and Yang MingMing and Pang, Jianhui and Wang, Longyue and Wong, Derek F and Yilmaz Emine and Shi, Shuming and Tu, Zhaopeng},
  journal={arXiv preprint arXiv:2401.12794},
  year={2024}
  }
```
