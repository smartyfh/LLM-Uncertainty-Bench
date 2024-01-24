<div align="center">

# Benchmarking LLMs via Uncertainty Quantification
[Paper](https://arxiv.org/abs/2401.12794)

</div>

## Introduction
The proliferation of open-source Large Language Models (LLMs) from various institutions has highlighted the urgent need for comprehensive evaluation methods. However, current evaluation platforms, such as the widely recognized HuggingFace open LLM leaderboard, neglect a crucial aspect -- **uncertainty**, which is vital for thoroughly assessing LLMs. 

<p align="center">
  <img src="images/intro_exp.jpg" width="50%" />
  <p align="center">Two LLMs can achieve the same accuracy score but demonstrate different levels of uncertainty.</p>
</p>

To bridge this gap, we introduce a new benchmarking approach for LLMs that integrates uncertainty quantification. Our examination involves eight LLMs (LLM series) spanning five representative natural language processing tasks. Additionally, we introduce an uncertainty-aware evaluation metric, UAcc, which takes into account both prediction accuracy and prediction uncertainty. Our findings reveal that: 

* **LLMs with higher accuracy may exhibit lower certainty**;
* **Larger-scale LLMs may display greater uncertainty compared to their smaller counterparts**;
* **Instruction-finetuning tends to increase the uncertainty of LLMs**.
  
By taking uncertainty into account, our new UAcc metric can either amplify or diminish the relative improvement of one LLM over another and may even change the relative ranking of two LLMs, thus underscoring the significance of incorporating uncertainty in the evaluation of LLMs.

