shared_zero_prompt = '''The following is a multiple-choice question with six potential answers. Only one of these options is correct. Please make your best effort and select the correct answer. You only need to output the option.\n\n'''

task_zero_prompt = '''
{"MMLU": "The following is a multiple-choice question about question answering. You should answer the question based on your world knowledge and problem solving ability. You only need to output the option.\n\n", 
"HellaSwag": "The following is a multiple-choice question about commonsense natural language inference. You are given a context and you should choose the most likely follow-up. You only need to output the option.\n\n",
"CosmosQA": "The following is a multiple-choice question about reading comprehension. You should answer the question based on the given context and you can use commonsense reasoning when necessary. You only need to output the option.\n\n",
"Halu-OpenDialKG": "The following is a multiple-choice question about dialogue response selection. You are given a dialogue history and you should select the best and correct response without hallucination and non-factual information. You only need to output the option.\n\n",
"Halu-CNN/DailyMail": "The following is a multiple-choice question about document summarization. You are given a document and you should select the best and correct summary without hallucination and non-factual information. You only need to output the option.\n\n"
}
'''

shared_few_prompt = '''Below are some examples of multiple-choice questions with six potential answers. For each question, only one option is correct.\n\n'''

task_few_prompt = '''
{"MMLU": "Below are some examples of multiple-choice questions about question answering. Each question should be answered based on your world knowledge and problem solving ability.\n\n", 
"HellaSwag": "Below are some examples of multiple-choice questions about commonsense natural language inference. For each question, there is a given context and the answer is the option that most likely follows the context.\n\n",
"CosmosQA": "Below are some examples of multiple-choice questions about reading comprehension. Each question should be answered based on the given context and commonsense reasoning when necessary.\n\n",
"Halu-OpenDialKG": "Below are some examples of multiple-choice questions about dialogue response selection. For each question, the answer is the option that represents the most suitable response for the given dialogue history, without hallucination and non-factual information.\n\n",
"Halu-CNN/DailyMail": "Below are some examples of multiple-choice questions about document summarization. For each question, the answer is the option that accurately summarizes the given document without hallucination and non-factual information.\n\n"
}
'''

base_cot_prompt = '''Please reason step-by-step and select the correct answer. You only need to output the option.\n\n'''

shared_cot_prompt = '''The following is a multiple-choice question with six potential answers. Only one of these options is correct. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n'''

task_cot_prompt = '''
{"MMLU": "The following is a multiple-choice question about question answering. You should answer the question based on your world knowledge and problem solving ability. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n", 
"HellaSwag": "The following is a multiple-choice question about commonsense natural language inference. You are given a context and you should choose the most likely follow-up. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n",
"CosmosQA": "The following is a multiple-choice question about reading comprehension. You should answer the question based on the given context and you can use commonsense reasoning when necessary. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n",
"Halu-OpenDialKG": "The following is a multiple-choice question about dialogue response selection. You are given a dialogue history and you should select the best and correct response without hallucination and non-factual information. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n",
"Halu-CNN/DailyMail": "The following is a multiple-choice question about document summarization. You are given a document and you should select the best and correct summary without hallucination and non-factual information. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n"
}
'''
