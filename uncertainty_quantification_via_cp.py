import pickle 
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

options = ["A", "B", "C", "D", "E", "F"]
ids_to_remove = [1, 3, 5, 7, 9] # remove data points that have been used as demonstration data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_raw_data(raw_data_dir, data_name, cal_ratio):
    """
    Get raw data from the json file and split it into a calibration set and a test set.
    """
    raw_data = json.load(open(os.path.join(raw_data_dir, data_name+".json"), "r"))
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in ids_to_remove]
    cal_raw_data, test_raw_data = train_test_split(raw_data, train_size=cal_ratio, random_state=42)
    print(len(raw_data), len(cal_raw_data), len(test_raw_data))
    return cal_raw_data, test_raw_data

def get_logits_data(model_name, data_name, cal_raw_data, test_raw_data, 
                    logits_data_dir, cal_ratio, prompt_methods, icl_methods):
    """
    Get logit scores of data instances and split these scores into a calibration set and a test set accordingly.
    """
    logits_data_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            logits_file = os.path.join(logits_data_dir, model_name+"_"+data_name+"_"+m+"_"+fs+".pkl")
            with open(logits_file, 'rb') as f:
                logits_data = pickle.load(f)
            logits_data = [item for idx, item in enumerate(logits_data) if idx not in ids_to_remove]
            cal_logits_data, test_logits_data = train_test_split(logits_data, train_size=cal_ratio, random_state=42)
            assert len(cal_logits_data) == len(cal_raw_data)
            assert len(test_logits_data) == len(test_raw_data)
            logits_data_all[m+"_"+fs] = {}
            logits_data_all[m+"_"+fs]["cal"] = cal_logits_data
            logits_data_all[m+"_"+fs]["test"] = test_logits_data
    return logits_data_all

def LAC_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    """
    Apply conformal prediction to obtain sets of predicted answers on each instance based on its softmax scores.
    Here the LAC score function is utilized.
    """
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m+"_"+fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m+"_"+fs]["cal"]
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"])
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_scores.append(1 - probs[options.index(truth_answer)])
            # calculate the threshold qhat
            n = len(cal_logits_data)
            q_level = np.ceil((n+1) * (1-alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method='higher')
            # print(f"{m}_{fs} quantile: {qhat}")
            # generate prediction sets
            pred_sets = {}
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"])
                ps = []
                for ii, p in enumerate(probs):
                    # 1 - p <= qhat, so p >= 1- qhat
                    if p >= 1 - qhat:
                        ps.append(options[ii])
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m+"_"+fs] = pred_sets
    return pred_sets_all

def APS_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    """
    Apply conformal prediction to obtain sets of predicted answers on each instance based on its softmax scores.
    Here the APS score function is utilized.
    """
    ada_pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            ada_pred_sets_all[m+"_"+fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m+"_"+fs]["cal"]
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"])
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_pi = np.argsort(probs)[::-1] # descending order
                cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                cal_sum_r = np.take_along_axis(cal_sum, cal_pi.argsort(), axis=0)
                cal_score = cal_sum_r[options.index(truth_answer)]
                cal_scores.append(cal_score)
            # calculate the threshold qhat
            n = len(cal_logits_data)
            q_level = np.ceil((n+1) * (1-alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method='higher')
            # print(f"{m}_{fs} quantile: {qhat}")
            # generate prediction sets
            pred_sets = {}
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"])
                cal_pi = np.argsort(probs)[::-1] # descending order
                cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                ps = []
                ii = 0
                while ii < len(cal_sum) and cal_sum[ii] <= qhat:
                    op_id = cal_pi[ii]
                    ps.append(options[op_id])
                    ii += 1
                if len(ps) == 0:
                    op_id = cal_pi[ii]
                    ps.append(options[op_id])
                # cal_sum_r = np.take_along_axis(cal_sum <= qhat, cal_pi.argsort(), axis=0)
                # ps = []
                # for ii, p in enumerate(list(cal_sum_r)):
                #     if p:
                #         ps.append(options[ii])
                pred_sets[str(row["id"])] = ps
            ada_pred_sets_all[m+"_"+fs] = pred_sets
    return ada_pred_sets_all

def get_accuracy(logits_data, raw_data):
    res = []
    preds = []
    for idx, row in enumerate(raw_data):
        truth_answer = row["answer"]
        pred = logits_data[idx]
        assert pred["id"] == row["id"]
        pred_answer = options[np.argmax(pred["logits_options"])]
        preds.append(pred_answer)
        if pred_answer == truth_answer:
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res), preds

def cal_acc(logits_data_all, test_raw_data, prompt_methods, icl_methods):
    results_acc = {}
    E_ratios = {}
    F_ratios = {}
    for m in prompt_methods:
        for fs in icl_methods:
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            acc, preds = get_accuracy(test_logits_data, test_raw_data)
            results_acc[m+"_"+fs] = acc
            counts = Counter(preds)
            E_ratio = counts["E"] / len(preds)
            F_ratio = counts["F"] / len(preds)
            E_ratios[m+"_"+fs] = E_ratio
            F_ratios[m+"_"+fs] = F_ratio
    return results_acc, E_ratios, F_ratios

def convert_id_to_ans(test_raw_data):
    test_id_to_answer = {}
    for row in test_raw_data:
        test_id_to_answer[str(row["id"])] = row["answer"]
    return test_id_to_answer

def cal_coverage(pred_sets_all, test_id_to_answer, prompt_methods, icl_methods):
    """
    Calculate the coverage rate of prediction sets.
    """""
    coverage_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            cover = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                if test_id_to_answer[k] in v:
                    cover.append(1)
                else:
                    cover.append(0)
            coverage_all[m+"_"+fs] = sum(cover) / len(cover)
    return coverage_all

def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    set_sizes = {}
    for m in prompt_methods:
        for fs in icl_methods:
            sz = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                sz.append(len(v))
            # print(f"{m}_{fs}: {min(sz)}, {max(sz)}")
            # average set size
            set_sizes[m+"_"+fs] = sum(sz) / len(sz)
    return set_sizes

def cal_uacc(results_acc, set_sizes):
    results_uacc = {}
    for k, v in results_acc.items():
        results_uacc[k] = v * np.sqrt(len(options)) / set_sizes[k]
    return results_uacc

def apply_conformal_prediction(args):
    all_data_results = {}
    for data_name in args.data_names:
        cal_raw_data, test_raw_data = get_raw_data(args.raw_data_dir, data_name, args.cal_ratio)
        logits_data_all = get_logits_data(args.model, data_name, cal_raw_data, test_raw_data, 
                                          args.logits_data_dir, args.cal_ratio,
                                          args.prompt_methods, args.icl_methods)
        results_acc, E_ratios, F_ratios = cal_acc(logits_data_all, test_raw_data,
                                                  args.prompt_methods, args.icl_methods)
        test_id_to_answer = convert_id_to_ans(test_raw_data)
        # cp method LAC
        pred_sets_all_LAC = LAC_CP(logits_data_all, cal_raw_data,
                                   args.prompt_methods, args.icl_methods,
                                   alpha=args.alpha)
        coverage_all_LAC = cal_coverage(pred_sets_all_LAC, test_id_to_answer,
                                        args.prompt_methods, args.icl_methods)
        set_sizes_LAC = cal_set_size(pred_sets_all_LAC, args.prompt_methods, args.icl_methods)
        results_uacc_LAC = cal_uacc(results_acc, set_sizes_LAC)
        # cp method APS
        pred_sets_all_APS = APS_CP(logits_data_all, cal_raw_data,
                                   args.prompt_methods, args.icl_methods,
                                   alpha=args.alpha)
        coverage_all_APS = cal_coverage(pred_sets_all_APS, test_id_to_answer,
                                        args.prompt_methods, args.icl_methods)
        set_sizes_APS = cal_set_size(pred_sets_all_APS, args.prompt_methods, args.icl_methods)
        results_uacc_APS = cal_uacc(results_acc, set_sizes_APS)

        all_data_results[data_name] = {}
        all_data_results[data_name]["Acc"] = results_acc
        all_data_results[data_name]["E_rate"] = E_ratios
        all_data_results[data_name]["F_rate"] = F_ratios
        all_data_results[data_name]["LAC_set_size"] = set_sizes_LAC
        all_data_results[data_name]["APS_set_size"] = set_sizes_APS
        all_data_results[data_name]["LAC_coverage"] = coverage_all_LAC
        all_data_results[data_name]["APS_coverage"] = coverage_all_APS
        all_data_results[data_name]["UAcc_LAC"] = results_uacc_LAC
        all_data_results[data_name]["UAcc_APS"] = results_uacc_APS
    
    return all_data_results

def main(args):
    all_data_results = apply_conformal_prediction(args)

    # calculate the average results of the two conformal prediction methods and the three prompting strategies
    acc = []
    for data_name in args.data_names:
        acc.append(100 * np.mean(list(all_data_results[data_name]["Acc"].values())))
        print(f"{data_name}_Acc: {acc[-1]:.2f}")
    print(f"Average acc: {np.mean(acc):.2f}")

    LAC_set_size, APS_set_size = [], []
    LAC_coverage, APS_coverage = [], []
    UAcc_LAC, UAcc_APS = [], []
    for data_name in args.data_names:
        # average set size
        LAC_set_size.append(np.mean(list(all_data_results[data_name]["LAC_set_size"].values())))
        APS_set_size.append(np.mean(list(all_data_results[data_name]["APS_set_size"].values())))
        # coverage rate
        LAC_coverage.append(100 * np.mean(list(all_data_results[data_name]["LAC_coverage"].values())))
        APS_coverage.append(100 * np.mean(list(all_data_results[data_name]["APS_coverage"].values())))
        # UAcc
        UAcc_LAC.append(100 * np.mean(list(all_data_results[data_name]["UAcc_LAC"].values())))
        UAcc_APS.append(100 * np.mean(list(all_data_results[data_name]["UAcc_APS"].values())))

    pred_set_size = []
    for sz1, sz2 in zip(LAC_set_size, APS_set_size):
        pred_set_size.append((sz1 + sz2) / 2)
    for idx, data_name in enumerate(args.data_names):
        print(f"{data_name}_SS: {pred_set_size[idx]:.2f}")
    print(f"Average SS: {np.mean(pred_set_size):.2f}")

    pred_coverage = []
    for cr1, cr2 in zip(LAC_coverage, APS_coverage):
        pred_coverage.append((cr1 + cr2) / 2)
    for idx, data_name in enumerate(args.data_names):
        print(f"{data_name}_Coverage Rate: {pred_coverage[idx]:.2f}")
    print(f"Average Coverage Rate: {np.mean(pred_coverage):.2f}")

    pred_uacc = []
    for ua1, ua2 in zip(UAcc_LAC, UAcc_APS):
        pred_uacc.append((ua1 + ua2) / 2)
    for idx, data_name in enumerate(args.data_names):
        print(f"{data_name}_UAcc: {pred_uacc[idx]:.2f}")
    print(f"Average UAcc: {np.mean(pred_uacc):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--raw_data_dir", type=str, default="data",
                        help="Directory where raw data are stored.")
    parser.add_argument("--logits_data_dir", type=str, default="outputs",
                        help="Directory where logits data are stored.")
    parser.add_argument("--data_names", nargs='*', 
                        default=['mmlu_10k', 'cosmosqa_10k', 'hellaswag_10k', 'halu_dialogue', 'halu_summarization'], 
                        help='List of datasets to be evaluated. If empty, all datasets are evaluated.')
    parser.add_argument("--prompt_methods", nargs='*', 
                        default=['base', 'shared', 'task'], 
                        help='List of prompting methods. If empty, all methods are evaluated.')
    parser.add_argument("--icl_methods", nargs='*', 
                        default=['icl1'], 
                        help='Select from icl1, icl0, icl0_cot.')
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                        help="The ratio of data to be used as the calibration data.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="The error rate parameter.")
    args = parser.parse_args()

    main(args)
