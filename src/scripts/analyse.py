import os
import sys

languages = [
    "TGLANG_LANGUAGE_OTHER",
    "TGLANG_LANGUAGE_C",
    "TGLANG_LANGUAGE_CPLUSPLUS",
    "TGLANG_LANGUAGE_CSHARP",
    "TGLANG_LANGUAGE_CSS",
    "TGLANG_LANGUAGE_DART",
    "TGLANG_LANGUAGE_DOCKER",
    "TGLANG_LANGUAGE_FUNC",
    "TGLANG_LANGUAGE_GO",
    "TGLANG_LANGUAGE_HTML",
    "TGLANG_LANGUAGE_JAVA",
    "TGLANG_LANGUAGE_JAVASCRIPT",
    "TGLANG_LANGUAGE_JSON",
    "TGLANG_LANGUAGE_KOTLIN",
    "TGLANG_LANGUAGE_LUA",
    "TGLANG_LANGUAGE_NGINX",
    "TGLANG_LANGUAGE_OBJECTIVE_C",
    "TGLANG_LANGUAGE_PHP",
    "TGLANG_LANGUAGE_POWERSHELL",
    "TGLANG_LANGUAGE_PYTHON",
    "TGLANG_LANGUAGE_RUBY",
    "TGLANG_LANGUAGE_RUST",
    "TGLANG_LANGUAGE_SHELL",
    "TGLANG_LANGUAGE_SOLIDITY",
    "TGLANG_LANGUAGE_SQL",
    "TGLANG_LANGUAGE_SWIFT",
    "TGLANG_LANGUAGE_TL",
    "TGLANG_LANGUAGE_TYPESCRIPT",
    "TGLANG_LANGUAGE_XML",
]
language2id = {lang: i for i, lang in enumerate(languages)}


def find_extra_time(time, max_time=0.015):
    extra_time_ids = []
    for i in range(len(time)):
        if time[i] > max_time:
            extra_time_ids.append(i)
    return extra_time_ids


def get_failed_ids(pred):
    failed_ids = []
    for i in range(len(pred)):
        if pred[i].strip() == "":
            failed_ids.append(i)
    return failed_ids


def prepare_gt(paths):
    langs = []
    for path in paths:
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if not folder_name.startswith("TGLANG_LANGUAGE_"):
            folder_name = "TGLANG_LANGUAGE_" + folder_name
        langs.append(language2id[folder_name])
    return langs


def calc_accuracy(pred, gt):
    assert len(pred) == len(gt)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct += 1
    return correct, correct / len(pred)


def get_inference_time_stats(times):
    intervals = [0, 5, 7, 10, 15, 20, 30, 40, 50]
    stats = {f"{intervals[i]}ms-{intervals[i+1]}ms": 0 for i in range(len(intervals) - 1)}
    stats[">50ms"] = 0
    for time in times:
        time_ms = time * 1000  # Convert seconds to milliseconds
        if time_ms > 50:
            stats[">50ms"] += 1
            continue
        for i in range(len(intervals) - 1):
            if intervals[i] <= time_ms <= intervals[i + 1]:
                stats[f"{intervals[i]}ms-{intervals[i+1]}ms"] += 1
                break
    return stats


def calc_per_label_accuracy(pred, gt):
    assert len(pred) == len(gt)
    correct = {}
    total = {}
    for i in range(len(pred)):
        if gt[i] not in correct:
            correct[gt[i]] = 0
            total[gt[i]] = 0
        total[gt[i]] += 1
        if pred[i] == gt[i]:
            correct[gt[i]] += 1
    return correct, total


def calc_language_pair_accuracy(pred, gt, lang1, lang2, include_only_gt=False):
    relevant_preds = []
    relevant_gts = []

    for p, g in zip(pred, gt):
        if include_only_gt:
            if g in [lang1, lang2]:
                relevant_preds.append(p)
                relevant_gts.append(g)
        else:
            if (p in [lang1, lang2]) and (g in [lang1, lang2]):
                relevant_preds.append(p)
                relevant_gts.append(g)

    correct = sum(1 for i in range(len(relevant_preds)) if relevant_preds[i] == relevant_gts[i])
    total = len(relevant_preds)

    if total == 0:
        return 0, 0  # Avoid division by zero if no relevant predictions

    return correct, total, correct / total


def get_file_length_stats(times, filelen, intervals):
    stats = {
        f"{intervals[i]}-{intervals[i+1]}": {"total_time": 0, "count": 0, "samples": 0}
        for i in range(len(intervals) - 1)
    }
    stats[">" + str(intervals[-1])] = {"total_time": 0, "count": 0, "samples": 0}

    for i, length in enumerate(filelen):
        length = int(length)
        time = times[i]
        matched = False
        for j in range(len(intervals) - 1):
            if intervals[j] <= length < intervals[j + 1]:
                stats[f"{intervals[j]}-{intervals[j+1]}"]["total_time"] += time
                stats[f"{intervals[j]}-{intervals[j+1]}"]["count"] += 1
                stats[f"{intervals[j]}-{intervals[j+1]}"]["samples"] += 1
                matched = True
                break
        if not matched and length >= intervals[-1]:
            stats[">" + str(intervals[-1])]["total_time"] += time
            stats[">" + str(intervals[-1])]["count"] += 1
            stats[">" + str(intervals[-1])]["samples"] += 1

    avg_stats = {}
    for key, value in stats.items():
        avg_time = value["total_time"] / value["count"] if value["count"] > 0 else 0
        avg_stats[key] = (avg_time, value["samples"])

    return avg_stats


def calc_accuracy_per_file_length(pred, gt, filelen, intervals):
    interval_accuracies = {}
    for interval in intervals:
        relevant_ids = [i for i, l in enumerate(filelen) if l <= interval]
        if len(relevant_ids) == 0:
            interval_accuracies[interval] = (0, 0)
            continue
        relevant_pred = [pred[i] for i in relevant_ids]
        relevant_gt = [gt[i] for i in relevant_ids]
        correct = sum(1 for i in range(len(relevant_pred)) if relevant_pred[i] == relevant_gt[i])
        interval_accuracies[interval] = (correct, len(relevant_pred))
    return interval_accuracies


def analyse(path_to_csv):
    with open(path_to_csv, "r") as f:
        data = f.readlines()

    data = [line.strip().split(",") for line in data]
    times, pred_labels, files, filelen = zip(*data)

    # Convert times and file lengths to float
    times = [float(t) for t in times]
    filelen = [float(l) for l in filelen]

    pred_labels = [int(l) for l in pred_labels]
    gt = prepare_gt(files)

    # General Accuracy
    correct_num, accuracy = calc_accuracy(pred_labels, gt)
    avg_time = sum(times) / len(times)
    print("### Overall Metrics ###")
    print(f"Accuracy: {accuracy:.5f}  [{correct_num}/{len(pred_labels)}]")
    print(f"Avg time: {avg_time:.3f}s [min: {min(times):.3f}s, max: {max(times):.3f}s]")

    # File Length and Inference Time Statistics
    file_length_intervals_stats = [0, 256, 512, 1024, 2048, 3072]
    file_length_stats = get_file_length_stats(times, filelen, file_length_intervals_stats)
    print("\n### File Length and Inference Time Statistics ###")
    for interval, (avg_time, samples) in file_length_stats.items():
        print(f"{interval:>10s}: {avg_time:.3f}s [files: {str(samples):<4s}]")

    # Inference Time Statistics
    time_stats = get_inference_time_stats(times)
    print("\n### Inference Time Statistics ###")
    for interval, count in time_stats.items():
        print(f"{interval:>10s}: {count}")

    file_length_intervals = [50, 100, 200, 300, 500, 1024, 2048]
    file_length_accuracies = calc_accuracy_per_file_length(pred_labels, gt, filelen, file_length_intervals)

    print("\n### Accuracy Per File Length ###")
    for interval, (correct, total) in file_length_accuracies.items():
        accuracy = correct / total if total > 0 else 0
        interval_str = f"0-{interval}"
        print(f"{interval_str:>7s}: {accuracy:.3f} [{correct}/{total}]")

    # Language Pair Accuracies
    CvsCPP_accuracy = calc_language_pair_accuracy(
        pred_labels,
        gt,
        language2id["TGLANG_LANGUAGE_C"],
        language2id["TGLANG_LANGUAGE_CPLUSPLUS"],
    )
    JSvsTS_accuracy = calc_language_pair_accuracy(
        pred_labels,
        gt,
        language2id["TGLANG_LANGUAGE_JAVASCRIPT"],
        language2id["TGLANG_LANGUAGE_TYPESCRIPT"],
    )
    OTHERvsCODE_accuracy = calc_language_pair_accuracy(
        pred_labels,
        gt,
        language2id["TGLANG_LANGUAGE_OTHER"],
        language2id["TGLANG_LANGUAGE_CPLUSPLUS"],
        include_only_gt=True,  # Include only ground truths for OTHER vs CODE comparison
    )

    print("\n### Language Pair Accuracies ###")
    print(
        f"OTHER vs CODE Accuracy: {OTHERvsCODE_accuracy[2]:.3f} [{OTHERvsCODE_accuracy[0]}/{OTHERvsCODE_accuracy[1]}]"
    )
    print(f"     JS vs TS Accuracy: {JSvsTS_accuracy[2]:.3f} [{JSvsTS_accuracy[0]}/{JSvsTS_accuracy[1]}]")
    print(f"     C vs CPP Accuracy: {CvsCPP_accuracy[2]:.3f} [{CvsCPP_accuracy[0]}/{CvsCPP_accuracy[1]}]")

    print("\n### Per Label Accuracy ###")
    correct, total = calc_per_label_accuracy(pred_labels, gt)
    for i in range(len(languages)):
        print(f"{languages[i]:<30s}: {correct[i] / total[i]:.3f} [{correct[i]}/{total[i]}]")

    print("\n\n")
    print("#" * 20 + "   MISCLASS  " + "#" * 20)
    max_label_len = max([len(l) for l in languages])
    for i in range(len(pred_labels)):
        if pred_labels[i] != gt[i]:
            pred_label_str = languages[pred_labels[i]]
            print(f"[{times[i]:.3f}s] [{pred_label_str:>{max_label_len + 2}}] {files[i]}")


if __name__ == "__main__":
    path_to_csv = sys.argv[1]
    analyse(path_to_csv)
