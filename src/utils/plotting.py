import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc


def plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=None, roc=False, titleRoc="ROC Curve", titleEer="FAR, FRR, and EER Curve"):
    if not isinstance(FAR, np.ndarray):
        FAR = np.array(FAR)

    if not isinstance(FRR, np.ndarray):
        FRR = np.array(FRR)

    if not isinstance(GRR, np.ndarray):
        GRR = np.array(GRR)

    if DIR is not None and not isinstance(DIR, np.ndarray):
        DIR = np.array(DIR)

    FAR_itp = interp1d(thresholds, FAR, kind='linear', fill_value="extrapolate")
    FRR_itp = interp1d(thresholds, FRR, kind='linear', fill_value="extrapolate")

    eer_threshold = brentq(lambda x: FAR_itp(x) - FRR_itp(x), 0.0, 1.0)
    index = np.where(np.round(thresholds, 2) == np.round(eer_threshold, 2))[0][0]
    eer = FAR_itp(eer_threshold)
    print(index)
    print(f"Threshold@EER: {thresholds[index]:.4f}")

    zero_far_index = np.where(FAR <= 0.0001)[0][0] if np.any(FAR <= 0.0001) else -1
    zero_frr_index = np.where(FRR <= 0.0001)[0][-1] if np.any(FRR <= 0.0001) else -1

    plt.figure()
    plt.plot(thresholds, FAR, label="FAR(t)", color="blue")
    plt.plot(thresholds, FRR, label="FRR(t)", color="green")
    plt.plot(thresholds, GRR, linestyle="--", label="GRR(t)", color="purple")
    if DIR is not None:
        plt.plot(thresholds, DIR[:,0], linestyle="--", label=f"Rank-1(EER): {np.round(DIR[:,0][index], 4)}", color="orange")

    plt.scatter(eer_threshold, eer, color="red", label="EER: {:.2f}".format(eer))
    if zero_far_index != -1:
        plt.scatter(thresholds[zero_far_index], FRR[zero_far_index], color="blue", label="ZeroFAR")
    if zero_frr_index != -1:
        plt.scatter(thresholds[zero_frr_index], FAR[zero_frr_index], color="blue", label="ZeroFRR")

    plt.annotate("EER", (eer_threshold, eer), textcoords="offset points", xytext=(-20, 10), ha='center', color='red')
    if zero_far_index != -1:
        plt.annotate("ZeroFAR", (thresholds[zero_far_index], FRR[zero_far_index]), textcoords="offset points", xytext=(-30, -15),
                     ha='center', color='blue')
        plt.axvline(x=thresholds[zero_far_index], linestyle="--", color="blue", alpha=0.7)
    if zero_frr_index != -1:
        plt.annotate("ZeroFRR", (thresholds[zero_frr_index], FAR[zero_frr_index]), textcoords="offset points", xytext=(-30, -15),
                     ha='center', color='green')
        plt.axvline(x=thresholds[zero_frr_index], linestyle="--", color="green", alpha=0.7)

    plt.axvline(x=eer_threshold, linestyle="--", color="gray", alpha=0.7)
    plt.axhline(y=eer, linestyle="--", color="gray", alpha=0.7)
    plt.xlabel("t (Threshold)")
    plt.ylabel("Error")
    plt.title(titleEer)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    tpr = 1 - FRR
    sorted_indices = np.argsort(FAR)
    FAR = FAR[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    roc_auc = auc(FAR, tpr_sorted)

    if roc:
        plt.figure()
        plt.plot(FAR, tpr_sorted, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.xlabel("False Positive Rate (FAR)")
        plt.ylabel("True Positive Rate (1 - FRR)")
        plt.title(titleRoc)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)  # Diagonal line for random classifier
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
