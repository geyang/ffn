from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmx import doc

if __name__ == "__main__":
    doc @ """
    ## Learning without a target network
    
    We need to first evaluate the bias using a target network. 
    """

    dumper = doc.window.logger


    def rmse(delta):
        return (delta ** 2).mean() ** 0.5


    def error_mean_var(predictions, targets):
        error = predictions - targets[None, ...]
        return rmse(error), np.std(error)


    def collect_all(path):
        files = dumper.glob(path)
        all = []
        for file in files:
            data, = dumper.load_pkl(file)
            all.append(data)
        return np.stack(all)


    with dumper.Prefix("data"):
        gt_q_values, = dumper.load_pkl("gt_q_values.pkl")
        gt = dumper.load_pkl("gt_q_values.pkl")

    with dumper.Prefix('data'):
        mlp_q_values = collect_all(f"mlp_q_values_*.pkl")
        rff_q_values = collect_all(f"rff_q_values_*.pkl")
        rff_no_tgt_q_values = collect_all(f"rff_no_tgt_q_values_*.pkl")

    with doc, doc.table().figure_row() as row:

        rmse, std = np.stack([
            error_mean_var(mlp_q_values, gt_q_values),
            error_mean_var(rff_q_values, gt_q_values),
            error_mean_var(rff_no_tgt_q_values, gt_q_values)
        ]).T

        labels = ["MLP\n+ Target", "FFN\n+ Target", "FFN\nNo Target"]
        colors = np.array([(1, 0, 0, 0.3), (1, 165 / 255, 0, 0.9),
                           (34 / 255, 169 / 255, 1, 0.9)])
        label_pos = rmse + [-0.8, 0.2, 0.2]
        # alphas = [0.3, 0.9, 0.8]

        plt.figure(figsize=(4.5, 4.8))
        plt.bar(range(3), rmse, yerr=std, color=colors, tick_label=labels, capsize=20)
        for pos, err, v_pos in zip(range(3), rmse, label_pos):
            # plt.bar(pos, err, yerr=var, color=c, alpha=a, tick_label=l, capsize=20)
            plt.text(pos, v_pos, f"{err:.2f}", ha="center", va="bottom")

        plt.title("Bias & Variance")
        plt.ylabel("RMSE")
        plt.ylabel("Approx. Error")
        plt.tight_layout()
        row.savefig(f"{Path(__file__).stem}/bias_comparison.png")
        plt.savefig(f"{Path(__file__).stem}/bias_comparison.pdf")

    # doc @ """
    # Now plot the comparison
    # """
    # with doc:
    #     plt.plot(states, gt_q_values[0], color="black", linewidth=1, label="Ground Truth", zorder=2)
    #     plt.plot(states, rff_no_tgt_q_values[0], color="#23aaff", linewidth=4, label="FFN (No Target)", alpha=0.8)
    #     plt.plot(states, rff_q_values[0], color="orange", linewidth=3, label="FFN", alpha=0.9)
    #     plt.plot(states, mlp_q_values[0], color="red", linewidth=3, label="MLP", alpha=0.3)
    #     plt.title("Neural Fitted Q Iteration")
    #     plt.xlabel("State [0, 1)")
    #     plt.ylabel("Value")
    #     plt.legend(loc="upper left", framealpha=0.8)
    #     plt.ylim(3, 7.5)
    #     plt.tight_layout()
    #     doc.savefig(f'{Path(__file__).stem}/comparison.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    #     plt.savefig(f'{Path(__file__).stem}/comparison.pdf', dpi=300)
    # doc.flush()![](../../../../../var/folders/0z/ckgrsnxj1cx0s4jr1tgw2lmm0000gn/T/TemporaryItems/NSIRD_screencaptureui_0UHP96/Screen Shot 2022-03-17 at 3.45.56 AM.png)
