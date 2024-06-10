import pickle
import matplotlib.pyplot as plt
import os
import config


def load_results(file_path):
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    return results

def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()

# 加载保存的结果
if __name__ == "__main__":
    results_file_path = os.path.join(config.modle_save_path, "results.pkl")
    results = load_results(results_file_path)
    # 可视化训练损失和测试准确率
    draw_process(f"{config.MODEL} training loss", "red", results["train_iters"], results["train_loss"], "training loss")
    draw_process(f"{config.MODEL} test accuracy", "green", results["test_iters"], results["test_acc"], "test accuracy")
