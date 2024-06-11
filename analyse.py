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

def draw_combined_process(title, model_results, label):
    """在同一个图上绘制多个模型的训练过程曲线"""
    plt.figure()
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    
    for model_name, results in model_results.items():
        iters = results['train_iters'] if label == 'training loss' else results['test_iters']
        data = results['train_loss'] if label == 'training loss' else results['test_acc']
        plt.plot(iters, data, label=model_name)
    
    plt.legend()
    plt.grid()
    plt.show()
# 加载保存的结果
if __name__ == "__main__":
    model_names = ['VGGNet', 'VisionTransformer', 'AlexNet', 'ResNet']
    base_path = r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\model_params"
    model_results = {}
    
    for model_name in model_names:
        model_save_path = os.path.join(base_path, model_name)
        results_file_path = os.path.join(model_save_path, "results.pkl")
        
        if not os.path.exists(results_file_path):
            print(f"Results file for {model_name} not found at {results_file_path}")
            continue
        
        results = load_results(results_file_path)
        model_results[model_name] = results
    
    # 绘制训练损失和测试准确率在同一个图上
    draw_combined_process("Training Loss Comparison", model_results, "training loss")
    draw_combined_process("Test Accuracy Comparison", model_results, "test accuracy")