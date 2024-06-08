import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import logging

class TextHandler(logging.Handler):
    """This class allows you to log to a Tkinter Text or ScrolledText widget"""
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)  # Autoscroll to the end
        self.text_widget.after(0, append)

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('图像分类系统')
        self.root.geometry("700x500+1100+150")
        self.interface()
        self.setup_logging()

    def interface(self):
        """"界面编写位置"""
        self.Button0 = tk.Button(self.root, text="数据选择", command=self.choose_dataset)
        self.Button0.grid(row=0, column=0, padx=5, pady=5)

        self.Button1 = tk.Button(self.root, text="模型选择", command=self.choose_model)
        self.Button1.grid(row=0, column=1, padx=5, pady=5)

        self.Button2 = tk.Button(self.root, text="模型训练", command=self.choose_image)
        self.Button2.grid(row=0, column=2, padx=5, pady=5)
        
        self.Button3 = tk.Button(self.root, text="模型推理", command=self.choose_image)
        self.Button3.grid(row=0, column=3, padx=5, pady=5)

        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=1, column=0, columnspan=3, pady=10)

        self.image_size_label = tk.Label(self.root, text="选择的图片尺寸: ")
        self.image_size_label.grid(row=2, column=0, columnspan=3, pady=5)

        self.log_display_label = tk.Label(self.root, text="操作日志")
        self.log_display_label.grid(row=3, column=0, columnspan=3, pady=5)

        self.log_display = tk.Text(self.root, width=100, height=10, state='disabled')
        self.log_display.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

    def setup_logging(self):
        # 创建一个日志处理程序，将日志消息发送到日志显示窗口
        text_handler = TextHandler(self.log_display)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
        logging.getLogger().setLevel(logging.INFO)

    def choose_dataset(self):
        logging.info("选择数据集")
        # 弹出选择框
        model_window = tk.Toplevel(self.root)
        model_window.title("选择数据集")
        model_window.geometry("300x400")

        tk.Label(model_window, text="请选择一个数据集类别:").pack(pady=10)

        dataset = ["西瓜 ", "苦瓜 ", "哈密瓜"]
        selected_dataset = tk.StringVar(value=dataset[0])

        for dataset in dataset:
            tk.Radiobutton(model_window, text=dataset, variable=selected_dataset, value=dataset).pack(anchor='w')

        tk.Label(model_window, text="数据集类别:").pack(pady=10)

        # chosen_model = simpledialog.askstring("模型选择", "请输入模型（例如：VIT、VGG、RESNET）:")


        def confirm_selection():
            chosen_dataset = selected_dataset.get()
            logging.info(f"选择的数据集：{chosen_dataset}")
            model_window.destroy()
            self.log_display.insert(tk.END, f"选择的数据集：{chosen_dataset}\n")
        confirm_button = tk.Button(model_window, text="确定", command=confirm_selection)
        confirm_button.pack(pady=20)

    def choose_model(self):
        logging.info("模型选择按钮点击")
        # Prompt user for model and size
        chosen_model = simpledialog.askstring("模型选择", "请输入模型（例如：VIT、VGG、RESNET）:")
        chosen_size = simpledialog.askstring("图片尺寸", "请输入图片尺寸（例如：224x224、384x384）:")
        if chosen_model and chosen_size:
            logging.info(f"选择的模型：{chosen_model}, 选择的尺寸: {chosen_size}")
            self.log_display.insert(tk.END, f"选择的模型：{chosen_model}, 选择的尺寸: {chosen_size}\n")
        else:
            logging.info("取消了模型选择")

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            logging.info(f"选择的图片路径: {file_path}")
            image = Image.open(file_path)
            image = image.resize((250, 250))  # 调整图片大小以适应标签
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 防止图片被垃圾回收

            # Display image dimensions
            width, height = image.size
            self.image_size_label.config(text=f"选择的图片尺寸: {width}x{height}")
        else:
            logging.info("未选择图片")

# 运行应用
if __name__ == "__main__":
    app = GUI()
    app.root.mainloop()
