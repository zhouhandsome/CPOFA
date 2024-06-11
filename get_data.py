from concurrent.futures import ThreadPoolExecutor
import requests
import os
import numpy as np
import threading
from PIL import Image
cookies = {
    'BDqhfp': '%E6%B0%B4%E6%9D%AF%26%260-10-1undefined%26%260%26%261',
    'BIDUPSID': 'D5B172F42F4253691C840077D1CDC2EA',
    'PSTM': '1662877779',
    'ZFY': 'EPzmzt7IRJiP:ApZdvAw2A1eNGCSdruPV6EfPIkY0cVQ:C',
    'BAIDU_WISE_UID': 'wapp_1664634518198_256',
    '__bid_n': '183d63a017555116c04207',
    'FEID': 'v10-62b6f5e4ca106de2bb783de7d36c2528a87381db',
    '__xaf_thstime__': '1672909781354',
    '__xaf_fpstarttimer__': '1672980069836',
    '__xaf_fptokentimer__': '1672980070075',
    'BAIDUID': '5DF0B9301F546EF49E353A4AC42F0BF5:FG=1',
    'BAIDUID_BFESS': '5DF0B9301F546EF49E353A4AC42F0BF5:FG=1',
    'jsdk-uuid': '62a78a3e-796b-4aef-8941-c9f76a96826c',
    'FPTOKEN': 'EGoSOVCgweDlLCzSuMK866jyRVBID7Oto5T2JtdXFriJZDxCYOS2ALOddgKTAf4qJmOxmNpgT0gPPRR6E1uZFB958UeWjDfyij6u0QiCTJ3H9rMnMrsWDGowuZiP8deR/fMZCbgjfgNhLHQoj5fTIu95C38KIh0b1v9hXDXgz3TqOqAy9h7d/LCgdqZDIhoWh2w9EwdRRmkQxGiyQSj/L/dD98jprCJ7Eg66bbeI9qBsOBM8KcIVF1zZ1b2Lgg0zNmPRzu0DLywYewJwDkt/6Tx/MboaJSeUAl0FyzXAi6gQc9LEjUTruWmk2C522OXUrfl7/O9hp/y8W88QNHGrjG4sHwOWZ0sePfEc++O1xXSXfu07S/rpuBBoEnTcuOi5HXdkt45a4ozMeDjqeetUAg==|9965ulOosA/8UeekyXb5Z8DAj+H/temWK+k8lnqb6Hs=|10|6f420f069c31bbcfd48d5ad72a8409ad',
    'BDUSS': 'VXdC1vay05WDY1UG9tbkUtWWF0N2xQWWQ1RkdzYkFuZzVqYmRGQW1IfnNmaTVrRVFBQUFBJCQAAAAAAQAAAAEAAACKQT0DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOzxBmTs8QZkcn',
    'BDUSS_BFESS': 'VXdC1vay05WDY1UG9tbkUtWWF0N2xQWWQ1RkdzYkFuZzVqYmRGQW1IfnNmaTVrRVFBQUFBJCQAAAAAAQAAAAEAAACKQT0DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOzxBmTs8QZkcn',
    'BDRCVFR[X_XKQks0S63]': 'mk3SLVN4HKm',
    'firstShowTip': '1',
    'indexPageSugList': '%5B%22%E6%B0%B4%E6%9D%AF%22%2C%22%E8%A5%BF%E7%93%9C%22%5D',
    'cleanHistoryStatus': '0',
    'BDRCVFR[dG2JNJb_ajR]': 'mk3SLVN4HKm',
    'BDRCVFR[-pGxjrCMryR]': 'mk3SLVN4HKm',
    'ab_sr': '1.0.1_NmExOWUxYWY4YzNmMzdjM2FhZWU1MDcxYzNiMjg3YmEyNzM2MTJlZDBkYTBkMDMzNGRiMGJiOGE1ZDU0ZDlmYjIyNTY3MTE5M2Q4NWI3MWU4Yzg4YTM1MDAwNDliZjhiMTkxMmJhNjFhYmQ3YzA0MjExMTU0NmJlMThmMmExYjc3N2FhZDJkOTlhOGNlMTMxZWU1MTBhNzE2ZmQxMDI3OA==',
    'userFrom': 'null',
}
headers = {
    'Accept': 'text/plain, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&fm=index&pos=history&word=%E6%B0%B4%E6%9D%AF',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.63',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
lock = threading.Lock()
dataset_path = r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\picture_classify_datast"
# 创建文件夹，用于存储图片
def creat_folder(keyword):
    if os.path.exists(os.path.join(dataset_path,keyword)):
        print('文件夹{} 已存在，之后直接将爬取到的图片保存至该文件夹中'.format(os.path.join(dataset_path ,keyword)))
    else:
        os.makedirs(os.path.join(dataset_path ,keyword))
        print('新建文件夹：{}'.format(os.path.join(dataset_path , keyword)))

# 下载单张图片
def download_picture(img_url, img_title,keyword):
    global sum  # 使用全局变量
    picture = requests.get(img_url).content
    picture_path = os.path.join(dataset_path , keyword ,str(sum) + '.jpg')
    if os.path.exists(picture_path):
        with lock:
            sum +=1
    else:
        with lock:
            sum +=1
            print('{}类，第{}张图片已完成下载'.format(keyword,sum))
        with open(picture_path, 'wb') as f:
            f.write(picture)

#下载一类图片
def download_pictures(keyword, number):
    global sum
    count = 1  # 页数
    sum = 0
    while sum < number:
        page = count * 30
        params = (
            ('tn', 'resultjson_com'),
            ('logid', '11331131744387162223'),
            ('ipn', 'rj'),
            ('ct', '201326592'),
            ('is', ''),
            ('fp', 'result'),
            ('fr', ''),
            ('word', f'{keyword}'),
            ('queryWord', f'{keyword}'),
            ('cl', '2'),
            ('lm', '-1'),
            ('ie', 'utf-8'),
            ('oe', 'utf-8'),
            ('adpicid', ''),
            ('st', '-1'),
            ('z', ''),
            ('ic', '0'),
            ('hd', ''),
            ('latest', ''),
            ('copyright', ''),
            ('s', ''),
            ('se', ''),
            ('tab', ''),
            ('width', ''),
            ('height', ''),
            ('face', '0'),
            ('istype', '2'),
            ('qc', ''),
            ('nc', '1'),
            ('expermode', ''),
            ('nojc', ''),
            ('isAsync', ''),
            ('pn', f'{page}'),
            ('rn', '30'),
            ('gsm', '1e'),
            ('1678518491165', ''),
        )
        response = requests.get('https://image.baidu.com/search/acjson', headers=headers, params=params, cookies=cookies)
        try:
            json_dicts = response.json()['data']
            # for dict in json_dicts:
            #     img_name = dict['fromPageTitle']
            #     img_url = dict['hoverURL']
            #     download_picture(img_url,img_name,keyword)
            with ThreadPoolExecutor() as executor:
                futures = []
                for d in json_dicts:
                    img_name = d['fromPageTitle']
                    img_url = d['hoverURL']
                    futures.append(executor.submit(download_picture, img_url, img_name,keyword))
        except Exception as e:
            print(f"Exception occurred: {e}")
            pass
        count+=1

def clear_dataset():
    dataset_path = 'D:\\VsCodeProjects\\pythonProjects\\Smart_Algorithm\\picture_classify_datast'
    # 删除gif格式
    classes = os.listdir(dataset_path)
    for picture_class in classes:
        for file in os.listdir(os.path.join(dataset_path, picture_class)):
            file_path = os.path.join(dataset_path, picture_class, file)
            # file_path = str(file_path.encode('utf-8'))
            img = np.array(Image.open(file_path))
            if img is None:
                print(file_path, '读取错误，删除')
                os.remove(file_path)
            try:
                channel = img.shape[2]
                if channel != 3:
                    print(file_path, '非三通道，删除')
                    os.remove(file_path)
            except:
                print(file_path, '非三通道，删除')
                os.remove(file_path)

if __name__ == '__main__':
    keywords = ['水杯', '衬衫', '西瓜', '手机', '铅笔']
    number = 60
    for keyword in keywords:
        creat_folder(keyword)
        download_pictures(keyword,number)
    clear_dataset()


