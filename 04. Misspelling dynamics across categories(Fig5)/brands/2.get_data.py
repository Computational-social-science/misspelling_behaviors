import csv
import requests
from lxml import html
import random
import time

def get_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    ]
    return random.choice(user_agents)


def get_ngram_data(key):
    key = key.replace(" ", "+")
    url = f"https://books.google.com/ngrams/graph?content={key}&year_start=1500&year_end=2019&corpus=en-2019&smoothing=3"
    user_agent = get_user_agent()
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)
    web_content = response.content

    tree = html.fromstring(web_content)

    script_content = tree.xpath("//script[@id='ngrams-data']/text()")
    if not script_content:
        print(f"No data found for key: {key}")
        return []

    script_content = script_content[0]
    data = eval(script_content)

    return data


def save_to_csv(ngrams, filename):
    start_time = time.time()
    all_data = []

    for ngram in ngrams:
        data = get_ngram_data(ngram)
        if data:
            all_data.extend(data)

    if all_data:
        max_length = max(len(item["timeseries"]) for item in all_data)

        column_data = [["ngram"] + [item["ngram"] for item in all_data]]
        for i in range(max_length):
            year = i + 1500
            row_data = [f"{year}"]
            for item in all_data:
                if i < len(item["timeseries"]):
                    row_data.append(item["timeseries"][i])
                else:
                    row_data.append("")
            column_data.append(row_data)

        with open(filename, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(column_data)

        end_time = time.time()  # 记录结束时间
        duration = end_time - start_time  # 计算耗时

        print(f"数据已成功写入 {filename}，耗时 {duration:.2f} 秒")
    else:
        end_time = time.time()  # 记录结束时间
        duration = end_time - start_time  # 计算耗时
        print(f"没有找到任何数据用于 {filename}，耗时 {duration:.2f} 秒")


# 处理CSV文件中的每一行
input_csv = "organization_misspellings.csv"

with open(input_csv, mode='r', newline='', encoding='ISO-8859-1') as infile:
    reader = csv.reader(infile)
    for row in reader:
        ngrams = row[0].split(',')
        filename = f"./data/{ngrams[0].split(',')[0]}.csv"
        save_to_csv(ngrams, filename)


