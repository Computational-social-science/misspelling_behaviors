import os
import re
import pandas as pd

folder_path = "significant_links_output"
# 正则表达式匹配
node_pattern = re.compile(r"Variable ([\w\s\-]+) has (\d+) link\(s\):")
link_pattern = re.compile(r"\(([\w\s\-]+?)\s*(-?\d+)\):\s*pval\s*=\s*([0-9\.]+)\s*\|\s*val\s*=\s*([0-9\.\-]+)")
all_links = []  # 存储所有链接信息


for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r") as file:
            content = file.read()

        segments = content.strip().split("\n\n")[1:]
        time = filename.split(".")[0]

        # 逐段解析
        for segment in segments:
            nodes = node_pattern.findall(segment)
            for node_name, link_count in nodes:
                link_count = int(link_count)

                # 如果有链接，解析链接特征
                if link_count > 0:
                    links = link_pattern.findall(segment)
                    for target, lag, pval, val in links:
                        all_links.append([time, node_name, target, int(lag), float(pval), float(val)])


columns = ["Time", "SourceNode", "TargetNode", "Lag", "Pval", "Val"]
df_links = pd.DataFrame(all_links, columns=columns)
freq_file_path = "frequency_data.xlsx"
df_freq = pd.read_excel(freq_file_path)
df_freq.rename(columns={"Date": "Time"}, inplace=True)
df_freq["Time"] = pd.to_datetime(df_freq["Time"]).dt.strftime("%Y-%m-%d")
df_combined = pd.merge(df_links, df_freq, on="Time", how="left")
output_path = "combined_data.xlsx"
df_combined.to_excel(output_path, index=False)


