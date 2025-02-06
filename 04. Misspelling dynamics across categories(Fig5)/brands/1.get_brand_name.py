import requests
from lxml import html
import csv

# 请求网页
url = 'https://www.forbes.com/lists/global2000/'
response = requests.get(url)
response.raise_for_status()  # 确保请求成功

# 解析网页
tree = html.fromstring(response.content)

# 使用 XPath 查找元素
elements = tree.xpath("//div[contains(@class, 'organizationName') and contains(@class, 'second') and contains(@class, 'table-cell') and contains(@class, 'name')]")

# 提取文本内容
organization_names = [element.text.strip() for element in elements if element.text is not None]

# 保存到 CSV 文件
csv_filename = 'organization_names.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Organization Name'])
    for name in organization_names:
        writer.writerow([name])

print(f"Saved {len(organization_names)} organization names to {csv_filename}")
