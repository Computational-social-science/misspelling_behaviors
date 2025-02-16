from lxml import html
import requests
import csv
import string

# 基础 URL 前缀
base_url = 'https://www.techopedia.com/it-terms/'

# 生成包含26个小写字母的链接
urls = [f"{base_url}{letter}" for letter in string.ascii_lowercase]

# 自定义 User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def fetch_terms_from_url(url, limit=40):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tree = html.fromstring(response.content)

        # 使用 XPath 表达式获取符合条件的文本内容
        terms = tree.xpath("//div/a[contains(@href, 'https://www.techopedia.com/definition/') and position() <= 40]/text()")

        # 限制提取的条数
        return terms[:limit]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []


def save_to_csv(data, filename='techopedia_terms.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Term'])
        writer.writerows([(term,) for term in data])


if __name__ == "__main__":
    all_terms = []
    for url in urls:
        print(f"Fetching terms from {url}")
        terms = fetch_terms_from_url(url, limit=40)
        all_terms.extend(terms)

    save_to_csv(all_terms)
    print(f"Saved {len(all_terms)} terms to techopedia_terms.csv")
