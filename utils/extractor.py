import sys
import re
from lxml import etree
from bs4 import BeautifulSoup
from weibo_preprocess_toolkit import WeiboPreprocess
from harvesttext import HarvestText
import pyhanlp


def get_post_time(timestr="Mon Dec 14 11:26:56 +0800 2020"):
    import calendar
    import locale

    # locale.setlocale(locale.LC_ALL, "C.UTF-8")

    if not timestr:
        return ""
    temp = timestr.split(" ")
    time_area = temp[-2]
    if time_area != "+0800":
        print(time_area)
    # day_time = ':'.join(temp[3].split(':')[:-1])
    day_time = temp[3]
    return (
        temp[-1]
        + "-"
        + "{:0=2}".format(list(calendar.month_abbr).index(temp[1]))
        + "-"
        + temp[2]
        + " "
        + day_time
    )


class WeiboText:
    def __init__(self):

        self.preprocess = WeiboPreprocess()
        self.ht = HarvestText()
        self.CharTable = pyhanlp.JClass("com.hankcs.hanlp.dictionary.other.CharTable")
        self.d1 = re.compile(r"（.*）")
        self.d2 = re.compile(r"点击播放")  # 点击播放>>
        self.d3 = re.compile(r"在.*获取更多信息")
        self.d4 = re.compile(r"速围观")
        self.d5 = re.compile(r"我获得了.*的红包")
        self.d6 = re.compile(r"#.*#")

    def get_cleaned_text(self, html):
        # TODO: <br /> -> 空格
        # TODO: 多空格压缩（HarvestText）
        # TODO: 去奇怪符号
        # TODO: 表情字符icon翻译（翻译表） + 冗余字符icon去除（HarvestText）
        # TODO: 去数字（？）
        # TODO: 繁简体转化
        # TODO: 固定噪音去除（weibo_preprocess_toolkit） + HarvestText中自定义
        # 1. （分享自）、（通过 录制）
        # 2. 点击播放>>
        # 3. 在XXX获取更多信息
        # 4. 速围观
        # 5. ...全文
        # 6. 我获得了XXX的红包
        # 7. 打卡第X天
        soup = BeautifulSoup(html, features="lxml")
        tmp_a = [i.extract() for i in soup.find_all("a")]
        # 保留图片表情文本（但是一些表情比如微笑可能有反讽意味）
        # for i in soup.find_all('span', class_='url-icon'):
        #     i.append(i.img.attrs['alt'])
        # return soup.get_text().lower().strip()
        text = soup.get_text()
        text = self.d1.sub("", text)
        text = self.d2.sub("", text)
        text = self.d3.sub("", text)
        text = self.d4.sub("", text)
        text = self.d5.sub("", text)
        text = self.d6.sub("", text)
        # 使用HarvestText清洗文本（空格压缩，去字符表情）
        content = self.CharTable.convert(text)
        cleaned_text = self.ht.clean_text(content, weibo_topic=True)
        # 使用weibo_preprocess_toolkit清洗文本（繁简体转化，去固定噪音，去数字，）
        cleaned_text = self.preprocess.clean(cleaned_text)
        return cleaned_text.strip()

    @staticmethod
    def get_raw_text(text_body):
        return etree.HTML(text_body).xpath("string(.)")

    @staticmethod
    def get_weibo_selector(text_body):
        return etree.HTML(text_body)

    @staticmethod
    def string_to_int(string):
        """字符串转换为整数"""
        if isinstance(string, int):
            return string
        elif string.endswith("万+"):
            string = int(string[:-2] + "0000")
        elif string.endswith("万"):
            string = int(string[:-1] + "0000")
        return int(string)

    @staticmethod
    def standardize_info(weibo):
        """标准化信息，去除乱码"""
        for k, v in weibo.items():
            if (
                "bool" not in str(type(v))
                and "int" not in str(type(v))
                and "list" not in str(type(v))
                and "long" not in str(type(v))
            ):
                weibo[k] = (
                    v.replace("\u200b", "")
                    .encode(sys.stdout.encoding, "ignore")
                    .decode(sys.stdout.encoding)
                )
        return weibo