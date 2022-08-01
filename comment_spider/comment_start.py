import sys

import pandas as pd

sys.path.append("..")
import os
import requests
import time
import random
import xlwt
from bs4 import BeautifulSoup
from tools.Emoji_Process import filter_emoji
from tools.Date_Process import time_process


url_comment = 'https://weibo.cn/comment/{}?&page={}'
'''爬取某个微博的的评论信息'''
def fetch_comment_data(wbid, keyword, cookie):
    cookies = {
        "Cookie": cookie}

    r_comment = requests.get('https://weibo.cn/comment/{}'.format(wbid), cookies=cookies)
    print(r_comment.url)
    soup_comment = BeautifulSoup(r_comment.text, 'lxml')
    flag = False
    try:
        flag = soup_comment.select('.c')[-1].text.startswith('还没有人针对')
    except Exception as e:
        page_num = 1

    if flag:
        print("--------- 此微博没有人评论！ ---------\n")
        return None
    else:
        try:
            page_num = int(soup_comment.select_one(".pa").text.split()[-1].split("/")[-1].split("页")[0])
        except Exception as e:
            page_num = 1

    page_id = 1
    commentinfos = []
    # print("--------- 此微博 {} 的评论页数共有 {} 页 ---------\n".format(wbid, page_num))
    # page_num = min(page_num, 50)
    while page_id < page_num + 1:

        time.sleep(random.uniform(0, 2))  # 设置睡眠时间

        # print("++++++ 正在爬取此微博 {} 的第 {} 页评论...... ++++++\n".format(wbid, page_id))
        r_comment = requests.get(url_comment.format(wbid, page_id), cookies=cookies)
        soup_comment = BeautifulSoup(r_comment.text, 'lxml')
        comment_list = soup_comment.select(".c")

        for l in comment_list:
            if str(l.get("id")).startswith("C_"):
                comment_content = filter_emoji(l.select_one(".ctt").text)
                comment_userid = l.select_one("a").get("href")[3:]
                comment_username = l.select_one("a").text
                comment_like = l.select_one(".cc").text.strip()[2:-1]
                comment_createtime = time_process(l.select_one(".ct").text.strip()[:-5])
                # print("评论内容  ：" + comment_content)
                # print("评论用户ID：" + comment_userid)
                # print("评论用户名：" + comment_username)
                # print("评论赞数  ：" + comment_like)
                # print("评论时间  ：" + comment_createtime)
                # print('----------------------------\n')
                commentinfo = {'wb_id': wbid,  # 生成一条评论信息的列表
                               'comment_content': comment_content,
                               'comment_userid': comment_userid,
                               'comment_username': comment_username,
                               'comment_like': comment_like,
                               'comment_createtime': comment_createtime,
                               'keyword': keyword
                               }
                commentinfos.append(commentinfo)

        page_id = page_id + 1

    # print("--------- 此微博的全部评论爬取完毕！---------\n\n")
    return commentinfos


def writeData(info, file_name):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('MySheet')

    worksheet.write(0, 0, "wb_id")
    worksheet.write(0, 1, "comment_content")
    worksheet.write(0, 2, "comment_userid")
    worksheet.write(0, 3, "comment_username")
    worksheet.write(0, 4, "comment_like")
    worksheet.write(0, 5, "comment_createtime")
    worksheet.write(0, 6, "keyword")

    for i in range(len(info)):
        worksheet.write(i + 1, 0, info[i]["wb_id"])
        worksheet.write(i + 1, 1, info[i]["comment_content"])
        worksheet.write(i + 1, 2, info[i]["comment_userid"])
        worksheet.write(i + 1, 3, info[i]["comment_username"])
        worksheet.write(i + 1, 4, info[i]["comment_like"])
        worksheet.write(i + 1, 5, info[i]["comment_createtime"])
        worksheet.write(i + 1, 6, info[i]["keyword"])

    workbook.save(os.path.join(info[0]["keyword"], file_name + '.xls'))
    print("已保存：", file_name + '.xls')


if __name__ == '__main__':
    cookie = input('请输入在浏览器登录weibo.cn时获取的cookie：')
    keyword = input('请输入要爬取评论的关键词：')
    os.makedirs(keyword, exist_ok=True)
    df = pd.read_excel(f'../search_spider/{keyword}.xls')
    wb_id_lst = df['wb_id']
    # wb_id_lst = ['LDBtDwF1W', 'LDtfUE66j', 'LDCS1nE9r', 'LDMbsB5Wb']
    wb_topic = df['wb_topic']
    wb_username = df['wb_username']
    # wb_content = ['火锅店回应厕所标语被质疑侮辱女性', '90后男公务员出轨50岁女领导被降职',
    #               '首例单身女性冻卵案一审败诉', '齐溪说任何阶段的女性都是美的']
    punctuations = '\\/:*?"<>|'
    for i, id in enumerate(wb_id_lst):
        for pun in punctuations:
            wb_topic[i] = wb_topic[i].replace(pun, '')
        print("正在爬取微博：", wb_username[i] + wb_topic[i], "的评论")
        data = fetch_comment_data(id, keyword, cookie)
        if data:
            # print(data)
            print("正在写入微博评论")
            writeData(data, wb_username[i] + wb_topic[i])