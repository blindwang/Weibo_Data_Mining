# -*- coding:utf-8 -*-
import sys

sys.path.append("..")
import re
import time
import requests
from bs4 import BeautifulSoup
from tools.Date_Process import time_process
from tools.Number_Process import num_process
import xlwt


# url_template = 'https://s.weibo.com/weibo?q={}&typeall=1&suball=1&timescope=custom:{}:{}&Refer=g&page={}'  # 要访问的微博搜索接口URL
url_template = "https://s.weibo.com/weibo?q={}&xsort=hot&suball=1&Refer=g&page={}"
# url_template = "https://s.weibo.com/weibo?q={}&suball=1&Refer=g&page={}"
"""抓取关键词某一页的数据"""


# def fetch_weibo_data(keyword, start_time, end_time, page_id, cookie):
def fetch_weibo_data(keyword, page_id, cookie):
    cookies = {
        "Cookie": cookie}
    # resp = requests.get(url_template.format(keyword, start_time, end_time, page_id), cookies=cookies)
    resp = requests.get(url_template.format(keyword, page_id), cookies=cookies)
    # print(resp.url)
    soup = BeautifulSoup(resp.text, 'lxml')
    all_contents = soup.select('.card-wrap')

    wb_count = 0
    mblog = []  # 保存处理过的微博
    for card in all_contents:
        if (card.get('mid') != None):  # 如果微博ID不为空则开始抓取
            wb_username = card.select_one('.txt').get('nick-name')  # 微博用户名
            href = card.select_one('.from').select_one('a').get('href')
            re_href = re.compile('.*com/(.*)/.*')
            wb_userid = re_href.findall(href)[0]  # 微博用户ID
            if len(card.select('.txt')) > 1:
                try:
                    wb_content = card.select_one('p[node-type ="feed_list_content_full"]').text.strip()  # 微博内容
                except:
                    print(card.select('.txt'))
                    wb_content = card.select_one('.txt').text.strip()  # 微博内容
                wb_content = wb_content.replace("收起全文d", '')
            else:
                wb_content = card.select_one('.txt').text.strip()  # 微博内容
            # wb_content_en = translator.translate(wb_content)
            if wb_content.count('#') >= 2:
                topic_list = wb_content.split('#')
                wb_topic = ' '.join('#' + topic_list[i] + '#' for i in range(1, len(topic_list), 2))
            else:
                wb_topic = wb_content.split(' ')[0]
            wb_create = card.select_one('.from').select_one('a').text.strip()  # 微博创建时间
            wb_url = 'https:' + str(card.select_one('.from').select_one('a').get('href'))  # 微博来源URL
            wb_id = str(card.select_one('.from').select_one('a').get('href')).split('/')[-1].split('?')[0]  # 微博ID
            wb_createtime = time_process(wb_create)
            wb_forward = str(card.select_one('.card-act').select('li')[1].text)  # 微博转发数
            wb_forwardnum = num_process(wb_forward)
            wb_comment = str(card.select_one('.card-act').select('li')[2].text)  # 微博评论数
            wb_commentnum = num_process(wb_comment)
            wb_like = str(card.select_one('.card-act').select_one('em').text)  # 微博点赞数

            if wb_like == '':  # 点赞数的处理
                wb_likenum = '0'
            else:
                wb_likenum = wb_like

            blog = {'wb_id': wb_id,  # 生成一条微博记录的列表
                    'wb_username': wb_username,
                    'wb_userid': wb_userid,
                    'wb_content': wb_content,
                    'wb_topic': wb_topic,
                    # 'wb_content_en': wb_content_en,
                    'wb_createtime': wb_createtime,
                    'wb_forwardnum': wb_forwardnum,
                    'wb_commentnum': wb_commentnum,
                    'wb_likenum': wb_likenum,
                    'wb_url': wb_url
                    }
            mblog.append(blog)
            wb_count = wb_count + 1  # 表示此页的微博数

    print("--------- 正在爬取第%s页 --------- " % page_id + "当前页微博数：" + str(wb_count))
    return mblog


"""抓取关键词多页的数据"""


# def fetch_pages(keyword, start_time, end_time, cookie):
def fetch_pages(keyword, cookie):
    cookies = {
        "Cookie": cookie}
    # resp = requests.get(url_template.format(keyword, start_time, end_time, '1'), cookies=cookies)
    resp = requests.get(url_template.format(keyword, '1'), cookies=cookies)
    # print(resp.url)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, 'lxml')
    if (str(soup.select_one('.card-wrap').select_one('p').text).startswith('抱歉')):  # 此次搜索条件的判断，如果没有相关搜索结果！退出...
        print("此次搜索条件无相关搜索结果！\n请重新选择条件筛选...")
        return
    try:
        page_num = len(soup.select_one('.m-page').select('li'))  # 获取此时间单位内的搜索页面的总数量，
        # print(page_num)
        page_num = int(page_num)
        # print(start_time + ' 到 ' + end_time + " 时间单位内搜索结果页面总数为：%d" % page_num)
        print("搜索结果页面总数为：%d" % page_num)
    except Exception as err:
        page_num = 1

    mblogs = []  # 此次时间单位内的搜索全部结果先临时用列表保存，后存入数据库
    for page_id in range(page_num):
        page_id = page_id + 1
        # mblogs.extend(fetch_weibo_data(keyword, start_time, end_time, page_id, cookie))  # 每页调用fetch_data函数进行微博信息的抓取
        mblogs.extend(fetch_weibo_data(keyword, page_id, cookie))  # 每页调用fetch_data函数进行微博信息的抓取

    writeData(mblogs, keyword)


def writeData(info, keyword):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('MySheet')

    worksheet.write(0, 0, "wb_id")
    worksheet.write(0, 1, "wb_username")
    worksheet.write(0, 2, "wb_userid")
    worksheet.write(0, 3, "wb_content")
    worksheet.write(0, 9, "wb_createtime")
    worksheet.write(0, 5, "wb_forwardnum")
    worksheet.write(0, 6, "wb_commentnum")
    worksheet.write(0, 7, "wb_likenum")
    worksheet.write(0, 8, "wb_url")
    worksheet.write(0, 4, "wb_topic")
    # worksheet.write(0, 5, "wb_content_en")

    for i in range(len(info)):
        # print(news[i])
        worksheet.write(i + 1, 0, info[i]["wb_id"])
        worksheet.write(i + 1, 1, info[i]["wb_username"])
        worksheet.write(i + 1, 2, info[i]["wb_userid"])
        worksheet.write(i + 1, 3, info[i]["wb_content"])
        worksheet.write(i + 1, 9, info[i]["wb_createtime"])
        worksheet.write(i + 1, 5, info[i]["wb_forwardnum"])
        worksheet.write(i + 1, 6, info[i]["wb_commentnum"])
        worksheet.write(i + 1, 7, info[i]["wb_likenum"])
        worksheet.write(i + 1, 8, info[i]["wb_url"])
        worksheet.write(i + 1, 4, info[i]["wb_topic"])
        # worksheet.write(i + 1, 5, info[i]['wb_content_en'])

    workbook.save(keyword + '.xls')


if __name__ == '__main__':
    # keyword = input("请输入要搜索的关键字：")
    # start_time = input("请输入要查询的开始时间：")
    # end_time = input("请输入要查询的结束时间：")

    keyword = '女性'
    # start_time = '2022-07-01-0'
    # end_time = '2022-07-23-0'
    cookie = input('请输入在浏览器登录weibo.cn时获取的cookie：')
    time_start_jishi = time.time()
    fetch_pages(keyword, cookie)
    time_end_jishi = time.time()

    print('本次操作数据全部爬取成功，爬取用时秒数:', (time_end_jishi - time_start_jishi))
