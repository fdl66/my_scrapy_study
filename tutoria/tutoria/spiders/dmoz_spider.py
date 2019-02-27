# coding=utf-8
import time
import scrapy
# 引入容器
from tutoria.items import CourseItem
from tutoria.items import HouseDetail


# class MySpider(scrapy.Spider):
#     #设置name
#     name = "MySpider"
#     #设定域名
#     allowed_domains = ["imooc.com"]
#     #填写爬取地址
#     start_urls = ["https://www.imooc.com/course/list"]
#     #编写爬取方法
#     def parse(self, response):
#         #实例一个容器保存爬取的信息
#         item = CourseItem()
#         #这部分是爬取部分，使用xpath的方式选择信息，具体方法根据网页结构而定
#         #先获取每个课程的div
#         for box in response.xpath('//div[@class="moco-course-wrap"]/a[@target="_self"]'):
#             #获取每个div中的课程路径
#             item['url'] = 'http://www.imooc.com' + box.xpath('.//@href').extract()[0]
#             #获取div中的课程标题
#             item['title'] = box.xpath('.//img/@alt').extract()[0].strip()
#             #获取div中的标题图片地址
#             item['image_url'] = box.xpath('.//@src').extract()[0]
#             #获取div中的学生人数
#             item['student'] = box.xpath('.//span/text()').extract()[0].strip()[:-3]
#             #获取div中的课程简介
#             item['introduction'] = box.xpath('.//p/text()').extract()[0].strip()
#             #返回信息
#             yield item


def _gen_url():
    ziru_url_list = list()
    base_url = "http://www.ziroom.com/z/nl/z2.html?qwd=%E9%A1%BA%E4%B9%89&p={}"
    for i in range(1, 31):
        ziru_url_list.append(base_url.format(i))

    return ziru_url_list


class Ziru(scrapy.Spider):
    name = "ziru"
    allowed_domains = ["ziroom.com"]
    start_urls = _gen_url()
    custom_settings = {
        # 'LOG_LEVEL': 'DEBUG',
        # 'LOG_FILE': '5688_log_{}.txt'.format(time.time()),
        "DEFAULT_REQUEST_HEADERS": {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36',
        }
    }

    def parse(self, response):
        """
        //*[@id="houseList"]/li[1]/div[2]/h3/a
        //*[@id="houseList"]/li[1]/div[3]/p[1]
        //*[@id="houseList"]/li[1]/div[3]/p[1]/span[1]
        //*[@id="houseList"]/li[1]/div[2]/div/p[1]/span[1]
        //*[@id="houseList"]/li[1]/div[2]/div/p[1]/span[2]
        //*[@id="houseList"]/li[1]/div[2]/div/p[2]/span
        :param response:
        :return:
        """
        item = HouseDetail()
        for box in response.xpath('//*[@id="houseList"]/li[@class="clearfix"]'):
            item['title'] = box.xpath(".//div[2]/h3/a/text()").extract()[0]
            item['pic_url'] = box.xpath(".//div[1]/a/img/@src").extract()[0]
            item['detail'] = {
                'size': box.xpath('.//div[2]/div/p[1]/span[1]/text()').extract()[0],
                'floor': box.xpath('.//div[2]/div/p[1]/span[2]/text()').extract()[0],
                'house_type': box.xpath('.//div[2]/div/p[1]/span[3]/text()').extract()[0],
                'addr': box.xpath('.//div[2]/div/p[2]/span/text()').extract()[0],
            }
            yield item
