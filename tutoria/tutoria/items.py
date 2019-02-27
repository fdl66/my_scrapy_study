# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class TutoriaItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class CourseItem(scrapy.Item):
    #课程标题
    title = scrapy.Field()
    #课程url
    url = scrapy.Field()
    #课程标题图片
    image_url = scrapy.Field()
    #课程描述
    introduction = scrapy.Field()
    #学习人数
    student = scrapy.Field()


class HouseDetail(scrapy.Item):
    title = scrapy.Field()
    price = scrapy.Field()
    pic_url = scrapy.Field()
    detail = scrapy.Field()
    