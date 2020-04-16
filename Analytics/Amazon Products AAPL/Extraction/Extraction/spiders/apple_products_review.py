
# -*- coding: utf-8 -*-
 
# Importing Scrapy Library
import scrapy
import pandas as pd
import os
# Creating a new class to implement Spide
class AmazonReviewsSpider1(scrapy.Spider):
     
    # Spider name
    name = 'AAPL_products_reviews'
     
    # Domain names to scrape
    allowed_domains = ['amazon.com']


    # Defining a Scrapy parser
    def start_requests(self):
        all_links = pd.read_csv("aapl_links.csv")

        for index, row in all_links.iterrows():
            url = "https://www.amazon.com/" + row["product_name"] + "/product-reviews/" + row['product_id'] + '/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
            data = response.css('#cm_cr-review_list')
            data2 = response.css('#cm_cr-product_info')

            product_name = data2.css('.product-title')
            # Collecting product star ratings
            star_rating = data.css('.review-rating')

            #Collecting review date
            date = data.css('.review-date')
             
            # Collecting user reviews
            comments = data.css('.review-text')

            #url_link  = response.request.url
            #new_link = url_link.split('/')
            #product_name = new_link[3]
            count = 0
             
            # Combining the results
            for review in star_rating:
                yield{'stars': ''.join(review.xpath('.//text()').extract()),
                        'date': ''.join(date[count].xpath('.//text()').extract()),
                      'comment': ''.join(comments[count].xpath(".//text()").extract()).strip(),
                      'product name': ''.join(product_name.xpath(".//text()").extract())
                      
                     }
                count=count+1
            next_page = response.css('li.a-last a::attr(href)').get()
            if next_page is not None:
                next_page = response.urljoin(next_page)
                yield scrapy.Request(next_page, callback=self.parse, dont_filter=True)

#scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/apple_products_reviews.py -o apple_products_review.csv
