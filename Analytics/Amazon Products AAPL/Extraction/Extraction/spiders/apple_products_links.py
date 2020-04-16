# -*- coding: utf-8 -*-
 
# Importing Scrapy Library
import scrapy
import re
 
# Creating a new class to implement Spide
class AmazonReviewsSpider(scrapy.Spider):
     
    # Spider name
    name = 'aapl_links'
     
    # Domain names to scrape
    allowed_domains = ['amazon.in']
     
    # Base URL for the MacBook air reviews
    start_urls = ["https://www.amazon.com/s?k=Apple&page=1"]
    #start_urls=[]
    
    # Creating list of urls to be scraped by appending page number a the end of base url
    #for i in range(1,121):
    #    start_urls.append(myBaseUrl+str(i))
    
    # Defining a Scrapy parser
    def parse(self, response):
            data = response.css('.s-result-list')

            products = data.css('.s-result-item')
            htmls = products.css('.s-line-clamp-2 a::attr(href)').getall()
             
            # Combining the results
            for html in htmls:
                slash_indices = [m.span() for m in re.finditer("/", html)]
                # eg.'/Apple-AirPods-Charging-Latest-Model/dp/B07PXGQC1Q'
                # [(0, 1), (36, 37), (39, 40)]
                product_name = html[slash_indices[0][1]:slash_indices[1][0]]
                product_id = html[slash_indices[2][1]:]

                yield{'link':html,
                      'product_name':product_name,
                      'product_id':product_id
                     }
            next_page = response.css('li.a-last a::attr(href)').get()
            if next_page is not None:
                next_page = response.urljoin(next_page)
                yield scrapy.Request(next_page, callback=self.parse, dont_filter=True)
