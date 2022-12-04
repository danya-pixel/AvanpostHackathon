import asyncio
from concurrent import futures
import time
import uuid

import aiofiles as aiofiles
import aiohttp as aiohttp
import requests
import json
import urllib
from fake_headers import Headers
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from twocaptcha import TwoCaptcha
from typing import Tuple


class Size:
    def __init__(self):
        self.large = 'large'
        self.medium = 'medium'
        self.small = 'small'


class Preview:
    def __init__(self, url: str,
                 width: int,
                 height: int):
        self.url = url
        self.width = width
        self.height = height
        self.size = str(width) + '*' + str(height)


class Result:
    def __init__(self, title: Tuple[str, None],
                 description: Tuple[str, None],
                 domain: str,
                 url: str,
                 width: int,
                 height: int,
                 preview: Preview):
        self.title = title
        self.description = description
        self.domain = domain
        self.url = url
        self.width = width
        self.height = height
        self.size = str(width) + '*' + str(height)
        self.preview = preview


class YandexImage:
    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        self.size = Size()
        self.driver = webdriver.Chrome("chromedriver", chrome_options=chrome_options)
        self.headers = Headers(headers=True).generate()
        self.version = '1.0-release'
        self.about = 'Yandex Images Parser'
        self.solver = TwoCaptcha("ac0ce6e5dd8cba371df12229a4955ce9")

    async def captchaSolver(self, image):
        loop = asyncio.get_running_loop()
        with futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, lambda: self.solver.normal(image))
            return result

    async def solve_captcha(self, url):
        try:
            captcha = WebDriverWait(self.driver, 7).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        '//*[@id="root"]/div/div/form/div[2]/div/div/div[1]/input',
                    )
                )
            )
            captcha.click()
        except TimeoutException:
            pass
        # //*[@id="advanced-captcha-form"]/div/div[1]/img
        img = WebDriverWait(self.driver, 7).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="advanced-captcha-form"]/div/div[1]/img',
                )
            )
        )
        text_ques = img.get_attribute('src')
        try:
            result = self.solver.normal(text_ques)
            result = result['code']

        except Exception as e:
            print(e)
            result = input("Captcha")

        # print(text_ques)
        # solve = input("Captcha")
        solve = result
        text_area = WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="xuniq-0-1"]',
                )
            )
        )
        text_area.send_keys(solve)

        sbmt_btn = WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="advanced-captcha-form"]/div/div[3]/button[3]',
                )
            )
        )
        sbmt_btn.click()

    async def search(self, pagenumber, text, queue: asyncio.Queue, sizes: Size = 'large') -> int:
        captcha = True
        while captcha:
            params = {"text": text,
                      "p": pagenumber,
                      "nomisspell": 1,
                      "noreask": 1,
                      "isize": sizes}
            safe_string = urllib.parse.urlencode(params)
            # r = requests.get(f"https://yandex.ru/images/search?{safe_string}", headers=self.headers)
            self.driver.get(f"https://yandex.ru/images/search?{safe_string}")
            time.sleep(0.3)
            url = self.driver.current_url
            if 'showcaptcha' in url:
                await self.solve_captcha(url)
            else:
                captcha = False
        body = WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located(
                (
                    By.TAG_NAME,
                    'body',
                )
            )
        )

        find = False
        cur_items_cnt = 0
        tries = 0
        while True:
            for i in range(10):
                body.send_keys(Keys.END)
                await asyncio.sleep(0.5)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            items_place = soup.find('div', {"class": "serp-list"})
            try:
                items = items_place.find_all("div", {"class": "serp-item"})
            except AttributeError as e:
                continue

            for item in items[cur_items_cnt:]:
                data = json.loads(item.get("data-bem"))
                image = data['serp-item']['img_href']
                # image_width = data['serp-item']['preview'][0]['w']
                # image_height = data['serp-item']['preview'][0]['h']

                # snippet = data['serp-item']['snippet']
                # try:
                #     title = snippet['title']
                # except KeyError:
                #     title = None
                # try:
                #     description = snippet['text']
                # except KeyError:
                #     description = None
                # domain = snippet['domain']
                #
                # preview = 'https:' + data['serp-item']['thumb']['url']
                # preview_width = data['serp-item']['thumb']['size']['width']
                # preview_height = data['serp-item']['thumb']['size']['height']
                await queue.put(image)
                # output.append(Result(title, description, domain, image,
                #                      image_width, image_height,
                #                      Preview(preview, preview_width, preview_height)))
            if cur_items_cnt == len(items):
                tries += 1
            else:
                tries = 0
            cur_items_cnt = len(items)
            print(cur_items_cnt)
            if cur_items_cnt > 800:
                return cur_items_cnt
            try:
                end = WebDriverWait(self.driver, 2).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            '/html/body/div[3]/div[2]/div[1]/div[2]/a',
                        )
                    )
                )
                end.click()
                await asyncio.sleep(5)
                if find:
                    find = True
                    break
                else:
                    find = True
            except Exception as e:
                if tries > 6:
                    break
                find = False
                pass

        #
        # soup = BeautifulSoup(self.driver.page_source, "html.parser")
        #
        # items_place = soup.find('div', {"class": "serp-list"})
        # output = list()
        # try:
        #     items = items_place.find_all("div", {"class": "serp-item"})
        # except AttributeError as e:
        #     print(e)
        #     return output
        #
        # for item in items:
        #     data = json.loads(item.get("data-bem"))
        #     image = data['serp-item']['img_href']
        #     image_width = data['serp-item']['preview'][0]['w']
        #     image_height = data['serp-item']['preview'][0]['h']
        #
        #     snippet = data['serp-item']['snippet']
        #     try:
        #         title = snippet['title']
        #     except KeyError:
        #         title = None
        #     try:
        #         description = snippet['text']
        #     except KeyError:
        #         description = None
        #     domain = snippet['domain']
        #
        #     preview = 'https:' + data['serp-item']['thumb']['url']
        #     preview_width = data['serp-item']['thumb']['size']['width']
        #     preview_height = data['serp-item']['thumb']['size']['height']
        #
        #     output.append(Result(title, description, domain, image,
        #                          image_width, image_height,
        #                          Preview(preview, preview_width, preview_height)))

        return 0


async def get_hrefs(text, queue):
    parser = YandexImage()
    await parser.search(0, text, queue)
    # for j in a:
    #     await queue.put(j.url)
    await asyncio.sleep(1)


async def download_image(queue: asyncio.Queue):
    while True:
        url = await queue.get()
        try:
            async with aiohttp.ClientSession(read_timeout=2) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        f = await aiofiles.open('train/' + str(uuid.uuid4()) + ".jpg", mode='wb')
                        payload = await response.read()
                        await f.write(payload)
                        await f.close()
            print(url, "Ready", queue.qsize())
        except AttributeError as e:
            pass
            # print(e)
        except Exception as e:
            # pass
            # print(e, type(e))
            pass
        finally:
            queue.task_done()


async def main(request):
    start = time.time()
    queue = asyncio.Queue()
    hrefs = asyncio.create_task(get_hrefs(request,queue))
    tasks = []
    for _ in range(5):
        task = asyncio.create_task(download_image(queue))
        tasks.append(task)
    total = await asyncio.gather(hrefs, return_exceptions=True)
    print(total)
    await queue.join()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print(time.time() - start)


if __name__ == '__main__':
    asyncio.run(main("asdasd"))
    # images = []
    # parser = YandexImage()
    # for i in range(40):
    #     a = parser.search(i, "Микроавтобус")
    #     images.extend([j.url for j in a])
    #     time.sleep(0.3)
    # with open('trains.json', 'w') as f:
    #     json.dump(images, f)
