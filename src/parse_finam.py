import os
import shutil

from typing import List
import datetime
from configparser import ConfigParser

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import time

month_dct = {0: 'январь', 1: 'февраль', 2: 'март', 3: 'апрель', 4: 'май', 5: 'июнь', 6: 'июль', 7: 'август', 8: 'сентябрь', 9: 'октябрь',
             10: 'ноябрь', 11: 'декабрь'}

def scroll_calendar(driver: webdriver.Chrome, action: ActionChains, key_month: int, key_year: int, first: bool):
    while True:
        month = driver.find_element(By.XPATH, '//p[@class="ui-kit-Body14-ui-kit-18ajdke cApxOD ui-kit-Month-ui-kit-b7tvtb iSnnyt ui-calendar__month weightMedium"]').text.lower()
        year = int(driver.find_element(By.XPATH, '//input[@class="ui-kit-YearInput-ui-kit-x3evgy jKzXzQ ui-calendar__year"]').get_attribute('value'))
        if (month == month_dct[key_month] and key_year == year):
            break
        button = driver.find_element(By.XPATH, '//button[@class="ui-kit-Root-ui-kit-1dybyb0 hOoKPm ui-calendar__button-left"]')
        action.click(button)
        action.perform()
        time.sleep(2)

    if first:
        start_date_element = driver.find_elements(By.XPATH, '//button[@class="ui-kit-Root-ui-kit-1dybyb0 hOoKPm ui-kit-Root-ui-kit-1nxmom8 gSfMwq"]')[0]
    else:
        start_date_element = driver.find_elements(By.XPATH, '//button[@class="ui-kit-Root-ui-kit-1dybyb0 hOoKPm ui-kit-Root-ui-kit-1nxmom8 gSfMwq"]')[-1]

    action.click(start_date_element)
    action.perform()

def parse_finam(link: str, start: datetime.date, end: datetime.date):
    driver = webdriver.Chrome()
    driver.get(link)
    driver.execute_script("document.body.style.zoom='80%'")
    time.sleep(2)
    action = ActionChains(driver)

    # Переодичность
    ticks = driver.find_element(by=By.XPATH, value='//p[@class="ui-kit-Body14-ui-kit-18ajdke cApxOD ui-kit-DisplayValue-ui-kit-w5feoh ikFxEs ui-select-field__value"]')
    action.click(ticks)
    action.perform()
    time.sleep(1)
    min_1 = driver.find_element(by=By.XPATH, value='//li[contains(text(), "1 мин.")]')
    action.click(min_1)
    action.perform()
    time.sleep(0.5)

    # Тип файла
    dtype = driver.find_element(by=By.XPATH, value='//p[contains(text(), "Тип файла")]')
    action.click(dtype)
    action.perform()
    time.sleep(1)
    csv = driver.find_element(by=By.XPATH, value='//li[contains(text(), ".csv")]')
    action.click(csv)
    action.perform()

    flag = True
    for year in range(end.year, start.year-1, -1):
        if flag:
            months = [0, 3, 6, 9]
            months = [m for m in months if m <= end.month-1]
        else:
            months = [0, 3, 6, 9]

        for month in months[::-1]:

            # Дата начала парсинга
            from_window = driver.find_element(by=By.NAME, value='from')
            action.click(from_window)
            action.perform()
            time.sleep(1)

            scroll_calendar(driver, action, month, year, True)

            # Дата конца парсинга
            to_window = driver.find_element(by=By.NAME, value='to')
            action.click(to_window)
            action.perform()
            time.sleep(1)

            if flag:
                end_month = end.month - 1
                flag = False
            else:
                end_month = month + 2
            
            scroll_calendar(driver, action, end_month, year, False)

            get_file = driver.find_element(by=By.XPATH, value='//button[@class="ui-kit-BaseButton-ui-kit-ekw2d7 ui-kit-PrimaryButton-ui-kit-1j75dgi bgmPVI dzTiIc sizeMedium colorPrimary"]')
            action.click(get_file)
            action.perform()
            time.sleep(5)
    time.sleep(5)
    driver.quit()
    

def get_all_quotes(companies: List[str], start: datetime.date, end: datetime.date):
    for company in companies:
        try:
            link = f'https://www.finam.ru/quote/moex/{company.lower()}/export/'
            parse_finam(link, start, end)
            print(f'{company} parsed')
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            print(ex)

def move_all_to_folder(companies, root):
    files = [el for el in os.listdir('C:/Users/boris/Downloads') if el.startswith(tuple(companies))]
    for file in files:
        shutil.move(f'C:/Users/boris/Downloads/{file}', f'{root}/{file}')

if __name__ == '__main__':
    config = ConfigParser()
    config.read('configs/finam.ini')
    start = list(map(int, config['GENERAL']['start'].split('.')))
    start = datetime.date(start[-1], start[1], start[0])
    end = list(map(int, config['GENERAL']['end'].split('.')))
    end = datetime.date(end[-1], end[1], end[0])

    get_all_quotes(eval(config['GENERAL']['companies']), start, end)
    move_all_to_folder(eval(config['GENERAL']['companies']), config['GENERAL']['save_root'])
