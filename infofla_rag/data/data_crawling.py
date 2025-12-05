from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib3.exceptions import ReadTimeoutError

import sys
import os
import re
import time
import atexit

class DualLogger:
    def __init__(self, filename: str):
        self._orig_stdout = sys.stdout
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")
        self._closed = False

    def write(self, message: str):
        try:
            self._orig_stdout.write(message)
        except Exception:
            pass
        if not self._closed and getattr(self, "log", None) and not self.log.closed:
            try:
                self.log.write(message)
            except Exception:
                pass

    def flush(self):
        try:
            self._orig_stdout.flush()
        except Exception:
            pass
        if not self._closed and getattr(self, "log", None) and not self.log.closed:
            try:
                self.log.flush()
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if sys.stdout is self:
                sys.stdout = self._orig_stdout
        except Exception:
            pass
        try:
            if getattr(self, "log", None) and not self.log.closed:
                try:
                    self.log.flush()
                except Exception:
                    pass
                self.log.close()
        except Exception:
            pass
        self.log = None


logger = DualLogger("news_output.txt")
sys.stdout = logger
atexit.register(logger.close) 


def slugify(text, maxlen=80):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = text[:maxlen].rstrip("._ ")
    return text or "article"


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--window-size=1920,1080")

    options.binary_location = "/usr/bin/chromium-browser"

    driver = webdriver.Chrome(
        service=Service("/usr/bin/chromedriver"),
        options=options,
    )
    return driver



def wait_dom_ready(driver, timeout=15):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
    )

def safe_get(driver, url, timeout=30, retries=1, sleep_between=1.0):

    last_err = None
    for attempt in range(retries + 1):
        try:
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            wait_dom_ready(driver, min(12, timeout))
            return True
        except (TimeoutException, ReadTimeoutError, WebDriverException) as e:
            last_err = e
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass
            if attempt < retries:
                time.sleep(sleep_between)
                continue
            else:
                return False, last_err
    return False, last_err

def extract_body_text(driver, soft_timeout_sec=10):

    t0 = time.time()
    selectors = [
        'div[class*="article-body"]',
        'section[class*="news_view"]',
        'article',
        'div[itemprop="articleBody"]',
        'div[id*="content"]',
        'main'
    ]
    for sel in selectors:
        if time.time() - t0 > soft_timeout_sec:
            break
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            text = el.text.strip()
            if text and len(text) > 60:
                return text
        except Exception:
            continue

    try:
        if time.time() - t0 <= soft_timeout_sec:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            for sel in ["article", "main", 'div[itemprop="articleBody"]', 'div[id*="content"]', 'section']:
                if time.time() - t0 > soft_timeout_sec:
                    break
                node = soup.select_one(sel)
                if node:
                    txt = node.get_text("\n", strip=True)
                    if txt and len(txt) > 60:
                        return txt
            txt = soup.get_text("\n", strip=True)
            return txt[:5000] if txt else ""
    except Exception:
        pass
    return ""

def main():
    HOME = "https://infofla.com/ko/home"
    SAVE_DIR = "news_articles"
    os.makedirs(SAVE_DIR, exist_ok=True)

    driver = setup_driver()
    wait = WebDriverWait(driver, 15)
    ITEM_MAX_SEC = 40  # 필요시 60~90으로 조정

    try:
        ok = safe_get(driver, HOME, timeout=25, retries=1)
        if ok is not True:
            success, err = ok  # (False, 예외)
            print(f"[오류] 홈 이동 실패: {err}")
            return

        title_sel = 'h3[class^="News_articleTitle__"]'
        desc_sel  = 'p[class^="News_articleDescription__"]'
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, title_sel)))

        titles = driver.find_elements(By.CSS_SELECTOR, title_sel)
        descs  = driver.find_elements(By.CSS_SELECTOR, desc_sel)

        items = []
        for i, t in enumerate(titles):
            try:
                a_tag = t.find_element(By.TAG_NAME, "a")
                title_text = (a_tag.text or "").strip()
                href = a_tag.get_attribute("href") or ""
                link = urljoin(HOME, href) if href else None
                desc_text = (descs[i].text.strip() if i < len(descs) else "")
                if link:
                    items.append({"title": title_text, "desc": desc_text, "link": link})
            except Exception:
                continue

        print(f"[정보] 수집할 기사 수: {len(items)}")

        for idx, it in enumerate(items, 1):
            item_start = time.time()
            print(f"\n[{idx}/{len(items)}] 📰 제목: {it['title']}")
            print(f"💬 요약: {it['desc']}")
            print(f"🔗 링크: {it['link']}")

            # 남은 시간에 맞춰 safe_get 타임아웃을 동적으로 축소
            remaining = ITEM_MAX_SEC - (time.time() - item_start)
            if remaining <= 0:
                print("[스킵] 아이템 시작 직후 하드 타임아웃 소진")
                continue

            # 페이지 로드 타임아웃은 남은 시간의 60~70%로 제한
            nav_timeout = max(8, int(remaining * 0.65))
            ok = safe_get(driver, it["link"], timeout=nav_timeout, retries=1)
            if ok is not True:
                print(f"[경고] 본문 페이지 접속 실패. 원인: {ok}")
                # 다음 기사로 진행하기 전에 정리
                try:
                    driver.execute_script("window.stop();")
                    driver.get("about:blank")
                except Exception:
                    pass
                continue

            # 본문 추출 (소프트 타임아웃: 남은 시간의 일부)
            remaining = ITEM_MAX_SEC - (time.time() - item_start)
            if remaining <= 0:
                print("[스킵] 네비게이션 후 하드 타임아웃 소진")
                try:
                    driver.execute_script("window.stop();")
                    driver.get("about:blank")
                except Exception:
                    pass
                continue

            soft_extract = max(5, int(min(12, remaining * 0.6)))
            body = extract_body_text(driver, soft_timeout_sec=soft_extract)

            # 하드 타임아웃 체크
            elapsed = time.time() - item_start
            if elapsed > ITEM_MAX_SEC:
                print(f"[스킵] 본문 추출 중 하드 타임아웃 초과({elapsed:.1f}s > {ITEM_MAX_SEC}s)")
                try:
                    driver.execute_script("window.stop();")
                    driver.get("about:blank")
                except Exception:
                    pass
                continue

            preview = body[:500] + ("..." if len(body) > 500 else "")
            print(f"⏱ 처리시간: {elapsed:.1f}s")
            print("📄 본문 미리보기:")
            print(preview if preview else "본문을 찾을 수 없습니다.")

            # 저장
            fname = f"{slugify(it['title'])}.txt"
            fpath = os.path.join(SAVE_DIR, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"제목: {it['title']}\n")
                f.write(f"요약: {it['desc']}\n")
                f.write(f"링크: {it['link']}\n\n")
                f.write("본문:\n")
                f.write(body if body else "본문을 찾을 수 없습니다.")
            print(f"💾 저장 완료: {fpath}")

            # 다음 기사에 영향 없게 정리
            try:
                driver.execute_script("window.stop();")
                driver.get("about:blank")
            except Exception:
                pass

    finally:
        try:
            driver.quit()
        except Exception:
            pass
        # 안전 종료: stdout 원복 → 파일 닫기
        try:
            if isinstance(sys.stdout, DualLogger):
                sys.stdout.close()
            else:
                try:
                    logger.close()
                except Exception:
                    pass
        except Exception:
            pass


if __name__ == "__main__":
    main()
