#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
뉴스 기사 텍스트 전처리 CLI

사용 예:
  python preprocess_news.py --src /path/news_articles --dst /path/news_articles_preprocessing
  python preprocess_news.py -s ./news_articles -d ./news_articles_clean --min-body 120
"""

import argparse
import os
import re
import glob
from typing import Tuple

# ===== 전처리 정규식 =====
RE_EMAIL         = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_MULTI_NL      = re.compile(r"\n{3,}")
RE_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
RE_REPORTER_LINE = re.compile(r"(?m)^[^\n]{0,60}\s*기자(?:\s*\(.{0,60}\))?$")
RE_CAPTION       = re.compile(r"(?m)^(.*(?:출처\s*=\s*|사진\s*[:=]|사진제공\s*[:=]).*)$")
RE_BOILER        = re.compile(r"(무단 전재 및 재배포 금지|All rights reserved\.?)", re.IGNORECASE)

def normalize_quotes(s: str) -> str:
    return s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

def normalize_ws(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\r\n?", "\n", s)
    s = RE_MULTI_SPACE.sub(" ", s)
    s = RE_MULTI_NL.sub("\n\n", s)
    return s.strip()

def parse_collected_txt(path: str) -> Tuple[str, str, str]:
    """
    수집 파일 포맷:
      제목: ...
      요약: ...
      링크: ...

      본문:
      (본문 ...)
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    def grab(label: str) -> str:
        m = re.search(rf"^{label}:\s*(.*)$", txt, re.MULTILINE)
        return m.group(1).strip() if m else ""

    body_m = re.search(r"본문:\s*(.*)\Z", txt, re.DOTALL)
    title = grab("제목")
    url   = grab("링크")
    body  = body_m.group(1).strip() if body_m else ""
    return title, url, body

def clean_body(raw: str) -> str:
    if not raw.strip():
        return ""
    s = normalize_quotes(raw)

    s = RE_CAPTION.sub("", s)

    s = re.sub(r"\(\s*" + RE_EMAIL.pattern + r"\s*\)", "", s) 
    s = RE_EMAIL.sub("", s)                                    
    s = RE_REPORTER_LINE.sub("", s)                            

    # '기자'가 들어가는 짧은 문장 라인 제거(<=200자)
    s = re.sub(r"(?m)^.{0,200}기자.{0,200}$",
               lambda m: "" if len(m.group(0)) < 200 else m.group(0),
               s)

    s = RE_BOILER.sub("", s)

    return normalize_ws(s)

def filename_from_title(title: str, maxlen: int) -> str:
    name = (title or "").strip()
    name = re.sub(r'[\\/:*?"<>|]', " ", name)  # 금지문자만 치환
    name = re.sub(r"\s{2,}", " ", name).strip()
    name = name[:maxlen].rstrip("._ ")
    return name or "article"

def unique_path(dst_dir: str, base_name: str) -> str:
    """동일 파일명이 있으면 (2), (3) ... 접미사 부여"""
    path = os.path.join(dst_dir, base_name + ".txt")
    if not os.path.exists(path):
        return path
    k = 2
    while True:
        alt = os.path.join(dst_dir, f"{base_name} ({k}).txt")
        if not os.path.exists(alt):
            return alt
        k += 1

def process_one(src_path: str, dst_dir: str, min_body: int, max_fn: int) -> Tuple[str, str, str, bool]:
    """returns: (source_file, output_file or reason, url, kept?)"""
    title, url, body = parse_collected_txt(src_path)
    cleaned = clean_body(body)
    if len(cleaned) < min_body:
        return (os.path.basename(src_path), "[SKIP:short]", url, False)

    base = filename_from_title(title, max_fn)
    dst_path = unique_path(dst_dir, base)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    return (os.path.basename(src_path), os.path.basename(dst_path), url, True)

def run(src_dir: str, dst_dir: str, min_body: int, max_fn: int, pattern: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, pattern)))
    kept = skipped = 0

    manifest_path = os.path.join(dst_dir, "manifest.tsv")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write("source_file\toutput_file\turl\n")
        for src in files:
            src_name, out_or_reason, url, ok = process_one(src, dst_dir, min_body, max_fn)
            if ok:
                kept += 1
            else:
                skipped += 1
            mf.write(f"{src_name}\t{out_or_reason}\t{url}\n")

    print(f"[완료] 입력 {len(files)}건 → 저장 {kept}건, 스킵 {skipped}건")
    print(f"[경로] 결과 폴더: {dst_dir}")
    print(f"[매니페스트] {manifest_path}")

def main():
    p = argparse.ArgumentParser(description="수집된 뉴스 텍스트(.txt) 전처리 CLI (본문만 추출 저장)")
    p.add_argument("-s", "--src", required=True, help="입력 폴더 (수집된 .txt 모음)")
    p.add_argument("-d", "--dst", required=True, help="출력 폴더 (전처리 결과 저장)")
    p.add_argument("--pattern", default="*.txt", help="입력 파일 글롭 패턴 (기본: *.txt)")
    p.add_argument("--min-body", type=int, default=80, help="본문 최소 길이 미만이면 스킵 (기본: 80)")
    p.add_argument("--max-filename", type=int, default=120, help="파일명 최대 길이 (기본: 120)")
    args = p.parse_args()

    run(args.src, args.dst, args.min_body, args.max_filename, args.pattern)

if __name__ == "__main__":
    main()
