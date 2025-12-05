# chunking.py
import os
import re
import glob
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_by_paragraph(text: str) -> List[str]:
    # \n * 3 -> \n * 2 로 통잃
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def clean_for_split(s: str, strip_brackets: bool = True) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if strip_brackets:
        s = re.sub(r"[<\[].*?[>\]]", "", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def build_splitter(chunk_size: int, chunk_overlap: int, seps: Optional[List[str]] = None):
    seps = seps or ["\n\n", "\n", " ", ""] # -> text 를 자를 때 기준이 되는 우선순위 구분자들 
    # \n\n->\n-> " " -> ""  
    return RecursiveCharacterTextSplitter(
        # 남아 있는 text 가 없을 떄까지 분할하기 위해서 recursivecharactertextsplitter 사용 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=seps,
    )

from .schemas import Chunk

def chunk_dir_to_list(
    src_dir: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 500,
    strip_brackets: bool = True,
    pattern: str = "*.txt",
) -> List[Chunk]:
    files = sorted(glob.glob(os.path.join(src_dir, pattern)))
    splitter = build_splitter(chunk_size, chunk_overlap)

    records: List[Chunk] = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = split_by_paragraph(content)
        cidx = 0
        for para in paragraphs:
            clean_text = clean_for_split(para, strip_brackets=strip_brackets)
            if not clean_text:
                continue
            parts = splitter.split_text(clean_text)
            final_parts = [
                re.sub(r"[<\[].*?[>\]]", "", c).strip() if strip_brackets else c.strip()
                for c in parts if c and c.strip()
            ]
            for c in final_parts:
                records.append(Chunk(
                    source_path=os.path.abspath(fpath),
                    source_name=os.path.basename(fpath),
                    chunk_index=cidx,
                    text=c,
                ))
                cidx += 1

    return records
