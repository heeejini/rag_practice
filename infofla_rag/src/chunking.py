import os
import re
import glob
import json
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
    pattern: str = "*", # json, txt 모두 허용하기 위해 와일드카드 변경 가능, 일단 내부에서 필터링
) -> List[Chunk]:
    # 패턴이 *.txt로 들어오면 txt만 보겠지만,
    # 만약 사용자가 패턴을 안주거나 *.*로 주면 다 볼 수 있게
    # 여기서는 glob 패턴을 그대로 따르되 확장자별 분기 처리
    
    # 1) 파일 목록 조회
    files = sorted(glob.glob(os.path.join(src_dir, pattern)))
    splitter = build_splitter(chunk_size, chunk_overlap)

    records: List[Chunk] = []

    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        
        # (A) JSON 처리
        if ext == ".json":
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # JSON이 list 형태일 수도 있고 dict 형태일 수도 있음
                # 1. List[Dict] 가정
                if isinstance(data, list):
                    items = data
                # 2. Dict 가정 -> 단일 문서 or 'documents' key안에 리스트
                elif isinstance(data, dict):
                    # 만약 "items"나 "documents" 등의 키가 있으면 그 안을 순회
                    # 없다면 단일 객체로 취급 
                    # 여기서는 간단히 단일 객체 리스트화
                    items = [data]
                else:
                    items = []
                
                for item in items:
                    if not isinstance(item, dict):
                        continue
                        
                    # 텍스트 필드 찾기
                    # 후보군: text, content, body, description, article
                    text_candidate = ""
                    for key in ["text", "content", "body", "desc", "description", "article"]:
                        if key in item and isinstance(item[key], str):
                            text_candidate = item[key]
                            break
                    
                    if not text_candidate:
                        continue
                        
                    # 메타데이터: 텍스트 제외한 나머지
                    meta = {k: v for k, v in item.items() if k not in ["text", "content", "body", "desc", "description", "article"] and isinstance(v, (str, int, float, bool))}
                    
                    # 청킹
                    chunks_text = process_text(text_candidate, splitter, strip_brackets)
                    
                    for i, c_txt in enumerate(chunks_text):
                        records.append(Chunk(
                            source_path=os.path.abspath(fpath),
                            source_name=os.path.basename(fpath),
                            chunk_index=i,
                            text=c_txt,
                            metadata=meta
                        ))

            except Exception as e:
                print(f"[ERROR] Failed to parse JSON {fpath}: {e}")
                continue
        
        # JSONL 처리 
        elif ext == ".jsonl":
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception as e:
                            print(f"[ERROR] Failed to parse JSONL line {line_idx} in {fpath}: {e}")
                            continue

                        if not isinstance(item, dict):
                            continue

                        # text 후보: body → text → content 순
                        text_candidate = (
                            item.get("body")
                            or item.get("text")
                            or item.get("content")
                            or ""
                        )
                        if not isinstance(text_candidate, str) or not text_candidate.strip():
                            continue

                        # 메타데이터: title, summary, link + 기타 단순 필드
                        meta: Dict[str, object] = {}

                        for key in ["title", "summary", "link"]:
                            val = item.get(key)
                            if isinstance(val, str):
                                meta[key] = val

                        # 그 외 단순 스칼라 필드도 추가 (bool/int/float/str)
                        for k, v in item.items():
                            if k in ["body", "text", "content"]:
                                continue
                            if k in meta:
                                continue
                            if isinstance(v, (str, int, float, bool)):
                                meta[k] = v

                        # 라인 번호도 참고용으로 넣어두기
                        meta.setdefault("record_index", line_idx)

                        chunks_text = process_text(text_candidate, splitter, strip_brackets)

                        for i, c_txt in enumerate(chunks_text):
                            records.append(
                                Chunk(
                                    source_path=os.path.abspath(fpath),
                                    source_name=os.path.basename(fpath),
                                    chunk_index=i,
                                    text=c_txt,
                                    metadata=meta,
                                )
                            )
            except Exception as e:
                print(f"[ERROR] Failed to read JSONL {fpath}: {e}")
                continue
        
        elif ext == ".txt":
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks_text = process_text(content, splitter, strip_brackets)
            for i, c_txt in enumerate(chunks_text):
                records.append(Chunk(
                    source_path=os.path.abspath(fpath),
                    source_name=os.path.basename(fpath),
                    chunk_index=i,
                    text=c_txt,
                    # txt 파일은 별도 메타데이터가 파일 내부에 구조적으로 없으므로 빈 딕셔너리
                    metadata={} 
                ))
        
        # (C) 그 외 무시
        else:
            continue

    return records


def process_text(raw_text: str, splitter, strip_brackets: bool) -> List[str]:
    """
    공통 텍스트 전처리 및 스플릿 로직 분리
    """
    paragraphs = split_by_paragraph(raw_text)
    results = []
    
    for para in paragraphs:
        clean_text = clean_for_split(para, strip_brackets=strip_brackets)
        if not clean_text:
            continue
        parts = splitter.split_text(clean_text)
        final_parts = [
            re.sub(r"[<\[].*?[>\]]", "", c).strip() if strip_brackets else c.strip()
            for c in parts if c and c.strip()
        ]
        results.extend(final_parts)
    
    return results
