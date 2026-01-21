#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm  # ✅ 进度条支持

Record = Dict[str, Any]

# ======================
# 基础工具
# ======================
import re
try:
    import jieba
    _JIEBA_OK = True
except Exception:
    _JIEBA_OK = False

# 汉字范围：CJK统一表意 + 扩展A + 兼容表意 + 扩展B-G（到 2EBEF）
_CJK_HAN_ONE = re.compile(r'[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[\U00020000-\U0002EBEF]')
# 连续汉字序列（用于无jieba时的退化分词）
_CJK_HAN_SEQ = re.compile(r'(?:[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[\U00020000-\U0002EBEF])+')

def _has_han(token: str) -> bool:
    """token中是否包含至少一个汉字"""
    return bool(_CJK_HAN_ONE.search(token))

def _tokenize_zh(text: str):
    """
    中文分词：
    - 优先使用 jieba.lcut
    - 若未安装 jieba，则退化为按“连续汉字序列”切分
    - 仅保留“含汉字”的token（过滤纯标点/纯英文/纯数字等）
    """
    if not text:
        return []

    if _JIEBA_OK:
        # HINT: 如果你有自定义词典，可在外层调用 jieba.load_userdict(...)
        toks = jieba.lcut(text, cut_all=False)
        toks = [t.strip() for t in toks if t.strip()]
        toks = [t for t in toks if _has_han(t)]
        return toks
    else:
        # 退化策略：把连续汉字片段当作“词”
        return _CJK_HAN_SEQ.findall(text)

# 你原来的统计“汉字个数”的函数可以保留（有需要时复用）
def _count_chinese_han(text: str) -> int:
    if not text:
        return 0
    return sum(1 for _ in _CJK_HAN_ONE.finditer(text))

# ===== 核心：将 fact_len 改为“中文分词后的长度” =====
def fact_len(rec: Record) -> int:
    text = rec.get("fact", "") or ""
    if not isinstance(text, str):
        text = str(text)
    # tokens = _tokenize_zh(text)
    # return len(tokens)
    return _count_chinese_han(text)

def normalize_accusation_text(acc: str) -> str:
    return (str(acc) if acc is not None else "").strip().replace(" ", "").replace("　", "")

def to_list_str(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]

def normalize_record_schema(rec: Dict[str, Any]) -> Record:
    fact = rec.get("fact") or rec.get("Fact") or rec.get("facts") or ""
    meta = rec.get("meta", {})
    acc_src = meta.get("accusation", rec.get("accusation"))
    arts_src = meta.get("relevant_articles", rec.get("relevant_articles", rec.get("articles", rec.get("article"))))
    term_src = meta.get("imprisonment", rec.get("imprisonment"))

    meta_out = dict(meta) if isinstance(meta, dict) else {}
    meta_out["accusation"] = [normalize_accusation_text(a) for a in to_list_str(acc_src)]
    meta_out["relevant_articles"] = [str(a).strip() for a in to_list_str(arts_src)]
    meta_out["imprisonment"] = term_src
    meta_out["death_penalty"] = meta.get("death_penalty", rec.get("death_penalty"))
    meta_out["life_imprisonment"] = meta.get("life_imprisonment", rec.get("life_imprisonment"))

    return {
        "fact": str(fact),
        "meta": meta_out
    }

def extract_accusations(rec: Record) -> List[str]:
    acc = rec.get("meta", {}).get("accusation", [])
    if isinstance(acc, list):
        return [normalize_accusation_text(x) for x in acc if str(x).strip()]
    if isinstance(acc, (str, int, float)):
        return [normalize_accusation_text(acc)]
    return []

def extract_articles(rec: Record) -> List[str]:
    arts = rec.get("meta", {}).get("relevant_articles", [])
    if isinstance(arts, list):
        return [str(x).strip() for x in arts if str(x).strip()]
    if isinstance(arts, (str, int, float)):
        return [str(arts).strip()]
    return []

# ======================
# 多线程 + tqdm
# ======================

def normalize_records_parallel(records: List[Record], workers: int = None) -> List[Record]:
    """多线程规范化，带 tqdm 进度条"""
    if not records:
        return []
    workers = workers or min(32, (os.cpu_count() or 4) * 2)
    print(f"开始多线程规范化：共 {len(records)} 条，用 {workers} 个线程")

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(normalize_record_schema, records, chunksize=2000), total=len(records), desc="规范化中"):
            out.append(r)
    return out

# ======================
# 统计与筛选
# ======================

def compute_basic_stats(records: List[Record]) -> Dict[str, Any]:
    art_set, acc_set = set(), set()
    total_len = 0
    for r in records:
        art_set.update(extract_articles(r))
        acc_set.update(extract_accusations(r))
        total_len += fact_len(r)
    n = len(records)
    return {
        "total_records": n,
        "unique_articles": len(art_set),
        "unique_accusations": len(acc_set),
        "avg_fact_len": (total_len / n) if n else 0.0
    }

def print_stats(title: str, records: List[Record], topn: int = 10):
    st = compute_basic_stats(records)
    print(f"\n—— {title} ——")
    print(f"总记录数：{st['total_records']}")
    print(f"法条总数（去重）：{st['unique_articles']}")
    print(f"罪名总数（去重）：{st['unique_accusations']}")
    print(f"Fact 平均字符长度：{st['avg_fact_len']:.2f}")

    if topn and records:
        acc_counter, art_counter = Counter(), Counter()
        for r in records:
            acc_counter.update(extract_accusations(r))
            art_counter.update(extract_articles(r))
        print(f"Top{topn} 罪名：{acc_counter.most_common(topn)}")
        print(f"Top{topn} 法条：{art_counter.most_common(topn)}")

def count_accusations(records: List[Record]) -> Counter:
    c = Counter()
    for r in records:
        c.update(extract_accusations(r))
    return c

import random

def sample_records(records: List[Record], sample_size: int = 82138, seed: int = 42) -> List[Record]:
    """从 records 中随机抽取 sample_size 条记录（不放回抽样）"""
    if len(records) < sample_size:
        raise ValueError(f"合法记录数量不足（{len(records)} 条），无法抽取 {sample_size} 条")
    
    random.seed(seed)  # 保证可复现性
    return random.sample(records, sample_size)

def filter_records(dataset: List[Record], min_fact_len: int, allowed_accusations: set) -> Tuple[List[Record], Counter]:
    """只保留 fact 长度合适、仅一个罪名&法条、且罪名在 allowlist 中的记录"""
    acc_counter = Counter()
    kept: List[Record] = []

    for rec in tqdm(dataset, desc="筛选中"):
        if fact_len(rec) < min_fact_len:
            continue

        death_penalty = rec.get("meta", {}).get("death_penalty", False)
        life_imprisonment = rec.get("meta", {}).get("life_imprisonment", False)
        imprisonment = rec.get("meta", {}).get("imprisonment", None)

        if imprisonment == 0 or death_penalty or life_imprisonment:
            continue

        accs = [a for a in extract_accusations(rec) if a]
        arts = [a for a in extract_articles(rec) if a]

        # if len(accs) != 1 or len(arts) != 1:
        #     continue
        if len(arts) != 1:
            continue
        if len(accs) != 1:
            continue

        acc = accs[0]
        if acc not in allowed_accusations:
            continue

        acc_counter[acc] += 1
        kept.append(rec)
    
    print("采样前长度：", len(kept))
    sampled_records = sample_records(kept, sample_size=82138)
    acc_counter = Counter()
    for rec in sampled_records:
        accs = extract_accusations(rec)
        if accs:
            acc_counter[accs[0]] += 1
    
    return sampled_records, acc_counter

# ======================
# 加载 CAIL2018
# ======================

def try_load_cail2018() -> List[Record]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "请先安装 `datasets`：pip install datasets -U"
        ) from e

    print("加载数据集 china-ai-law-challenge/cail2018 ...")
    dataset = load_dataset("china-ai-law-challenge/cail2018")

    candidate_splits = [
        "exercise_contest_train",
        "exercise_contest_valid",
        "exercise_contest_test",
        "final_test",
        # "first_stage_train",
        # "first_stage_test",
    ]

    merged_raw: List[Record] = []
    for name in candidate_splits:
        if name in dataset:
            ds = dataset[name]
            print(f"  加载 {name}（{len(ds)} 条）")
            for rec in tqdm(ds, desc=f"  读取 {name}"):
                merged_raw.append(rec)
        else:
            print(f"  [跳过] 无切分：{name}")
    print(f"合并后总记录数：{len(merged_raw)}")
    return merged_raw

# ======================
# 主流程
# ======================

def save_jsonl(records: List[Record], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split_train_val_test(records: List[dict], seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """将记录随机打乱，并按 8:1:1 分为 train / val / test"""
    random.seed(seed)
    random.shuffle(records)

    total = len(records)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test

def main():
    ALLOWED_ACCUSATIONS = {
        "妨害信用卡管理", "非国家工作人员受贿", "单位行贿", "交通肇事", "挪用公款", "非法捕捞水产品", "赌博",
        "非法种植毒品原植物", "盗伐林木", "过失致人死亡", "抢劫", "妨害公务", "非法吸收公众存款", "盗窃",
        "重大责任事故", "非法占用农用地", "容留他人吸毒", "串通投标", "故意伤害", "开设赌场", "行贿",
        "拒不支付劳动报酬", "强奸", "失火", "滥伐林木", "故意杀人", "危险驾驶", "受贿", "非法拘禁",
        "非法侵入住宅", "诈骗", "放火", "合同诈骗", "挪用资金", "销售假冒注册商标的商品", "污染环境",
        "贪污", "非法采矿", "聚众斗殴", "故意毁坏财物", "职务侵占", "非法狩猎"
    }
    parser = argparse.ArgumentParser(
        description="只依赖 cail2018，打印筛选前/后数据统计，支持多线程 + tqdm 进度条"
    )
    parser.add_argument("--min_fact_len", type=int, default=0, help="事实最小长度（默认：20）")
    # parser.add_argument("--min_acc_count", type=int, default=50, help="罪名在全库出现的最小次数（默认：50）")
    parser.add_argument("--workers", type=int, default=None, help="线程数（默认：2×CPU，最多 32）")
    parser.add_argument("--output_dir", default="./dataset")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    parser.add_argument("--topn", type=int, default=10, help="打印 TopN 罪名/法条（默认：10，设 0 关闭）")
    args = parser.parse_args()

    raw_records = try_load_cail2018()
    workers = args.workers or min(32, (os.cpu_count() or 4) * 2)

    merged = normalize_records_parallel(raw_records, workers=workers)

    print_stats("数据概览（筛选前）", merged, topn=args.topn)

    kept, acc_counter = filter_records(merged, args.min_fact_len, ALLOWED_ACCUSATIONS)

    print_stats("筛选后数据概览", kept, topn=args.topn)

    train, val, test = split_train_val_test(kept, seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))

    print(f"\n✅ 已保存 train/val/test 文件至目录：{args.output_dir}")
    print(f"    - train.jsonl: {len(train)} 条")
    print(f"    - val.jsonl:   {len(val)} 条")
    print(f"    - test.jsonl:  {len(test)} 条")

if __name__ == "__main__":
    main()
