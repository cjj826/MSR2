import tiktoken
import os
import argparse
import os
from typing import Dict, Any, Optional
import numpy as np
from datasets import load_dataset, Dataset
import tiktoken
import argparse
from tqdm import *

def make_prefix(dp, template_type):
    fact = dp['fact']
    if template_type == 'base':
        """This works for any base model"""
#         prefix = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
        prefix = f"""你是一名中国刑事量刑助手，请阅读以下材料描述的案情事实并回答问题：
【案情事实开始】
 {fact}
【案情事实结束】

【任务目标】
根据案情事实，预测被告人的宣告刑期，并仅按指定格式输出。

请根据以下量刑流程进行推理，坚持**定性为主、定量为辅**原则，依次确定量刑起点、基准刑和宣告刑。
在推理过程中，如果你认为需要检索外部知识，可以发起一个检索服务。
## 一、量刑流程
1. **确定量刑起点**  
   根据基本犯罪构成事实，在相应法定刑幅度内确定量刑起点。  
2. **确定基准刑**  
   在量刑起点基础上，结合其他影响犯罪构成的情节（如数额、次数、后果等）加重或减轻，得出基准刑。  
3. **调节基准刑**  
   - 先适用**第一层级情节**（未成年人、从犯、未遂、中止、防卫过当等），采用连乘方式调节；  
   - 再适用**第二层级情节**（自首、坦白、累犯、赔偿、认罪认罚等），按同向相加、逆向相减方式调节；  
   - 数罪并罚的，先对各罪分别量刑，再依法合并执行。  
4. **确定宣告刑**  
   - 调节结果在法定幅度内且适当，可直接作为宣告刑；  
   - 必须减轻处罚的情节，应在法定最低刑以下判处；  
   - 仅有从轻情节而结果低于最低刑的，一般以上限就最低刑判处；  
   - 高于法定最高刑的，以最高刑判处；  
   - 在调节结果基础上可允许 ±20% 的合理浮动，超过需提交审委会决定；  
   - 符合缓刑、免刑、单处罚金条件的，依法适用。  
## 二、推理与检索要求
**1. 推理步骤**  
   - 思考过程放在 `<reasoning>` 和 `</reasoning>` 标签之间。  
   - 如需检索外部知识（法律条文或量刑细则），必须先推理，再发起检索。
   - 在推理过程中，必须给出量刑情节，并用一个列表表示，并把量刑情节列表的内容放到`<factors>` 和 `</factors>` 标签之间。
**2. 检索规则**  
- 当需要法条或者量刑细则时，必须发起检索以确保信息的准确性。 
   - 检索请求写在 `<search>` 和 `</search>` 标签内，并标注类型，每次只能发起一个检索，statute或者guideline  
     - `<statute>`：检索刑法及相关法律条文；  
     - `<guideline>`：检索司法解释、量刑指导意见等。  
   - 示例：  
     ```xml
     <search><statute>搜索故意伤害罪相关法条</statute></search>  
     <search><guideline>搜索故意伤害罪量刑细则</guideline></search>#<guideline>标签内要包含罪名
     ```  
**3. 外部信息使用**  
   - 检索结果将以 `<information>` 标签提供给你，你需要基于该信息重新推理。
     注意：<information>标签不是由你提供，是外部检索器提供给你的。  
   - 在获得外部`<information>`后，再进行下一步的推理。
**4. 答案输出**  
   - 如果不再需要外部信息，结合推理给出最终量刑结果。  
   - 结果用 `<answer>` 标签输出，格式为：<answer>最终宣告刑的刑期值</answer>（单位：月）。
   - 示例：  
     ```xml
     <answer>最终宣告刑的刑期值，一个具体的数值，例如 6</answer>
     ```
【一次完整的解答流程示例】
**第一个单元**
<reasoning>...</reasoning>  #第一步的思考过程
<factors>...</factors> #量刑情节信息（用列表表示）。例如："量刑情节": ["盗窃金额既遂3631元","盗窃次数1次","盗窃数额较大","扒窃","当庭自愿认罪","前科"]
<search><statute>搜索故意伤害罪相关法条</statute></search> 
<information>...</information>  #外部搜索引擎提供的信息会放在<information>标签中
**第二个单元** 
<reasoning>...</reasoning>   #下一步的思考过程
<search><guideline>搜索故意伤害罪量刑细则</guideline></search>
<information>...</information>   #外部搜索引擎提供的信息会放在<information>标签中
**第三个单元**
<reasoning>...</reasoning>  #确定最终答案的思考过程
<answer>6</answer>
请开始你的思考和推理：
"""
    else:
        raise NotImplementedError
    return prefix

def build_map_fn(data_source: str, template_type: str):
    """
    返回传给 datasets.map 的映射函数。这里我们只返回新增字段，避免覆盖原字段。
    """
    def _process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # 你也可替换为自有 make_prefix(example, template_type)
        question = make_prefix(example, template_type=template_type)
        solution = {"target": example['meta']["imprisonment"], "fact": example['fact']}

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "fact-reasoning",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": None, "index": idx},  # split 稍后补
        }
    return _process_fn


def filter_by_rules(ds: Dataset, split_name: str) -> Dataset:
    """
    规则过滤（与你原逻辑一致）：
      - imprisonment != 0
      - death_penalty == False
      - life_imprisonment == False
    """
    original = ds.num_rows

    ds = ds.filter(lambda x: x['meta']["imprisonment"] != 0)
    ds = ds.filter(lambda x: x['meta']["death_penalty"] is False)
    ds = ds.filter(lambda x: x['meta']["life_imprisonment"] is False)

    print("%s数据集: 原始 %d 条, 筛选后 %d 条, 过滤 %d 条"
                 % (split_name, original, ds.num_rows, original - ds.num_rows))
    return ds


def add_token_len_and_filter(ds: Dataset, tokenizer, max_length: int = 4096) -> Dataset:
    """
    用 datasets.map 计算 token_len，随后在 Arrow 层过滤，更快/更省内存。
    同时打印统计信息。
    """
    # 计算 token 长度（并行）
    ds = ds.map(
        lambda x: {"token_len": estimate_token_length(x["prompt"][0]["content"], tokenizer)},
        num_proc=max(os.cpu_count() // 2, 1),
        desc="计算 token 长度"
    )

    # 统计
    arr = np.array(ds["token_len"], dtype=np.int32) if ds.num_rows > 0 else np.array([])
    if arr.size > 0:
        print("- 原始样本数量：%d" % ds.num_rows)
        print("- 问题长度（token）最小值：%d" % int(arr.min()))
        print("- 问题长度（token）最大值：%d" % int(arr.max()))
        print("- 问题长度（token）平均值：%.2f" % float(arr.mean()))

    # 过滤
    ds_filtered = ds.filter(lambda x: x["token_len"] <= max_length)
    print("- 过滤后样本数量：%d（移除 %d 个超过 %d token 的样本）" %
                 (ds_filtered.num_rows, ds.num_rows - ds_filtered.num_rows, max_length))

    return ds_filtered


def to_parquet(ds: Dataset, out_path: str, take_first: Optional[int] = None):
    if take_first is not None:
        ds = ds.select(range(min(take_first, ds.num_rows)))
    # 直接 to_pandas 再 to_parquet，确保与你原来输出一致
    df = ds.to_pandas()
    df.to_parquet(out_path, index=False)
    print("保存：%s  （行数：%d）" % (out_path, len(df)))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_tokenizer(name: str = "cl100k_base"):
    try:
        return tiktoken.get_encoding(name)
    except Exception as e:
        raise RuntimeError(f"加载 tokenizer '{name}' 失败: {e}")

def estimate_token_length(text: str, tokenizer) -> int:
    text = text or ""
    # tiktoken.encode 返回 List[int]
    return len(tokenizer.encode(text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/w_reward")
    parser.add_argument("--hdfs_dir", default=None)  # 如需上传 HDFS，可在此扩展
    parser.add_argument("--template_type", type=str, default="base")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--test_take_first", type=int, default=32)  # 与原代码一致
    parser.add_argument("--num_proc", type=int, default=max(os.cpu_count() // 2, 1))
    parser.add_argument("--train_jsonl", type=str, default="/private/*/dataset/train.jsonl",
                        help="Path to the training JSONL file")
    parser.add_argument("--test_jsonl", type=str, default="/private/*/dataset/val.jsonl",
                        help="Path to the test JSONL file")
    args = parser.parse_args()

    ensure_dir(args.local_dir)

    data_source = "ljp"
    print("加载数据集 pljp 训练集&测试集")
    train_raw: Dataset = load_dataset("json", data_files=args.train_jsonl, split="train")

    if args.test_jsonl:
        print(f"加载本地测试集：{args.test_jsonl} ...")
        test_raw: Dataset = load_dataset("json", data_files=args.test_jsonl, split="train")
    else:
        test_raw = None

    # 规则过滤
    train = filter_by_rules(train_raw, "训练")
    test = filter_by_rules(test_raw, "测试")

    # 构造映射函数并并行映射
    map_fn = build_map_fn(data_source=data_source, template_type=args.template_type)

    print("映射生成 prompt/标签（train）...")
    train = train.map(
        map_fn,
        with_indices=True,
        num_proc=args.num_proc,
        desc="生成 train prompt"
    )

    print("映射生成 prompt/标签（test）...")
    test = test.map(
        map_fn,
        with_indices=True,
        num_proc=args.num_proc,
        desc="生成 test prompt"
    )

    # 补充 split 字段
    train = train.map(lambda x: {"extra_info": {**x["extra_info"], "split": "train"}},
                      num_proc=args.num_proc)
    test = test.map(lambda x: {"extra_info": {**x["extra_info"], "split": "test"}},
                     num_proc=args.num_proc)

    # 初始化 tokenizer
    tokenizer = get_tokenizer("cl100k_base")

    # 过滤过长样本（打印统计）
    print("train 数据集处理：")
    train = add_token_len_and_filter(train, tokenizer, max_length=args.max_tokens)

    print("test 数据集处理：")
    test = add_token_len_and_filter(test, tokenizer, max_length=args.max_tokens)

    # 保存为 parquet
    train_out = os.path.join(args.local_dir, "train.parquet")
    test_out = os.path.join(args.local_dir, "test.parquet")

    to_parquet(train, train_out)
    to_parquet(test, test_out, take_first=args.test_take_first)

    print("全部完成。")


if __name__ == "__main__":
    main()