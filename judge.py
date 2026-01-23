import json
import re
import random 
from typing import Optional, Tuple, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import Counter
import argparse
import numpy as np

INPUT_JSONL = ""

# 类别名称（与原脚本保持一致的顺序）
pt_cls2str = [
    "其他",          # 0
    "六个月以下",     # 1
    "六到九个月",     # 2
    "九个月到一年",   # 3
    "一到两年",       # 4
    "二到三年",       # 5
    "三到五年",       # 6
    "五到七年",       # 7
    "七到十年",       # 8
    "十年以上"        # 9
]

# ---------- 工具函数 ----------
def get_pt_cls(months: float) -> int:
    """与原脚本一致的区间划分（严格大于阈值）。输入 months 为月数。"""
    if months > 10 * 12:
        return 9  # 十年以上
    elif months > 7 * 12:
        return 8  # 七到十年
    elif months > 5 * 12:
        return 7  # 五到七年
    elif months > 3 * 12:
        return 6  # 三到五年
    elif months > 2 * 12:
        return 5  # 二到三年
    elif months > 1 * 12:
        return 4  # 一到两年
    elif months > 9:
        return 3  # 九个月到一年
    elif months > 6:
        return 2  # 六到九个月
    elif months > 0:
        return 1  # 六个月以下
    else:
        return 0  # 其他
    
def text2int(s: str) -> Optional[int]:
    if s == "六个月以下":
        return 1
    elif s == "六到九个月":
        return 2
    elif s == "九个月到一年":       
        return 3
    elif s == "一到两年":
        return 4
    elif s == "二到三年":
        return 5
    elif s == "三到五年":
        return 6
    elif s == "五到七年":
        return 7
    elif s == "七到十年":
        return 8
    elif s == "十年以上":
        return 9
    elif s == "其他":
        return 0
    else:
        return None

def parse_answer_text_to_months(ans: Optional[str]) -> Optional[float]:
    """
    从 answer_text 提取区间/单值（单位：月），返回用于分箱的代表值（平均数）。
    允许格式如："12-18"、"18-18"、"24"、"  6 - 12 " 等，出现小数会四舍五入到最近整数。
    返回 None 表示解析失败。
    """
    if not ans or not isinstance(ans, str):
        return None, None, None
    ans = ans.strip()
    # 抓取所有数字（允许小数）
    nums = re.findall(r"\d+(?:\.\d+)?", ans)
    if not nums:
        return None, None, None
    # 转浮点再四舍五入到整数月
    vals = [round(float(x)) for x in nums]
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    # 视为区间，取平均
    low, high = vals[0], vals[1]
    return low, high, (low + high) / 2.0

def load_jsonl_files(paths: List[str]) -> List[str]:
    """按顺序读取多个 JSONL 文件，返回拼接后的所有行。
    任意一个文件不存在或读取失败会给出警告并跳过。
    """
    all_lines: List[str] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                all_lines.extend(f.readlines())
        except Exception as e:
            print(f"[WARN] 无法读取文件：{p}，已跳过。原因：{e}")
    return all_lines

import regex as re
import string
import random


# 中文数字映射表
cn_num = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9
}
cn_unit = {"十": 10, "百": 100, "千": 1000}

def chinese_to_digit(cn: str) -> int:
    raw_cn = cn
    try:
        cn = cn.replace("两", "二")
        cn = cn.lstrip('零')
        pattern = r'[零一二两三四五六七八九十百]+'
        matches = re.search(pattern, cn)
        if matches:
            cn = matches.group()
        else:
            integers_search = re.search(r'\d+', str(cn))
            if integers_search:
                number = float(integers_search.group())
            else:
                number = 0.0
            return number

        import cn2an

        number_text = cn2an.cn2an(cn)
        # print(number_text)
        integers_search = re.search(r'\d+', str(number_text))
        if integers_search:
            number = float(integers_search.group())
        else:
            number = 0.0
    except Exception as e:
        print(f'e = {e}, text = {raw_cn}')
        number = 0.0
        
    return number



def sentence_to_months(text: str) -> float:
    years = months = days = 0
    if text == '':
        return -1
    try:
        # 年
        year_match = re.search(r'([零一二两三四五六七八九十百\d]+)年', text)
        if year_match:
            years = chinese_to_digit(year_match.group(1))

        # 月
        month_match = re.search(r'([零一二两三四五六七八九十百\d]+)(个)?月', text)
        if month_match:
            months = chinese_to_digit(month_match.group(1))

        # 天
        day_match = re.search(r'([零一二两三四五六七八九十百\d]+)(天|日)', text)
        if day_match:
            days = chinese_to_digit(day_match.group(1))
    except:
        return -1

    return round(years * 12 + months + days / 30, 2)

ALLOWED_ACCUSATIONS = [
        "妨害信用卡管理", "非国家工作人员受贿", "单位行贿", "交通肇事", "挪用公款", "非法捕捞水产品", "赌博",
        "非法种植毒品原植物", "盗伐林木", "过失致人死亡", "抢劫", "妨害公务", "非法吸收公众存款", "盗窃",
        "重大责任事故", "非法占用农用地", "容留他人吸毒", "串通投标", "故意伤害", "开设赌场", "行贿",
        "拒不支付劳动报酬", "强奸", "失火", "滥伐林木", "故意杀人", "危险驾驶", "受贿", "非法拘禁",
        "非法侵入住宅", "诈骗", "放火", "合同诈骗", "挪用资金", "销售假冒注册商标的商品", "污染环境",
        "贪污", "非法采矿", "聚众斗殴", "故意毁坏财物", "职务侵占", "非法狩猎"
]

def int_to_zh(num: int) -> str:
    """
    将整数转为中文数字读法（简体，常用口语/书面混合写法）：
    - 支持负数（前缀“负”）
    - 大单位：万、亿、兆（到 10^16-1）
    - 10~19 的最高位写作“十/十一/...”，而非“一十/一十一/...”
    """
    num = int(num)
    digits = "零一二三四五六七八九"
    small_units = ["", "十", "百", "千"]
    big_units = ["", "万", "亿", "兆"]  # 每 4 位一组

    if num == 0:
        return "零"
    if abs(num) >= 10**16:
        raise ValueError("暂仅支持绝对值 < 10^16 的整数。")

    def four_to_zh(n: int) -> str:
        """0..9999：转中文（不加大单位）。内部允许出现‘一十’。"""
        assert 0 <= n < 10000
        if n == 0:
            return ""
        parts = []
        zero_flag = False
        for i in range(3, -1, -1):  # 千百十个
            d = (n // (10**i)) % 10
            if d == 0:
                zero_flag = zero_flag or (len(parts) > 0)  # 只在已有非零后记录零
            else:
                if zero_flag:
                    parts.append("零")
                    zero_flag = False
                parts.append(digits[d] + small_units[i])
        s = "".join(parts)
        # 特例：组内恰为“十/十一/...”，只发生在 n 在 10..19 且高位为空时
        # 但这里只在最高组需要省略“一”，在非最高组保留“一十”
        return s

    neg = num < 0
    n = abs(num)

    # 拆 4 位组
    groups = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000
    # groups[0] 是最低组，对应无大单位

    # 逐组拼接，处理中间零
    res_parts = []
    zero_between = False
    for idx in range(len(groups)-1, -1, -1):
        g = groups[idx]
        if g == 0:
            zero_between = zero_between or (len(res_parts) > 0)
            continue
        chunk = four_to_zh(g)
        if zero_between:
            res_parts.append("零")
            zero_between = False
        res_parts.append(chunk + big_units[idx])

    res = "".join(res_parts)

    # 最高位 10..19：把前缀“一十”改为“十”
    if res.startswith("一十"):
        res = res[1:]

    return ("负" + res) if neg else res


ARTICLE = {'133': 1266, '264': 795, '234': 529, '345': 432, '303': 365, '266': 299, '233': 246, '134': 222, '338': 215, '115': 214, '214': 212, '236': 208, '351': 208, '176': 205, '271': 186, '276': 185, '114': 185, '275': 172, '292': 172, '342': 171, '272': 160, '224': 151, '163': 142, '384': 133, '245': 132, '232': 122, '238': 118, '277': 107, '354': 95, '177': 92, '393': 79, '341': 78, '343': 74, '340': 46, '263': 45, '150': 30, '382': 13, '223': 8, '267': 7, '231': 7, '390': 7, '269': 6, '132': 6, '237': 5, '364': 5, '385': 5, '333': 4, '226': 4, '130': 4, '260': 3, '164': 3, '243': 3, '172': 3, '196': 3, '389': 2, '312': 2, '383': 2, '336': 1, '280': 1, '141': 1, '274': 1, '262': 1, '140': 1, '363': 1, '125': 1, '175': 1, '200': 1, '273': 1, '293': 1, '227': 1, '209': 1, '244': 1, '347': 1, '353': 1, '246': 1, '225': 1, '184': 1, '315': 1, '302': 1}
# ---------- 主评测 ----------
def main(input_files: List[str], method: Optional[str] = None):
    y_true: List[str] = []
    y_pred: List[str] = []

    reg_true: List[float] = []
    reg_pred: List[float] = []

    y_true_charge: List[str] = []
    y_pred_charge: List[str] = []

    y_true_article: List[str] = []
    y_pred_article: List[str] = []

    # 支持多个文件：后面的文件内容拼接到前面之后
    datas = load_jsonl_files(input_files)
    count = 0
    for data in datas:
        obj = json.loads(data)

        # 真值（单位：月）
        true_months = obj.get("meta", {}).get("term_of_imprisonment", {}).get("imprisonment", None)
        if true_months is None:
            true_months = obj.get("meta", {}).get("imprisonment", None)
        if true_months is None:
            # 缺真值时，跳过该样本
            print("here")
            continue
        true_cls = get_pt_cls(float(true_months))

        pred_months = -1
        pred_cls = 0

        # # 预测：先解析 answer_text -> 月数代表值 -> 分箱

        if method == "star":
            pred_months = obj.get("answer_text", None)
            pred_months = sentence_to_months(pred_months)
            try:
                pred_months = float(pred_months)
                pred_cls = get_pt_cls(pred_months)
                if pred_months == -1:
                    count += 1
            except:
                count += 1
                pred_cls = 0
                pred_months = -1

        if method == "MSR^2":
            pred_months = obj.get("answer_text", None)
            try:
                pred_months = float(pred_months)
                pred_cls = get_pt_cls(pred_months)
            except:
                print(obj.get("answer_text", None))
                count += 1
                pred_cls = 0
                pred_months = -1

        # Bert
        if method == "bert" or method == "roberta":
            try:
                pred_cls = text2int(obj.get("answer_text", None))
            except:
                count += 1
                pred_cls = 0

        # SVM CNN
        if method == "svm" or method == "cnn" or method == "direct_infer":
            try:
                pred_cls = int(obj.get("answer_text", None))
            except:
                count += 1
                pred_cls = 0
        
        y_true.append(pt_cls2str[true_cls])
        y_pred.append(pt_cls2str[pred_cls])

        if pred_months >= 0:
            reg_true.append(true_months)
            reg_pred.append(pred_months)
    
    print("-" * 30)
    if len(reg_true) > 0:
        y_t = np.array(reg_true)
        y_p = np.array(reg_pred)

        # 1. 绝对误差 (Absolute Error)
        abs_err = np.abs(y_t - y_p)
        mae = np.mean(abs_err)       # 平均绝对误差
        md_ae = np.median(abs_err)   # 中位数绝对误差 (抗干扰更强)

        # 2. 相对误差 (Relative Error)
        # 这里的 mask 处理分母为0的情况 (例如判决无罪/免予刑事处罚为0个月)
        # 策略：如果 true 为 0，通常不计算相对误差，或者仅当 pred 也为 0 时误差为 0
        mask = y_t > 0
        if np.sum(mask) > 0:
            # 只计算 true > 0 的样本的相对误差
            rel_err = np.abs(y_t[mask] - y_p[mask]) / y_t[mask]
            mre = np.mean(rel_err)    # 平均相对误差
            md_re = np.median(rel_err) # 中位数相对误差
        else:
            mre, md_re = 0.0, 0.0
        
        # 3. 距离准确率 (例如：预测偏差在 ±20% 以内的比例)
        # acc_20pct = np.mean(rel_err < 0.20) * 100

        print(f"Regression Metrics (Valid N={len(reg_true)}):")
        print(f"MAE (平均绝对误差): {mae:.4f} 个月")
        print(f"MRE (平均相对误差): {mre:.4f} ({mre*100:.2f}%)")
        print(f"MedAE (中位绝对误差): {md_ae:.4f} 个月")
    else:
        print("无有效的连续预测值，跳过回归评测。")
    print("-" * 30)

    LABELS = ['其他', '六个月以下', '六到九个月', '九个月到一年', '一到两年', '二到三年', '三到五年', '五到七年', '七到十年', '十年以上']
    acc = accuracy_score(y_true, y_pred)
    # acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0, labels=LABELS)
    map_, mar, maf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0, labels=LABELS)

    acc = round(acc, 6) * 100
    map_ = round(map_, 6) * 100
    mar = round(mar, 6) * 100
    maf = round(maf, 6) * 100

    # print(f"acc:{acc}, map:{map_}, mar:{mar}, maf:{maf}")
    print(f"{acc:.2f} {map_:.2f} {mar:.2f} {maf:.2f}", " count none: ", count)

    print(Counter(y_true))
    print(Counter(y_pred))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0, labels=LABELS))
    
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with single or multiple JSONL files.")
    # 支持多个文件输入：例如 --i a.jsonl b.jsonl c.jsonl
    parser.add_argument(
        "--i",
        type=str,
        nargs="+",
        default=["xxx/output_0925.jsonl"],
        help="一个或多个 JSONL 文件路径，按给定顺序拼接后统一评测"
    )

    parser.add_argument("--m", type=str, default="MSR^2")
    args = parser.parse_args()
    INPUT_FILES = args.i
    main(INPUT_FILES, args.m)