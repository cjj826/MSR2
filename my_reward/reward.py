
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
        return 0.0
    # if '无期' in text:
    #     return 360.0
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
        return 0.0

    return round(years * 12 + months + days / 30, 2)

def split_value(value, split_symbol='-'):
    return_value = []
    value = str(value)
    value = value.replace('，',',')
    value = value.split(split_symbol)
    if len(value) != 2:
        return return_value
    
    if not re.search(r'年|月|日|天', value[0]):
        if '年' in value[1]:
            value[0] = value[0]+ '年'
        elif '月' in value[1]:
            value[0] = value[0]+ '月'
        elif '天' in value[1] or '日' in value[1]:
            value[0] = value[0]+ '天'

    start_value = sentence_to_months(value[0])
    if '无期' in value[1]:
        end_value = start_value + 240.0 
    else:
        end_value = sentence_to_months(value[1])
    if start_value > end_value:
        return return_value 
    return_value.append(start_value)
    return_value.append(end_value)

    return return_value


def calculate_score(pre_value, true_value):
    if len(true_value) == 2 and len(pre_value) == 2:
        return 2.0*(
            min(true_value[1], pre_value[1]) - max(true_value[0], pre_value[0])
            )/(
                true_value[1]-true_value[0] + pre_value[1]-pre_value[0]

            )
    else:
        return 0.0

def calculate_iou(interval1, interval2):
    """
    计算两个区间的交并比（IoU）
    
    参数:
        interval1: (start, end) 元组，如 (1, 5)
        interval2: (start, end) 元组，如 (3, 8)
    
    返回:
        float: IoU 值，范围 [0, 1]
    """

    a_start, a_end = interval1
    b_start, b_end = interval2
    
    # 确保 start <= end
    assert a_start <= a_end, "interval1: start > end"
    assert b_start <= b_end, "interval2: start > end"
    
    # 计算交集
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0, inter_end - inter_start)
    # 计算并集
    union = (a_end - a_start) + (b_end - b_start) - intersection
    # 防止除以零
    if union == 0:
        return 1.0 if intersection == 0 else 0.0  # 两个零长度区间重合则为1
    
    return intersection / union

def get_from_list(text):
    pattern = fr'\[.*\]'
    matches = re.findall(pattern, text)
    if len(matches) != 0:
        last_text = matches[-1]
    return split_value(last_text, split_symbol=',')

def get_from_em(text):
    return split_value(text, split_symbol='-')

from openai import OpenAI
# client = OpenAI(base_url="http://10.0.12.21:1031/v1", api_key="EMPTY")
client = OpenAI(base_url="http://10.0.8.5:24333/v1", api_key="EMPTY")

# max_process_reward 一个缩放尺度，限制了 process reward 的最大值
def get_process_reward(ground_truth, outputs, output_reward=0.0, max_process_reward=0.5):
    factors_pattern = r'<factors>(.*?)</factors>'
    factors_search = re.search(factors_pattern, outputs, re.DOTALL)
    if factors_search:
        factors = factors_search.group()
    else:
        return 0.0
    factors = factors.replace('"', '').replace("'", "")
    fact = ground_truth["fact"]
    # caipanjieguo = ground_truth["caipanjieguo"]

    resp = client.chat.completions.create(
        model="qwen3-32b",  
        messages=[
            {"role": "user",
              "content": f"""你是一名经验丰富的刑事法官。给定案件事实与“量刑情节列表”，请结合法律实务经验与生活常识，判断这些情节在本案中是否可认定，并给出0–10的整数总分（越高表示列表整体越可信、越多情节可被支持）。

说明：情节可能涉及金额/次数/数额认定、作案方式与场景（如入户、扒窃、破坏性手段造成损失等）、身份前科（累犯）、事后表现与程序性情节（退赃退赔、坦白、认罪认罚等）。允许合理推断，但不得脱离事实臆测；信息不足时从严给低分。

评分参考（0–10，整数）：
- 9–10：多数情节与事实明确对应或可短链推断，关键要点齐全，几乎无冲突。
- 7–8：大部分情节可信，少量要点缺失/不确定但不影响整体。
- 5–6：仅部分情节可信，多个情节缺关键认定要点，需补充信息。
- 3–4：可认定情节很少，多数仅弱推断，事实支撑不足。
- 0–2：存在关键情节与事实明显冲突/被否定，或多为无事实锚点的臆测。

扣分触发（在所属档位内下调）：
- 多个情节为“标签式表述”，但事实缺少对应的认定要点/支撑信息：-1~2
- 出现与事实直接冲突或被否定的关键情节（如入户/扒窃/累犯等要件不满足）：-3~5
- 明显无事实锚点、脱离本案的臆测性情节：-1~3
- 情节之间明显互相矛盾：-1~3

只输出一个整数，格式：<answer>7</answer>。不要输出任何解释。

案件事实：{fact}
量刑情节列表：{factors}
请输出总分："""}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}     
    )
    model_output = resp.choices[0].message.content
    print('model_output_process = ', model_output)
    score = extract_solution(model_output)
    if score:
        pattern = r'\d+(?:\.\d+)?'
        score_search = re.search(pattern, str(score))
        if score_search:
            process_reward = min(10.0, float(score_search.group()))/10.0
        else:
            process_reward = 0.0
        # process_reward = max_process_reward*process_reward
        return process_reward

    else:
        return 0.0


def get_universal(text):
    pattern = fr'\[.*\]'
    if re.search(pattern, text):
        return get_from_list(text)
    else:
        return get_from_em(text)



def em_check(prediction, golden_answers, 
            alpha: float = 0.2,   # 目标覆盖率(例如80%置信区间)
            s0: float = 10):
    # if isinstance(golden_answers, str):
    #     golden_answers = [golden_answers]

    # normalized_prediction = normalize_answer(prediction)
    score = 0
    x = float(golden_answers)
    prediction = get_universal(prediction)
    # print(prediction)
    # golden_answers = get_universal(golden_answers)
    if len(prediction) != 2:
        return 0.0
    
    l = prediction[0]
    u = prediction[1]
    S = (u - l)
    # print(S)
    if x < l:
        S += (2.0/alpha) * (l - x)
    elif x > u:
        S += (2.0/alpha) * (x - u)

    # 转为奖励（单调递减变换）
    score = 1.0 / (1.0 + S / s0)
    # print(score)
    # score = calculate_iou(prediction, golden_answers)
    return score

def get_pt_cls(pt):
    if pt > 10 * 12:
        pt_cls = 9
    elif pt > 7 * 12:
        pt_cls = 8
    elif pt > 5 * 12:
        pt_cls = 7
    elif pt > 3 * 12:
        pt_cls = 6
    elif pt > 2 * 12:
        pt_cls = 5
    elif pt > 1 * 12:
        pt_cls = 4
    elif pt > 9:
        pt_cls = 3
    elif pt > 6:
        pt_cls = 2
    elif pt > 0:
        pt_cls = 1
    else:
        pt_cls = 0
    return pt_cls

def em_check_single(prediction, golden_answers):
    pt_cls2str = ["其他", "六个月以下", "六到九个月", "九个月到一年", "一到两年", "二到三年", "三到五年", "五到七年", "七到十年", "十年以上"]
    try:
        golden_answers = float(golden_answers)
        if prediction == '' or prediction is None:
            reward = 0.0
        else:
            reward = 1.0 if pt_cls2str[get_pt_cls(golden_answers)] == pt_cls2str[get_pt_cls(float(prediction))] else 0.0

        return reward
    except:
        return 0.0

def em_check_error(prediction, 
                   golden_answers, 
                    tolerate_error: float = 0.2):
    golden_answers = float(golden_answers)
    if prediction == '':
        reward = 0.0
    else:
        prediction = sentence_to_months(prediction)
        min_tolerate_error_value = max(golden_answers * (1-tolerate_error), 1.0)
        max_tolerate_error_value = max(golden_answers * (1+tolerate_error), 1.0)

        if prediction <= max_tolerate_error_value and prediction >= min_tolerate_error_value:
            reward = 1.0
        else:
            reward = 0.0

    return reward

def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def is_valid_sequence(text):
    content = text
    
    # Check for balanced tags
    tags_to_check = ["reasoning", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:reasoning|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> reasoning -> search -> information -> reasoning -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:reasoning|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<reasoning>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</reasoning>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"

def compute_score(data_source, solution_str, ground_truth, extra_info=None, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1
    # do_print=True
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    output_reward = 0.0
    reward = 0.0
    if answer is None:
        output_reward = 0.0
    else:
        output_reward = em_check_single(answer, ground_truth['target'])##多分类

    # 添加 format_score 奖励
    # format_score = 0.2
    # is_valid, msg = is_valid_sequence(solution_str)
    # if not is_valid:
    #     print(f"Invalid format: {msg}, apply format_score penalty {format_score}")
    #     output_reward = max(0.0, output_reward - format_score)
    # else:
    #     print("Valid format.")
    #     output_reward = max(output_reward, format_score)
    
    # 添加 process_reward 奖励
    if_add_process_reward = True
    if if_add_process_reward:
        process_reward = get_process_reward(ground_truth, solution_str, output_reward)
    else:
        process_reward = 0.0
    reward = 0.8 * output_reward + 0.2 * process_reward

    if do_print:
        print('reward = ', reward)
        print('output_reward = ', output_reward)
        print('process_reward = ', process_reward)
    
    return round(reward, 4)
