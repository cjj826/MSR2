
import regex as re
import string
import random

# def normalize_answer(s):
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_time_components(time_str):
    """
    从时间字符串中提取年、月、日信息并转换为数值类型
    支持格式：1.8年、1年5个月、2年6个月15天、5个月、15天等
    返回格式：{'year': float, 'mouth': float, 'day': float}
    """
    time_str = re.sub(r'\s+', '', time_str)
    final_value = 0.0

    if '年' not in time_str and '月' not in time_str and '天' not in time_str:
        pattern = r'(\d+\.?\d*)\s*'
        match = re.search(pattern, time_str)
        if match:
            try:
                # 转换为浮点数
                value = float(match.group(1))
                final_value = value
            except ValueError:
                # 处理转换错误（理论上不会发生）
                final_value = 0.0
        return final_value
    
    # 初始化结果字典，默认值为0.0
    result = {'year': 0.0, 'mouth': 0.0, 'day': 0.0}
    
    # 定义正则表达式模式
    patterns = {
        'year': r'(\d+\.?\d*)\s*年',
        'mouth': r'(\d+\.?\d*)\s*(?:个)?月',
        'day': r'(\d+\.?\d*)\s*(?:天|日)'
    }
    
    # 按顺序提取各个部分（年->月->日）
    for unit, pattern in patterns.items():
        match = re.search(pattern, time_str)
        if match:
            try:
                # 转换为浮点数
                value = float(match.group(1))
                result[unit] = value
            except ValueError:
                # 处理转换错误
                result[unit] = 0.0
            # 移除已匹配部分，避免重复匹配
            time_str = time_str.replace(match.group(0), '', 1)
    final_value = result['year'] + result['mouth']/12 + result['day']/365

    return final_value

def split_value(value):
    return_value = []
    value = str(value)
    value = value.replace('，',',')
    value = value.split(',')
    if len(value) != 2:
        return return_value
    
    
    start_value = extract_time_components(value[0])
    end_value = extract_time_components(value[1])
    # print('start_value = ', start_value)
    # print('end_value = ', end_value)


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


def em_check(prediction, golden_answers):
    # if isinstance(golden_answers, str):
    #     golden_answers = [golden_answers]

    # normalized_prediction = normalize_answer(prediction)
    score = 0
    pattern_target = r'[\d年个零一二三四五六七八九十月日天\.，\,\'\"\s]{3,30}'
    pattern = fr'\[{pattern_target}\]'
    # print('prediction = ', prediction)
    matches_prediction = re.findall(pattern, prediction)
    matches_golden_answers = re.findall(pattern, golden_answers)

    # print('matches_prediction = ', matches_prediction)
    # print('matches_golden_answers = ', matches_golden_answers)
    last_prediction = ''
    last_golden_answers = ''
    if len(matches_prediction) != 0:
        last_prediction = matches_prediction[-1]
    if len(matches_golden_answers) != 0:
        last_golden_answers = matches_golden_answers[-1]
    

    score = calculate_score(split_value(last_prediction), split_value(last_golden_answers))


    # print('last_prediction = ', split_value(last_prediction))
    # print('last_golden_answers = ', split_value(last_golden_answers))

    return score



def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
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
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        score = em_check(answer, ground_truth['target'])
        if do_print:
            print('score = ', score)
        # print('score = ', score)
        return score
        # if em_check(answer, ground_truth['target']):
        #     return score
        # else:
        #     return format_score

def test_reward():
    #测试奖励函数中间过程奖励
    # print(em_check("['5.52年', '8.645年'], [8年,13年]  [2,7年]   [3.2个月, 5.3月]", "['5.52年', '8.645年'], [8年,13年]  [5,7]"))
    aaa = """Solution string: <think>
(TaskRunner pid=42272) 好的，我需要确定刘传枢的醉酒驾驶行为的量刑范围。首先，根据案件事实，刘传枢在2022年8月13日无证醉酒驾驶，血液乙醇含量为94mg/100ml，属于醉酒驾驶。根据中国刑法，醉酒驾驶属于危险驾驶罪，依据《刑法》第133条之一。
(TaskRunner pid=42272) 
(TaskRunner pid=42272) 接下来，我需要查证危险驾驶罪的具体量刑标准。根据《刑法》第133条之一，醉酒驾驶机动车的，处拘役，并处罚金。拘役的刑期范围通常是1个月到6个月，但具体还要看是否有加重或减轻情节。
(TaskRunner pid=42272) 
(TaskRunner pid=42272) 案件事实中提到刘传枢无证驾驶，这可能构成从重情节。此外，他在审查起诉期间签署了认罪认罚具结书，到案后如实供述，构成坦白，这可能属于从轻情节。需要确认这些情节对应的量刑影响。
(TaskRunner pid=42272) 
(TaskRunner pid=42272) 另外，血液酒精含量为94mg/100ml，属于超过80mg/100ml的标准，可能属于从重处罚的情节。但无证驾驶是否属于加重情节？可能需要进一步确认法律条文。
(TaskRunner pid=42272) 
(TaskRunner pid=42272) 现在需要搜索危险驾驶罪的量刑标准以及无证驾驶和认罪认罚的具体影响。例如，是否无证驾驶会导致刑期增加，认罪认罚是否减少刑期。同时，检查是否有其他法定或酌定情节影响量刑。
(TaskRunner pid=42272) 
(TaskRunner pid=42272) 可能需要搜索相关司法解释，比如最高人民法院、最高人民检察院关于危险驾驶罪的指导意见，或者相关案例的量刑标准。例如，是否有专门针对醉酒驾驶的量刑指导意见。
(TaskRunner pid=42272) 
(TaskRunner pid=42272) 如果找到明确的条文，可以确定刑期范围。例如，如果无证驾驶和血液酒精含量高的情况下，量刑可能在拘役1-6个月，但根据情节可能会调整。认罪认罚和坦白可能减轻处罚，从而降低刑期。
(TaskRunner pid=42272) 
(TaskRunner pid=42272) 如果信息不足，可能需要进一步搜索具体的法条或司法解释，确保量刑的准确性。
(TaskRunner pid=42272) </think>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <reason>
(TaskRunner pid=42272) 根据案件事实，刘传枢的行为构成危险驾驶罪。依据《中华人民共和国刑法》第133条之一，醉酒驾驶机动车的，处拘役，并处罚金。拘役的刑期为1个月以上6个月以下。但需结合具体情节确定具体刑期。
(TaskRunner pid=42272) </reason>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <search>搜索危险驾驶罪量刑标准及无证驾驶、认罪认罚情节的适用</search>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <information>法条名称: 中华人民共和国道路交通安全法第九十九条内容：有下列行为之一的，由公安机关交通管理部门处二百元以上二千元以下罚款：（一）未取得机动车驾驶证、机动车驾驶证被吊销或者机动车驾驶证被暂扣期间驾驶机动车的；（二）将机动车交由未取得机动车驾驶证或者机动车驾驶证被吊销、暂扣的人驾驶的；（三）造成交通事故后逃逸，尚不构成犯罪的；（四）机动车行驶超过规定时速百分之五十
的；（五）强迫机动车驾驶人违反道路交通安全法律、法规和机动车安全驾驶要求驾驶机动车，造成交通事故，尚不构成犯罪的；（六）违反交通管制的规定强行通行，不听劝阻的；（七）故意损毁、移动、涂改交通设施，造成危害后果，尚不构成犯罪的；（八）非法拦截、扣留机动车辆，不听劝阻，造成交通严重阻塞或者较大财产损失的。行为人有前款第二项、第四项情形之一的，可以并处吊销机动车驾驶证；有第一项、第三项、第五项�
�第八项情形之一的，可以并处十五日以下拘留。
(TaskRunner pid=42272) 法条名称: 中华人民共和国刑法第一百三十三条之一内容：之一 在道路上驾驶机动车，有下列情形之一的，处拘役，并处罚金：（一）追逐竞驶，情节恶劣的；（二）醉酒驾驶机动车的；（三）从事校车业务或者旅客运输，严重超过额定乘员载客，或者严重超过规定时速行驶的；（四）违反危险化学品安全管理规定运输危险化学品，危及公共安全的。机动车所有人、管理人对前款第三项、第四项行为负有直接责任的，
依照前款的规定处罚。有前两款行为，同时构成其他犯罪的，依照处罚较重的规定定罪处罚。
(TaskRunner pid=42272) 法条名称: 中华人民共和国道路交通安全法第一百零一条内容：违反道路交通安全法律、法规的规定，发生重大交通事故，构成犯罪的，依法追究刑事责任，并由公安机关交通管理部门吊销机动车驾驶证。造成交通事故后逃逸的，由公安机关交通管理部门吊销机动车驾驶证，且终生不得重新取得机动车驾驶证。</information>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <reason>
(TaskRunner pid=42272) 根据《道路交通安全法》第九十九条，无证驾驶可处200-2000元罚款，并可能被拘留。《刑法》第133条之一规定醉酒驾驶处拘役并处罚金。结合案件事实，刘传枢存在无证驾驶和醉酒驾驶双重违法情形，且认罪认罚。根据司法实践，醉酒驾驶一般判处拘役1-6个月，无证驾驶可能作为从重情节，但认罪认罚和坦白可作为从轻情节。综合考虑，可能判处拘役1-4个月。</reason>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <search>搜索危险驾驶罪中无证驾驶和认罪认罚的量刑指导意见</search>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <information>法条名称: 中华人民共和国道路交通安全法第九十九条内容：有下列行为之一的，由公安机关交通管理部门处二百元以上二千元以下罚款：（一）未取得机动车驾驶证、机动车驾驶证被吊销或者机动车驾驶证被暂扣期间驾驶机动车的；（二）将机动车交由未取得机动车驾驶证或者机动车驾驶证被吊销、暂扣的人驾驶的；（三）造成交通事故后逃逸，尚不构成犯罪的；（四）机动车行驶超过规定时速百分之五十
的；（五）强迫机动车驾驶人违反道路交通安全法律、法规和机动车安全驾驶要求驾驶机动车，造成交通事故，尚不构成犯罪的；（六）违反交通管制的规定强行通行，不听劝阻的；（七）故意损毁、移动、涂改交通设施，造成危害后果，尚不构成犯罪的；（八）非法拦截、扣留机动车辆，不听劝阻，造成交通严重阻塞或者较大财产损失的。行为人有前款第二项、第四项情形之一的，可以并处吊销机动车驾驶证；有第一项、第三项、第五项�
�第八项情形之一的，可以并处十五日以下拘留。
(TaskRunner pid=42272) 法条名称: 最高人民法院关于适用《中华人民共和国刑事诉讼法》的解释第三百五十条内容：人民法院应当将被告人认罪认罚作为其是否具有社会危险性的重要考虑因素。被告人罪行较轻，采用非羁押性强制措施足以防止发生社会危险性的，应当依法适用非羁押性强制措施。
(TaskRunner pid=42272) 法条名称: 中华人民共和国刑法第一百三十三条之一内容：之一 在道路上驾驶机动车，有下列情形之一的，处拘役，并处罚金：（一）追逐竞驶，情节恶劣的；（二）醉酒驾驶机动车的；（三）从事校车业务或者旅客运输，严重超过额定乘员载客，或者严重超过规定时速行驶的；（四）违反危险化学品安全管理规定运输危险化学品，危及公共安全的。机动车所有人、管理人对前款第三项、第四项行为负有直接责任的，
依照前款的规定处罚。有前两款行为，同时构成其他犯罪的，依照处罚较重的规定定罪处罚。</information>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <reason>
(TaskRunner pid=42272) 综合分析，刘传枢构成危险驾驶罪（醉酒驾驶），同时存在无证驾驶的加重情节。依据《刑法》第133条之一，醉酒驾驶一般判处拘役1-6个月。但无证驾驶可能被认定为从重情节，而认罪认罚和坦白可从轻处罚。结合司法实践，通常醉酒驾驶无证驾驶的刑期为3-6个月，但认罪认罚可能减至2-4个月。因此，量刑范围可能为拘役2个月至4个月。</reason>
(TaskRunner pid=42272) 
(TaskRunner pid=42272) <answer>[2月,4月]</answer>"""
    # print(em_check("[2月,5月]", "[2月,4月]"))
    print(compute_score(aaa, {"target": "[1月,2月]"}))


if __name__ == '__main__':
    test_reward()
