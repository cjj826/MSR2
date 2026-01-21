import os
import re
import time
import json
import torch
import requests
import transformers
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
from tqdm import tqdm
import hashlib 

from legal_retrieval.retrieval import Legal_Retrieval
law_article_retrieval = Legal_Retrieval()

# -----------------------------
# 可选：vLLM 依赖（仅在 use_vllm=True 时需要）
# -----------------------------
try:
    from vllm import LLM, SamplingParams
    _VLLM_OK = True
except Exception:
    _VLLM_OK = False


# -----------------------------
# 配置
# -----------------------------
@dataclass
class RunConfig:
    model_id: str = 'Qwen/Qwen3-8B'
    use_vllm: bool = True               # ← 开关：是否使用 vLLM(0.8.4)
    visible_gpus: str = "0"             # ← 指定用哪些GPU，例如 "0,1"；CPU 跑就置空 ""
    gpu_memory_utilization: float = 0.92# ← vLLM 显存占用比例(0~1)，建议 0.85~0.95
    dtype: str = "bfloat16"             # "bfloat16"/"float16"/"auto"（vLLM & HF）
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    react_max_turns: int = 8            # ReAct 最多轮数
    search_topk: int = 3
    search_timeout: int = 12
    print_each_output: bool = False     # 是否逐题打印完整模型输出（便于调试）
    eval_prompt: int = 0 
    retrieve_statute: bool = True
    retrieve_guideline: bool = True

# -----------------------------
# 公共工具
# -----------------------------
def set_visible_gpus(visible_gpus: str):
    """
    控制使用哪些 GPU。传入 "0,1" 则使用 0/1 两张卡；传入 "" 则禁用 GPU（走 CPU/HF）。
    必须在模型/引擎初始化之前设置。
    """
    if visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

def to_torch_dtype(dtype_str: str):
    m = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": None}
    return m.get(dtype_str.lower(), None)

def build_user_prompt(fact: str, eval_prompt) -> str:
    prompt = f"""你是一名中国刑事量刑助手，请阅读以下材料描述的案情事实并回答问题：
【案情事实开始】
 {fact}
【案情事实结束】

【任务目标】
根据案情事实，预测被告人的最终宣告刑的刑期值（单位：月），并仅按指定格式输出。

【量刑基本方法（必须遵循）】
量刑时，应当以定性分析为主，定量分析为辅，依次确定量刑起点、基准刑和宣告刑。（一）量刑步骤 1.根据基本犯罪构成事实在相应的法定刑幅度内确定量刑起点。2.根据其他影响犯罪构成的犯罪数额、犯罪次数、犯罪后果等犯罪事实，在量刑起点的基础上增加刑罚量确定基准刑。3.根据量刑情节调节基准刑，并综合考虑全案情况，依法确定宣告刑。（二）调节基准刑的方法1.具有单个量刑情节的，根据量刑情节的调节比例直接调节基准刑。2.具有多个量刑情节的，一般根据各个量刑情节的调节比例，采用同向相加、逆向相减的方法调节基准刑；具有未成年人犯罪、老年人犯罪、限制行为能力的精神病人犯罪、又聋又哑的人或者盲人犯罪，防卫过当、避险过当、犯罪预备、犯罪未遂、犯罪中止，从犯、胁从犯和教唆犯等量刑情节的，先适用该量刑情节对基准刑进行调节，在此基础上，再适用其他量刑情节进行调节。3.被告人犯数罪，同时具有适用于个罪的立功、累犯等量刑情节的，先适用该量刑情节调节个罪的基准刑，确定个罪所应判处的刑罚，再依法实行数罪并罚，决定执行的刑罚。（三）确定宣告刑的方法1.量刑情节对基准刑的调节结果在法定刑幅度内，且罪责刑相适应的，可以直接确定为宣告刑；具有应当减轻处罚情节的，应当依法在法定最低刑以下确定宣告刑。2.量刑情节对基准刑的调节结果在法定最低刑以下，具有法定减轻处罚情节，且罪责刑相适应的，可以直接确定为宣告刑；只有从轻处罚情节的，可以依法确定法定最低刑为宣告刑；但是根据案件的特殊情况，经最高人民法院核准，也可以在法定刑以下判处刑罚。3.量刑情节对基准刑的调节结果在法定最高刑以上的，可以依法确定法定最高刑为宣告刑。4.综合考虑全案情况，独任审判员或合议庭可以在20%的幅度内对调节结果进行调整，确定宣告刑。当调节后的结果仍不符合罪责刑相适应原则的，应当提交审判委员会讨论，依法确定宣告刑。5.综合全案犯罪事实和量刑情节，依法应当判处无期徒刑以上刑罚、拘役、管制或者单处附加刑、缓刑、免予刑事处罚的，应当依法适用。（四）判处罚金刑，应当以犯罪情节为根据，并综合考虑被告人缴纳罚金的能力，依法决定罚金数额。（五）适用缓刑，应当综合考虑被告人的犯罪情节、悔罪表现、再犯罪的危险以及宣告缓刑对所居住社区的影响，依法作出决定。
量刑时应当充分考虑各种法定和酌定量刑情节，根据案件的全部犯罪事实以及量刑情节的不同情形，依法确定量刑情节的适用及其调节比例。对黑恶势力犯罪、严重暴力犯罪、毒品犯罪、性侵未成年人犯罪等危害严重的犯罪，在确定从宽的幅度时，应当从严掌握；对犯罪情节较轻的犯罪，应当充分体现从宽。具体确定各个量刑情节的调节比例时，应当综合平衡调节幅度与实际增减刑罚量的关系，确保罪责刑相适应。\n （一）对于未成年人犯罪，综合考虑未成年人对犯罪的认知能力、实施犯罪行为的动机和目的、犯罪时的年龄、是否初犯、偶犯、悔罪表现、个人成长经历和一贯表现等情况，应当予以从宽处罚。\n 1.已满十二周岁不满十六周岁的未成年人犯罪，减少基准刑的30%-60%；\n 2.已满十六周岁不满十八周岁的未成年人犯罪，减少基准刑的10%-50%。\n（二）对于已满七十五周岁的老年人故意犯罪，综合考虑犯罪的性质、情节、后果等情况，可以减少基准刑的40%以下；过失犯罪的，减少基准刑的20％-50%。\n（三）对于又聋又哑的人或者盲人犯罪，综合考虑犯罪性质、情节、后果以及聋哑人或者盲人犯罪时的控制能力等情况，可以减少基准刑的50%以下；犯罪较轻的，可以减少基准刑的50%以上或者依法免除处罚。\n（四）对于未遂犯，综合考虑犯罪行为的实行程度、造成损害的大小、犯罪未得逞的原因等情况，可以比照既遂犯减少基准刑的50%以下。\n（五）对于从犯，综合考虑其在共同犯罪中的地位、作用等情况，应当予以从宽处罚，减少基准刑的20%-50%；犯罪较轻的，减少基准刑的50%以上或者依法免除处罚。\n（六）对于自首情节，综合考虑自首的动机、时间、方式、罪行轻重、如实供述罪行的程度以及悔罪表现等情况，可以减少基准刑的40%以下；犯罪较轻的，可以减少基准刑的40%以上或者依法免除处罚。恶意利用自首规避法律制裁等不足以从宽处罚的除外。\n（七）对于坦白情节，综合考虑如实供述罪行的阶段、程度、罪行轻重以及悔罪表现等情况，确定从宽的幅度。\n 1.如实供述自己罪行的，可以减少基准刑的20%以下；\n 2.如实供述司法机关尚未掌握的同种较重罪行的，可以减少基准刑的10%-30%；\n 3.因如实供述自己罪行，避免特别严重后果发生的，可以减少基准刑的30%-50%。\n（八）对于当庭自愿认罪的，根据犯罪的性质、罪行的轻重、认罪程度以及悔罪表现等情况，可以减少基准刑的10%以下。依法认定自首、坦白的除外。\n（九）对于立功情节，综合考虑立功的大小、次数、内容、来源、效果以及罪行轻重等情况，确定从宽的幅度。\n 1.一般立功的，可以减少基准刑的20%以下；\n 2.重大立功的，可以减少基准刑的20%-50%；犯罪较轻的，减少基准刑的50%以上或者依法免除处罚。\n（十）对于退赃、退赔的，综合考虑犯罪性质，退赃、退赔行为对损害结果所能弥补的程度，退赃、退赔的数额及主动程度等情况，可以减少基准刑的30%以下；对抢劫等严重危害社会治安犯罪的，应当从严掌握。\n（十一）对于积极赔偿被害人经济损失并取得谅解的，综合考虑犯罪性质、赔偿数额、赔偿能力以及认罪悔罪表现等情况，可以减少基准刑的40%以下；积极赔偿但没有取得谅解的，可以减少基准刑的30%以下；尽管没有赔偿，但取得谅解的，可以减少基准刑的20%以下。对抢劫、强奸等严重危害社会治安犯罪的，应当从严掌握。\n（十二）对于当事人根据刑事诉讼法第二百八十八条达成刑事和解协议的，综合考虑犯罪性质、赔偿数额、赔礼道歉以及真诚悔罪等情况，可以减少基准刑的50%以下；犯罪较轻的，可以减少基准刑的50%以上或者依法免除处罚。\n（十三）对于被告人在羁押期间表现好的，可以减少基准刑的10%以下。\n（十四）对于被告人认罪认罚的，综合考虑犯罪的性质、罪行的轻重、认罪认罚的阶段、程度、价值、悔罪表现等情况，可以减少基准刑的30%以下；具有自首、重大坦白、退赃退赔、赔偿谅解、刑事和解等情节的，可以减少基准刑的60%以下，犯罪较轻的，可以减少基准刑的60%以上或者依法免除处罚。认罪认罚与自首、坦白、当庭自愿认罪、退赃退赔、赔偿谅解、刑事和解、羁押期间表现好等量刑情节不作重复评价。\n（十五）对于累犯，综合考虑前后罪的性质、刑罚执行完毕或赦免以后至再犯罪时间的长短以及前后罪罪行轻重等情况，应当增加基准刑的10%-40%，一般不少于3个月。\n（十六）对于有前科的，综合考虑前科的性质、时间间隔长短、次数、处罚轻重等情况，可以增加基准刑的10%以下。前科犯罪为过失犯罪和未成年人犯罪的除外。\n（十七）对于犯罪对象为未成年人、老年人、残疾人、孕妇等弱势人员的，综合考虑犯罪的性质、犯罪的严重程度等情况，可以增加基准刑的20%以下。\n（十八）对于在重大自然灾害、预防、控制突发传染病疫情等灾害期间故意犯罪的，根据案件的具体情况，可以增加基准刑的20%以下。

【推理与检索要求】	
1. 思考和推理的过程放在<reasoning>和</reasoning>标签之间。
2. 若推理后发现缺少某些知识，如当需要法律条文、量刑指导意见、量刑标准或者量刑细则时，必须发起检索以确保信息的准确性。 
3. 检索请求写在<search>和</search>标签内，并标注类型，每次只能发起一个检索，statute 或者 guideline  
    - <statute>：检索相关的法律条文；  
    - <guideline>：检索相关的量刑指导意见、量刑标准或量刑细则等。  
    - 示例：  
     ```
     <search><statute>搜索故意伤害罪相关法条</statute></search>  
     <search><guideline>搜索故意伤害罪量刑指导意见、量刑标准或量刑细则</guideline></search> # 注：<guideline>标签内要包含具体准确的罪名
     ```  
4. 搜索结果将位于<information>和</information>标签之间返回。

【输出要求（必须遵循）】
1. 若无需外部知识，直接给出答案；如需检索，待收到<information>后再给出答案。
2. 结果用 `<answer>` 标签输出，格式为：<answer>最终宣告刑的刑期值</answer>（单位：月）。

【一次完整的解答流程示例】
**第一个单元**
<reasoning>第一步的思考和推理过程</reasoning>
<search><statute>搜索故意伤害罪相关法条</statute></search> 
<information>检索的法条信息</information>  #外部搜索引擎提供的信息会放在<information>标签中
**第二个单元**
<reasoning>下一步的思考和推理过程</reasoning>
<search><guideline>搜索故意伤害量刑指导意见、量刑标准或量刑细则</guideline></search>
<information>检索的量刑指导意见、量刑标准或量刑细则信息</information>   #外部搜索引擎提供的信息会放在<information>标签中
**第三个单元**
<reasoning>确定最终答案的思考和推理过程</reasoning>
<answer>最终宣告刑的刑期值，一个具体的数值，例如 6</answer>

请开始你的思考和推理：
"""

    if eval_prompt == 1:
        prompt = f"""你是一名中国刑事量刑助手，请阅读以下材料描述的案情事实并回答问题：
【案情事实开始】
 {fact}
【案情事实结束】

【任务目标】
根据案情事实，预测被告人的最终宣告刑的刑期区间对应的种类编号，只输出 0-9 的int数字，分别代表：
- 0: 其他
- 1: 六个月以下
- 2: 六到九个月
- 3: 九个月到一年
- 4: 一到两年
- 5: 二到三年
- 6: 三到五年
- 7: 五到七年
- 8: 七到十年
- 9: 十年以上

【量刑基本方法（必须遵循）】
量刑时，应当以定性分析为主，定量分析为辅，依次确定量刑起点、基准刑和宣告刑。（一）量刑步骤 1.根据基本犯罪构成事实在相应的法定刑幅度内确定量刑起点。2.根据其他影响犯罪构成的犯罪数额、犯罪次数、犯罪后果等犯罪事实，在量刑起点的基础上增加刑罚量确定基准刑。3.根据量刑情节调节基准刑，并综合考虑全案情况，依法确定宣告刑。（二）调节基准刑的方法1.具有单个量刑情节的，根据量刑情节的调节比例直接调节基准刑。2.具有多个量刑情节的，一般根据各个量刑情节的调节比例，采用同向相加、逆向相减的方法调节基准刑；具有未成年人犯罪、老年人犯罪、限制行为能力的精神病人犯罪、又聋又哑的人或者盲人犯罪，防卫过当、避险过当、犯罪预备、犯罪未遂、犯罪中止，从犯、胁从犯和教唆犯等量刑情节的，先适用该量刑情节对基准刑进行调节，在此基础上，再适用其他量刑情节进行调节。3.被告人犯数罪，同时具有适用于个罪的立功、累犯等量刑情节的，先适用该量刑情节调节个罪的基准刑，确定个罪所应判处的刑罚，再依法实行数罪并罚，决定执行的刑罚。（三）确定宣告刑的方法1.量刑情节对基准刑的调节结果在法定刑幅度内，且罪责刑相适应的，可以直接确定为宣告刑；具有应当减轻处罚情节的，应当依法在法定最低刑以下确定宣告刑。2.量刑情节对基准刑的调节结果在法定最低刑以下，具有法定减轻处罚情节，且罪责刑相适应的，可以直接确定为宣告刑；只有从轻处罚情节的，可以依法确定法定最低刑为宣告刑；但是根据案件的特殊情况，经最高人民法院核准，也可以在法定刑以下判处刑罚。3.量刑情节对基准刑的调节结果在法定最高刑以上的，可以依法确定法定最高刑为宣告刑。4.综合考虑全案情况，独任审判员或合议庭可以在20%的幅度内对调节结果进行调整，确定宣告刑。当调节后的结果仍不符合罪责刑相适应原则的，应当提交审判委员会讨论，依法确定宣告刑。5.综合全案犯罪事实和量刑情节，依法应当判处无期徒刑以上刑罚、拘役、管制或者单处附加刑、缓刑、免予刑事处罚的，应当依法适用。
量刑时应当充分考虑各种法定和酌定量刑情节，根据案件的全部犯罪事实以及量刑情节的不同情形，依法确定量刑情节的适用及其调节比例。对黑恶势力犯罪、严重暴力犯罪、毒品犯罪、性侵未成年人犯罪等危害严重的犯罪，在确定从宽的幅度时，应当从严掌握；对犯罪情节较轻的犯罪，应当充分体现从宽。具体确定各个量刑情节的调节比例时，应当综合平衡调节幅度与实际增减刑罚量的关系，确保罪责刑相适应。\n （一）对于未成年人犯罪，综合考虑未成年人对犯罪的认知能力、实施犯罪行为的动机和目的、犯罪时的年龄、是否初犯、偶犯、悔罪表现、个人成长经历和一贯表现等情况，应当予以从宽处罚。\n 1.已满十二周岁不满十六周岁的未成年人犯罪，减少基准刑的30%-60%；\n 2.已满十六周岁不满十八周岁的未成年人犯罪，减少基准刑的10%-50%。\n（二）对于已满七十五周岁的老年人故意犯罪，综合考虑犯罪的性质、情节、后果等情况，可以减少基准刑的40%以下；过失犯罪的，减少基准刑的20％-50%。\n（三）对于又聋又哑的人或者盲人犯罪，综合考虑犯罪性质、情节、后果以及聋哑人或者盲人犯罪时的控制能力等情况，可以减少基准刑的50%以下；犯罪较轻的，可以减少基准刑的50%以上或者依法免除处罚。\n（四）对于未遂犯，综合考虑犯罪行为的实行程度、造成损害的大小、犯罪未得逞的原因等情况，可以比照既遂犯减少基准刑的50%以下。\n（五）对于从犯，综合考虑其在共同犯罪中的地位、作用等情况，应当予以从宽处罚，减少基准刑的20%-50%；犯罪较轻的，减少基准刑的50%以上或者依法免除处罚。\n（六）对于自首情节，综合考虑自首的动机、时间、方式、罪行轻重、如实供述罪行的程度以及悔罪表现等情况，可以减少基准刑的40%以下；犯罪较轻的，可以减少基准刑的40%以上或者依法免除处罚。恶意利用自首规避法律制裁等不足以从宽处罚的除外。\n（七）对于坦白情节，综合考虑如实供述罪行的阶段、程度、罪行轻重以及悔罪表现等情况，确定从宽的幅度。\n 1.如实供述自己罪行的，可以减少基准刑的20%以下；\n 2.如实供述司法机关尚未掌握的同种较重罪行的，可以减少基准刑的10%-30%；\n 3.因如实供述自己罪行，避免特别严重后果发生的，可以减少基准刑的30%-50%。\n（八）对于当庭自愿认罪的，根据犯罪的性质、罪行的轻重、认罪程度以及悔罪表现等情况，可以减少基准刑的10%以下。依法认定自首、坦白的除外。\n（九）对于立功情节，综合考虑立功的大小、次数、内容、来源、效果以及罪行轻重等情况，确定从宽的幅度。\n 1.一般立功的，可以减少基准刑的20%以下；\n 2.重大立功的，可以减少基准刑的20%-50%；犯罪较轻的，减少基准刑的50%以上或者依法免除处罚。\n（十）对于退赃、退赔的，综合考虑犯罪性质，退赃、退赔行为对损害结果所能弥补的程度，退赃、退赔的数额及主动程度等情况，可以减少基准刑的30%以下；对抢劫等严重危害社会治安犯罪的，应当从严掌握。\n（十一）对于积极赔偿被害人经济损失并取得谅解的，综合考虑犯罪性质、赔偿数额、赔偿能力以及认罪悔罪表现等情况，可以减少基准刑的40%以下；积极赔偿但没有取得谅解的，可以减少基准刑的30%以下；尽管没有赔偿，但取得谅解的，可以减少基准刑的20%以下。对抢劫、强奸等严重危害社会治安犯罪的，应当从严掌握。\n（十二）对于当事人根据刑事诉讼法第二百八十八条达成刑事和解协议的，综合考虑犯罪性质、赔偿数额、赔礼道歉以及真诚悔罪等情况，可以减少基准刑的50%以下；犯罪较轻的，可以减少基准刑的50%以上或者依法免除处罚。\n（十三）对于被告人在羁押期间表现好的，可以减少基准刑的10%以下。\n（十四）对于被告人认罪认罚的，综合考虑犯罪的性质、罪行的轻重、认罪认罚的阶段、程度、价值、悔罪表现等情况，可以减少基准刑的30%以下；具有自首、重大坦白、退赃退赔、赔偿谅解、刑事和解等情节的，可以减少基准刑的60%以下，犯罪较轻的，可以减少基准刑的60%以上或者依法免除处罚。认罪认罚与自首、坦白、当庭自愿认罪、退赃退赔、赔偿谅解、刑事和解、羁押期间表现好等量刑情节不作重复评价。\n（十五）对于累犯，综合考虑前后罪的性质、刑罚执行完毕或赦免以后至再犯罪时间的长短以及前后罪罪行轻重等情况，应当增加基准刑的10%-40%，一般不少于3个月。\n（十六）对于有前科的，综合考虑前科的性质、时间间隔长短、次数、处罚轻重等情况，可以增加基准刑的10%以下。前科犯罪为过失犯罪和未成年人犯罪的除外。\n（十七）对于犯罪对象为未成年人、老年人、残疾人、孕妇等弱势人员的，综合考虑犯罪的性质、犯罪的严重程度等情况，可以增加基准刑的20%以下。\n（十八）对于在重大自然灾害、预防、控制突发传染病疫情等灾害期间故意犯罪的，根据案件的具体情况，可以增加基准刑的20%以下。

【推理与检索要求】	
1. 思考和推理的过程放在<reasoning>和</reasoning>标签之间。
2. 若推理后发现缺少某些知识，如当需要法律条文、量刑指导意见、量刑标准或者量刑细则时，必须发起检索以确保信息的准确性。 
3. 检索请求写在<search>和</search>标签内，并标注类型，每次只能发起一个检索，statute 或者 guideline  
    - <statute>：检索相关的法律条文；  
    - <guideline>：检索相关的量刑指导意见、量刑标准或量刑细则等。  
    - 示例：  
     ```
     <search><statute>搜索故意伤害罪相关法条</statute></search>  
     <search><guideline>搜索故意伤害罪量刑指导意见、量刑标准或量刑细则</guideline></search> # 注：<guideline>标签内要包含具体准确的罪名
     ```  
4. 搜索结果将位于<information>和</information>标签之间返回。

【输出要求（必须遵循）】
1. 若无需外部知识，直接给出答案；如需检索，待收到<information>后再给出答案。
2. 结果用 `<answer>` 标签输出，格式为：<answer>最终宣告刑的刑期区间对应的种类编号</answer>。

【一次完整的解答流程示例】
**第一个单元**
<reasoning>第一步的思考和推理过程</reasoning>
<search><statute>搜索故意伤害罪相关法条</statute></search> 
<information>检索的法条信息</information>  #外部搜索引擎提供的信息会放在<information>标签中
**第二个单元**
<reasoning>下一步的思考和推理过程</reasoning>
<search><guideline>搜索故意伤害量刑指导意见、量刑标准或量刑细则</guideline></search>
<information>检索的量刑指导意见、量刑标准或量刑细则信息</information>   #外部搜索引擎提供的信息会放在<information>标签中
**第三个单元**
<reasoning>确定最终答案的思考和推理过程</reasoning>
<answer>最终宣告刑的刑期区间对应的种类编号，一个 0-9 的int数字，例如 1</answer>

请开始你的思考和推理：
"""
    
    if eval_prompt == 2:
        prompt = f"""你是一名中国刑事量刑助手，请阅读以下材料描述的案情事实并回答问题：
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
   - 在推理过程中，当你给出量刑情节时，请把量刑情节的内容放到`<factors>` 和 `</factors>` 标签之间。
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
   - 结果用 `<answer>` 标签输出。  
   - 示例：  
     ```xml
     <answer>3年6个月</answer>
     ```
【一次完整的解答流程示例】
**第一个单元**
<reasoning>...</reasoning>  #第一步的思考过程
<factors>...</factors> #量刑情节信息，例如："量刑情节": ["盗窃金额既遂3631元","盗窃次数1次","盗窃数额较大","扒窃","当庭自愿认罪","前科"]
<search><statute>搜索故意伤害罪相关法条</statute></search> 
<information>...</information>  #外部搜索引擎提供的信息会放在<information>标签中
**第二个单元** 
<reasoning>...</reasoning>   #下一步的思考过程
<search><guideline>搜索故意伤害罪量刑细则</guideline></search>
<information>...</information>   #外部搜索引擎提供的信息会放在<information>标签中
**第三个单元**
<reasoning>...</reasoning>  #确定最终答案的思考过程
<answer>3年6个月</answer>
请开始你的思考和推理：
"""
    if eval_prompt == 3:
        prompt = f"""你是一名中国刑事量刑助手，请阅读以下材料描述的案情事实并回答问题：
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

    return prompt

def wrap_chat(tokenizer, content: str) -> str:
    # 兼容 Qwen 的 chat_template（你原来用了 enable_thinking=True；保留）
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
        except TypeError:
            # 旧版 tokenizer 没有 enable_thinking 参数
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False
            )
    return content

def extract_last_search_query(text: str) -> Optional[str]:
    # 只抓取 <search> ... </search> 中的最后一次
    m = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    return m[-1] if m else None

def legal_search(query: Optional[str], topk: int, timeout: int, retrieve_statute, retrieve_guideline) -> str:
    if not query:
        return ""
    try:
        # 你已有的检索封装（返回结构保持你原来那套）
        results = law_article_retrieval.search_law_articles([query], retrieve_statute=retrieve_statute, retrieve_guideline=retrieve_guideline)["result"]
        if not results or not results[0]:
            return "未检索到相关法条或量刑指导。"
        format_reference = []
        for idx in results[0]:
            try:
                name = idx["law_name"]
                title = idx["title"]
                content = idx["content"].replace("\n", "")
                format_reference.append(f"法条名称: {name}{title}内容：{content}")
            except Exception:
                format_reference.append(str(idx))
        return "\n".join(format_reference)
    except Exception as e:
        return f"检索异常：{e}"

STOP_VARIANTS = [
    "</search>", "</guideline></search>", "</statute></search>"
]
CURR_EOS = [151645, 151643]  # Qwen2.5 常见结束符（兜底）


# -----------------------------
# HF 后端：自定义停止条件
# -----------------------------
class StopOnStrings(transformers.StoppingCriteria):
    def __init__(self, stop_strings, tokenizer, window_tokens=64):
        self.stop_strings = stop_strings
        self.tok = tokenizer
        self.win = window_tokens

    def __call__(self, input_ids, scores, **kwargs):
        # 只看最后 win 个 token，转成文本（不跳过特殊符号，避免被吞空格/换行）
        tail_ids = input_ids[0][-self.win:]
        tail_text = self.tok.decode(tail_ids, skip_special_tokens=False)
        t = tail_text.rstrip()
        for s in self.stop_strings:
            if s in t:
                print(f"[STOP triggered] found {repr(s)} in tail: {repr(t)}")
                return True
        return False

# -----------------------------
# HF 推理一步（截到 </search>）
# -----------------------------
def hf_generate_once(model, tokenizer, prompt: str, cfg: RunConfig) -> Tuple[str, int]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    attention_mask = torch.ones_like(input_ids)

    # 放到与模型一致的设备（HF: device_map=auto 时内部会处理；这里简化为 cuda 优先）
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    stopping = transformers.StoppingCriteriaList([StopOnStrings(STOP_VARIANTS, tokenizer, window_tokens=64)])

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        stopping_criteria=stopping,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ⛏ 后处理：只保留到最后一个停词
    def trim_stop(text: str, stop_strings) -> str:
        cut = len(text)
        for s in stop_strings:
            idx = text.rfind(s)
            if idx != -1:
                cut = min(cut, idx + len(s))
        return text[:cut]

    text_trimmed = trim_stop(raw_text, STOP_VARIANTS)
    last_tid = outputs[0][-1].item()
    return text_trimmed, last_tid


# -----------------------------
# vLLM 推理一步（截到 </search>）
# -----------------------------
def vllm_generate_once(vllm_engine, prompt: str, cfg: RunConfig) -> str:
    # 1. 构造采样参数
    sampling = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        stop=STOP_VARIANTS,  # 仅核心停词，无空格换行变体
        include_stop_str_in_output=True,
        repetition_penalty=1.05,
        frequency_penalty=0.3,  
    )

    # 2. 执行生成
    outs = vllm_engine.generate([prompt], sampling_params=sampling, use_tqdm=False)
    raw_text = outs[0].outputs[0].text

    # 3. 后处理裁剪到最后一个停词
    def trim_stop(text: str, stop_strings) -> str:
        cut = len(text)
        for s in stop_strings:
            idx = text.rfind(s)
            if idx != -1:
                cut = min(cut, idx + len(s))
        return text[:cut]

    trimmed_text = trim_stop(raw_text, STOP_VARIANTS)

    # 4. 返回裁剪后的文本（避免多出 < 或换行）
    return trimmed_text

# -----------------------------
# ReAct 主循环（兼容 HF / vLLM）
# -----------------------------

def react_infer(tokenizer, model_or_engine, cfg: RunConfig, fact: str) -> str:
    user_prompt = build_user_prompt(fact, cfg.eval_prompt)
    prompt = wrap_chat(tokenizer, user_prompt)
    final_chunks = []

    for _ in range(cfg.react_max_turns):
        if cfg.use_vllm:
            text_new = vllm_generate_once(model_or_engine, prompt, cfg)
            last_eos_hit = False  # vLLM 不返回最后 token id，这里不靠它收束
        else:
            text_new, last_tid = hf_generate_once(model_or_engine, tokenizer, prompt, cfg)
            last_eos_hit = (last_tid in CURR_EOS)

        # 结束条件（启发式）：出现 <answer> 且没有新的 <search>；或 HF 明确命中 eos
        if ("<answer>" in text_new and "</answer>" in text_new and "</search>" not in text_new) or last_eos_hit:
            # print(text_new)
            final_chunks.append(text_new)
            break

        # 抓取本轮 search → 检索 → 注入 information
        query = extract_last_search_query(text_new)
        search_results = legal_search(query, cfg.search_topk, cfg.search_timeout, cfg.retrieve_statute, cfg.retrieve_guideline) if query else ""
        info_block = f"\n\n{text_new}<information>{search_results}</information>\n\n"
        prompt += info_block
        # print(info_block)
        final_chunks.append(info_block)

    return "".join(final_chunks)


# -----------------------------
# 直推（不循环）
# -----------------------------
def direct_infer(tokenizer, model_or_engine, cfg: RunConfig, fact: str) -> str:
    user_prompt = build_user_prompt(fact)
    prompt = wrap_chat(tokenizer, user_prompt)

    if cfg.use_vllm:
        text = vllm_generate_once(model_or_engine, prompt, cfg)
    else:
        text, _ = hf_generate_once(model_or_engine=tokenizer,  # 故意写错会报错；留着提醒不要误用
                                   tokenizer=None, prompt="", cfg=cfg)  # 占位防误用
        # 上面两行只是保护，真正 HF 直推请改为：
        input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                     truncation=True, max_length=tokenizer.model_max_length)
        attention_mask = torch.ones_like(input_ids)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        outputs = model_or_engine.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text


# -----------------------------
# 加载器：HF
# -----------------------------
def load_hf(cfg: RunConfig):
    torch_dtype = to_torch_dtype(cfg.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype if torch_dtype is not None else "auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    return tokenizer, model

# -----------------------------
# 加载器：vLLM
# -----------------------------
def load_vllm(cfg: RunConfig):
    if not _VLLM_OK:
        raise RuntimeError("未检测到 vLLM，请先 pip install 'vllm==0.8.4'")
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_id)
    # 重要：visible_gpus 通过 CUDA_VISIBLE_DEVICES 控制；并发/分片由 vLLM 自行适配
    # gpu_memory_utilization 控制显存占用比例；dtype 传入字符串
    engine = LLM(
        model=cfg.model_id,
        dtype=cfg.dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=True,  # Qwen 系列通常需要
        # 其他可选：max_model_len=..., tensor_parallel_size=..., download_dir=...
    )
    return tokenizer, engine

import re

def extract_answer_span(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else None

def save_outputs_to_jsonl(outputs: List[str], input_data: List[dict], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        for out, item in zip(outputs, input_data):
            answer = extract_answer_span(out)

            json.dump({
                "fact": item["fact"],
                "meta": item.get("meta", {}),
                "model_output": out,
                "answer_text": answer
            }, f, ensure_ascii=False)
            f.write("\n")

def make_item_id(item: dict) -> str:
    """稳定 ID：sha1(fact + json(meta))"""
    fact = item.get("fact", "")
    meta_str = json.dumps(item.get("meta", {}), ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1()
    h.update(fact.encode("utf-8"))
    h.update(b"||")
    h.update(meta_str.encode("utf-8"))
    return h.hexdigest()

def load_done_ids_from_jsonl(path: str) -> Set[str]:
    """从已存在的输出 jsonl 中收集已完成样本的 id。"""
    done: Set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                _id = obj.get("_id") or make_item_id({"fact": obj.get("fact",""), "meta": obj.get("meta", {})})
                done.add(_id)
            except Exception:
                continue
    return done

def append_result_line(fh, item: dict, model_output: str, *, failed: Optional[str]=None):
    rec = {
        "_id": make_item_id(item),
        "fact": item["fact"],
        "meta": item.get("meta", {}),
        "model_output": model_output if failed is None else f"❌ 推理失败: {failed}",
        "answer_text": extract_answer_span(model_output) if failed is None else None,
    }
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

def evaluate(cfg: RunConfig, dataset: List[dict], save_path: str, use_react: bool = True,
             flush_every: int = 10, resume: bool = True, state_path: Optional[str] = None):
    set_visible_gpus(cfg.visible_gpus)

    # 初始化后端
    if cfg.use_vllm:
        tokenizer, engine = load_vllm(cfg)
        backend = engine
    else:
        tokenizer, model = load_hf(cfg)
        backend = model

    # 已完成集合（断点）
    done_ids = load_done_ids_from_jsonl(save_path) if resume else set()
    skipped = 0

    # 侧车状态文件
    if state_path is None:
        state_path = save_path + ".state.json"

    # 以 append 模式打开输出文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fh = open(save_path, "a", encoding="utf-8")

    per_case_times = []
    processed_now = 0
    total_start = time.perf_counter()

    try:
        pbar = tqdm(dataset, desc=f"Evaluating ({'vLLM' if cfg.use_vllm else 'HF'} | {'ReAct' if use_react else 'Direct'})")
        for item in pbar:
            _id = make_item_id(item)
            if _id in done_ids:
                skipped += 1
                continue

            t0 = time.perf_counter()
            try:
                fact = item["fact"].strip()
                if use_react:
                    out = react_infer(tokenizer, backend, cfg, fact)
                else:
                    out = direct_infer(tokenizer, backend, cfg, fact)
                dt = time.perf_counter() - t0
                per_case_times.append(dt)
                append_result_line(fh, item, out)
            except Exception as e:
                dt = time.perf_counter() - t0
                per_case_times.append(dt)
                append_result_line(fh, item, "", failed=str(e))

            processed_now += 1
            # 定期 flush + 写入进度状态
            if processed_now % flush_every == 0:
                fh.flush()
                os.fsync(fh.fileno())
                with open(state_path, "w", encoding="utf-8") as sf:
                    json.dump({
                        "processed_now": processed_now,
                        "skipped": skipped,
                        "done_ids_count": len(done_ids) + processed_now,
                        "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, sf, ensure_ascii=False, indent=2)

        # 最终 flush
        fh.flush()
        os.fsync(fh.fileno())

    finally:
        fh.close()

    total_dt = time.perf_counter() - total_start

    # 打印计时信息
    print("\n========== Timing Summary ==========")
    print(f"Backend : {'vLLM 0.8.4' if cfg.use_vllm else 'HF generate'}")
    print(f"Mode    : {'ReAct (search loop)' if use_react else 'Direct inference'}")
    print(f"GPUs    : {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)') or 'CPU'}")
    if cfg.use_vllm:
        print(f"vLLM gpu_memory_utilization : {cfg.gpu_memory_utilization}")
        print(f"vLLM dtype                  : {cfg.dtype}")
    else:
        print(f"HF dtype : {cfg.dtype}")
    print(f"Total   : {total_dt:.2f}s  | processed_now={processed_now}  | skipped(existing)={skipped}")
    if per_case_times:
        print(f"Avg     : {sum(per_case_times)/len(per_case_times):.2f}s / sample")
        print(f"Min/Max : {min(per_case_times):.2f}s  /  {max(per_case_times):.2f}s")
    print("====================================\n")

    return {"processed_now": processed_now, "skipped": skipped, "total_time_s": total_dt}


def load_facts_from_jsonl(path: str) -> List[dict]:
    dataset = []
    datas = open(path, "r", encoding="utf-8").readlines()

    for data in datas:
        data = json.loads(data)
        dataset.append({
            "fact": data["fact"],
            "meta": data.get("meta", {})  # 可选：保留元信息
        })
    return dataset

import argparse

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in ("true", "1", "yes", "y"):
        return True
    if x in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Invalid boolean: {x}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with configurable settings.")

    # 添加配置参数
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--visible_gpus", type=str, default="0")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--react_max_turns", type=int, default=16)
    parser.add_argument("--print_each_output", action="store_true")

    # 输入输出参数
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output result JSONL.")
    parser.add_argument("--flush_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_react", action="store_true")
    parser.add_argument("--eval_prompt", type=int, default=0) # 0 就是默认，1 是增加了提示的版本
    parser.add_argument("--retrieve_statute", type=str2bool, default=True)
    parser.add_argument("--retrieve_guideline", type=str2bool, default=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cfg = RunConfig(
        model_id=args.model_id,
        use_vllm=args.use_vllm,
        visible_gpus=args.visible_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        react_max_turns=args.react_max_turns,
        print_each_output=args.print_each_output,
        eval_prompt=args.eval_prompt,
        retrieve_statute=args.retrieve_statute,
        retrieve_guideline=args.retrieve_guideline,
    )

    dataset = load_facts_from_jsonl(args.input_path)

    evaluate(
        cfg,
        dataset=dataset,
        save_path=args.output_path,
        use_react=args.use_react,
        flush_every=args.flush_every,
        resume=args.resume
    )