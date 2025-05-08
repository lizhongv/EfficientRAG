TOKEN_LABELING_SYSTEM_MSG = """
你是一位出色的语言学家，擅长从文档中提取与特定问题相关的信息。
""".strip()

TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_MUSIQUE = """
你被分配了一个信息提取任务。
你的任务是从给定段落中提取单词，以便其他人仅使用你提取的单词就能回答问题。
我们将向你展示问题(以<Question>开头)、文档段落(以<Document>开头)和答案(以<Answer>开头)。
你应该逐步思考，提取与问题和答案相关的单词。
你的回答应包含<Thought>部分和最终的JSON格式提取单词(以<JSON_OUTPUT>开头，用```json和```包裹)。

你的<Thought>部分必须包含以下步骤：
1. 识别文档中与问题相关的信息
2. 识别文档中与答案相关的信息
3. 提取同时与问题和答案相关的单词
你的<JSON_OUTPUT>必须包含以下键：
- "question_related_words": 从段落中提取的与问题相关的单词列表字符串，用空格分隔
- "answer_related_words": 从段落中提取的与答案相关的单词列表字符串，用空格分隔
- "extracted_words": 从段落中提取的单词列表字符串，用空格分隔

请遵守以下准则：
- 不要重新排序、更改、遗漏或添加单词。保持与原段落相同。
- 仅识别和提取明确出现在问题或其答案中的单词，以及与问题或其答案强相关的单词。
- 绝不标注对问题或答案没有有意义贡献的单词。
- 仅提取段落中出现的单词。

以下是一些供你参考的示例：

<Question>: Alas在哪里被发现？
<Document>: Alas人：Alas人是居住在印度尼西亚亚齐省东南亚齐摄政区的民族。他们说Alas语，与巴塔克语有关。
<Answer>: 印度尼西亚
<Thought>:
1. 问题询问Alas的位置。文档中的"Alas"与问题相关
2. 文档中的"东南亚齐摄政区 亚齐 印度尼西亚"与答案相关
3. 提取的单词是"Alas 东南亚齐摄政区 亚齐 印度尼西亚"
<JSON_OUTPUT>:
```json
{{
    "question_related_words": "Alas",
    "answer_related_words": "东南亚齐摄政区 亚齐 印度尼西亚",
    "extracted_words": "Alas 东南亚齐摄政区 亚齐 印度尼西亚"
}}
```

<Question>: 阿拉伯语词典中Hindu的意思是什么？
<Document>: 印度教徒：Hindu一词源自印度-雅利安语和梵语Sindhu，意思是"大片水域"，涵盖"河流、海洋"。它被用作印度河的名称，也指其支流。Gavin Flood指出，'hindu'一词最早作为"居住在印度河(梵语：Sindhu)彼岸的人的波斯地理术语"出现，更具体地出现在大流士一世公元前6世纪的铭文中。吠陀中称为Sapta Sindhava的旁遮普地区，在Zend Avesta中称为Hapta Hindu。大流士一世公元前6世纪的铭文提到了Hi(n)dush省，指的是印度西北部。在8世纪的文本Chachnama中，印度人民被称为Hinduvān(印度教徒)，hindavī被用作印度的形容词。这些古代记录中的术语'Hindu'是一个民族地理术语，不指代宗教。阿拉伯语对应的Al-Hind同样指印度这个国家。
<Answer>: 印度这个国家
<Thought>:
1. 问题询问阿拉伯语词典中Hindu的意思。文档中的"Hindu"和"阿拉伯"与问题相关
2. 文档中的"印度这个国家"与答案相关
3. 提取的单词是"Hind 阿拉伯 印度这个国家"
<JSON_OUTPUT>:
```json
{{
    "question_related_words": "Hind 阿拉伯",
    "answer_related_words": "印度这个国家",
    "extracted_words": "Hind 阿拉伯 印度这个国家"
}}
```

现在轮到你了！

<Question>: {question}
<Document>: {paragraph}
<Answer>: {answer}
<Thought>:
""".strip()

TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_WIKIMQA = """
你被分配了一个信息提取任务。
你的任务是从给定段落中提取单词，以便其他人(GPT3.5)仅使用你提取的单词就能回答问题。
你提取的单词应涵盖问题和答案中的信息，包括实体(如人物、地点、电影)和核心关系。
你的响应应为JSON格式并包含以下键：
- "extracted_words": 从段落中提取的单词列表字符串，用空格分隔

请遵守以下准则：
- 不要重新排序、更改、遗漏或添加单词。保持与原段落相同。
- 仅识别和提取明确出现在问题或其答案中的单词，以及与问题或其答案强相关的单词。
- 绝不标注对问题或答案没有有意义贡献的单词。
- 仅提取段落中出现的单词。
- 尽可能少地提取单词。

问题：电影《波俄战争》的导演是谁？
段落：《波俄战争》(电影)：《波俄战争》(Wojna polsko-ruska)是2009年波兰电影，由Xawery Żuławski执导，改编自Dorota Masłowska的小说《白红旗下波俄战争》。
答案：Xawery Żuławski
你的响应：
```json
{{"extracted_words": "Alas 东南亚齐摄政区 亚齐 印度尼西亚"}}
```

问题：Elio Petri什么时候去世的？
段落：Elio Petri：Elio Petri(1929年1月29日-1982年11月10日)是意大利政治电影制作人，以1970年奥斯卡获奖电影《对一个不容怀疑的公民的调查》而闻名。
答案：1982年11月10日
你的响应：
```json
{{"extracted_words": "Elio Petri 1982年11月10日"}}
```

问题：《乔西的歌谣》什么时候发行的？
段落：《乔西的歌谣》：《乔西的歌谣》是1967年Technicolor美国喜剧西部片，由Andrew V. McLaglen执导，Doris Day、Peter Graves和George Kennedy主演。它以幽默的方式在传统西部片背景下探讨1960年代的女权主义主题。该片是William Talman的最后一部表演作品。影片在加利福尼亚州千橡市的两个地点拍摄：North Ranch和Wildwood Regional Park。
答案：1967年
你的响应：
```json
{{"extracted_words": "乔西的歌谣 1967年"}}
```

问题：{question}
段落：{paragraph}
答案：{answer}
你的响应：
""".strip()

TOKEN_LABEL_REDUNDANT_SYSTEM_MSG = """
你是一位出色的语言学家，擅长评估答案是否冗余。
""".strip()

TOKEN_LABEL_REDUNDANT_EVALUATION_PROMPT = """
你被分配了一个信息评估任务。
你的任务是评估提取的单词是否包含足够的信息来回答问题，以及提取的单词是否包含冗余。
我将向你提供问题、答案和提取的单词。你应该检查提取的单词是否包含无关信息，或者提取的单词是否遗漏了任何重要信息。

你的响应应为JSON格式并包含以下键：
- "redundant": 布尔值，表示提取的单词是否包含过多冗余信息
- "missing": 布尔值，表示提取的单词是否遗漏了任何重要信息

# 示例

问题：Alas在哪里被发现？
答案：印度尼西亚
提取的单词：Alas 东南亚齐摄政区 亚齐 印度尼西亚
你应该响应：
{{"redundant": false, "missing": false}}
解释：Alas在东南亚齐摄政区、亚齐、印度尼西亚被发现。提取的单词与问题和答案相关。

问题：阿拉伯语词典中Hindu的意思是什么？
答案：印度这个国家
提取的单词：阿拉伯 Hind 印度这个国家 8世纪文本 Chachnama
你应该响应：
{{"redundant": true, "missing": false}}
解释：单词"8世纪文本 Chachnama"与问题无关。

问题：歌曲《Green》的表演者是谁？
答案：Steve Hillage
提取的单词：Ron Ehrenreich 副总统 社会主义党 美国选举 Willa Kenoyer 新锡拉丘兹联邦信贷 绿党 Sondra Roth
你应该响应：
{{"redundant": true, "missing": true}}
解释：提取的单词与问题无关，且《Green》的表演者未包含在提取的单词中。

# 任务

问题：{question}
答案：{answer}
提取的单词：{extracted_words}
你的响应：
""".strip()
