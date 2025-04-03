# given question, decomposed question, answer, e.t.c
# decompose the original question into successor sub_questions

hotpotQAFactPrompt = """
document for sub_question #{question_id}
supporting facts: {facts}"""


OriginHotpotQAPromptComparison = """You are assigned a multi-hop question decomposition task.
Your mission is to decompose a multi-hop question into a list of single-hop sub_questions based on supporting documents, and such that you (GPT-4) can answer each sub_question independently from each document.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. The sub_question MUST NOT contain more information than the original question and its dependent sub_question. NEVER introduce information from the documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": an empty list []. Because the sub_question is independent.

The origin multi-hop questions is: {question}
Followings are documents to answer each sub_question.
You MUST decompose the original multi-hoip question based on the given documents. DO NOT change the order or miss anyone of them.
{chunks}

Your output must always be a JSON object only, do not explain yourself or output anything else.
Follow the documents, synthesize the sub_questions and answers one-by-one. NEVER miss any of them.
"""

OriginHotpotQAPromptCompose = """You are assigned a multi-hop question decomposition task.
Your mission is to decompose a multi-hop question into a list of single-hop sub_questions based on supporting documents, and such that you (GPT-4) can answer each sub_question independently from each document.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. The sub_question MUST NOT contain more information than the original question and its dependent sub_question. NEVER introduce information from the documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions, you should list the sub_question number here.

The origin multi-hop questions is: {question}
And its answer is: {answer}
Followings are documents to answer each sub_question.
Make sure one sub_question depends on the other! Identify which sub_question depends on the answer of another according to the question.
You MUST decompose the question based on the documents with the sub_questions and answers. DO NOT change the order or miss anyone of them.
{chunks}

Your output must always be a JSON object only, do not explain yourself or output anything else.
Follow the documents, synthesize the sub_questions and answers one-by-one. NEVER miss any of them.
"""


HotpotQAPromptComparison = """You are assigned a multi-hop question decomposition task. Your mission is to break down the given multi-hop question into sub-questions based on the provided supporting documents, such that:
1. The number of sub-questions matches the number of supporting documents.
2. Each sub-question must be answered independently based on a single document.
3. The sub-questions must not rely on any other document or other sub-question.

Your output should be in the following JSON format:

1. The "question" key should contain the original multi-hop question.
2. The "decomposed_questions" key should contain a dictionary where each key is the sub-question number (e.g., "1", "2", ...) and each value is another dictionary with:
   - "sub_question": The decomposed single-hop sub-question derived from the corresponding document. This question should only rely on the information contained in the specific supporting document assigned to it.
   - "answer": The answer to the sub-question, derived from the specific supporting document assigned to it.
   - "dependency": This list must always be empty [] because each sub-question is independent and only relies on its assigned document.

**Here’s how you should proceed:**
1. Read the original multi-hop question carefully to understand the context.
2. Carefully read each supporting document provided. Each document will correspond to a single sub-question.
3. For each document, form a sub-question that can be answered solely based on that document. Ensure that the sub-question doesn’t introduce information not present in the document.
4. Maintain the order of the documents and ensure each sub-question corresponds to exactly one document. The number of sub-questions should match the number of supporting documents.
5. Each sub-question should have its own independent answer derived from the specific document assigned to it.

**Important Points to Remember:**
- DO NOT introduce new information beyond what is provided in each specific supporting document.
- DO NOT change the order of the documents or skip any of them.
- Each sub-question must be directly answerable from its respective document.
- The "dependency" list must be empty [] for each sub-question, as each is independent.

### The original multi-hop question is: {question}
### The supporting documents for the sub-questions are as follows:
{chunks}

Please output only the JSON object in the specified format, without any additional explanations or text."""


HotpotQAPromptCompose="""You are assigned a multi-hop question decomposition task. Your mission is to decompose a multi-hop question into a list of single-hop sub-questions based on the supporting documents, such that:

1. The number of sub-questions matches the number of supporting documents.
2. Each sub-question must be answered independently based on its respective supporting document, but some sub-questions may depend on the answers to previous sub-questions.
3. If a sub-question relies on previous sub-question answers, record those dependencies in the `dependency` key.

Your output should be in the following JSON format:

- `"question"`: a string, the original multi-hop question.
- `"decomposed_questions"`: a dictionary where each key is the sub-question number (e.g., "1", "2", ...) and each value is another dictionary containing:
    - `"sub_question"`: a string, the decomposed single-hop sub-question based on the corresponding document. This sub-question must not introduce any additional information beyond the original question or its dependent sub-question.
    - `"answer"`: a string, the answer derived from the corresponding supporting document.
    - `"dependency"`: a list of sub-question numbers (strings) that the sub-question relies on for its answer. If the sub-question relies on the answer from another sub-question, list those sub-question numbers here (e.g., `["1", "3"]`). If it does not rely on any previous sub-questions, the list should be empty `[]`.

### Instructions:
1. Carefully read the original multi-hop question and understand the overall context.
2. Analyze each supporting document carefully. Each document corresponds to a single sub-question.
3. For each document, form a sub-question that can be answered independently from that document. If a sub-question requires information from other sub-questions to be answered, record those dependencies.
4. The number of sub-questions should match the number of supporting documents, and the order of the documents must not be changed.
5. For each sub-question, ensure the `dependency` list is correctly populated based on the sub-questions it relies on.

### Important Points to Remember:
- **Do not introduce any new information** beyond what is provided in the supporting documents or the original question.
- **Do not change the order** of the documents or skip any of them.
- **Do not add any explanations**—only provide the output in the specified JSON format.
- The "dependency" list should reflect whether a sub-question relies on other sub-question answers.

### The original multi-hop question is: {question}
### The supporting documents for the sub-questions are as follows:
{chunks}

Please output only the JSON object in the specified format, without any additional explanations or text."""