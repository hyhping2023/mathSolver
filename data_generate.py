from openai import OpenAI
base_url = "https://api.chsdw.top/v1"
question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
# question = "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?"
# question = "已知函数 $f(x)=a x^{3}-3 x^{2}+1$, 若 $f(x)$ 存在唯一的零点 $x_{0}$, 且 $x_{0}>$ 0 , 则实数 $\\mathrm{a}$ 的取值范围是 ( )\n从以下选项中选择:\n(A) $(1,+\\infty)$\n(B) $(2,+\\infty)$\n(C) $(-\\infty,-1)$\n(D) $(-\\infty,-2)$"
# question = "设函数 $f(x)=\\left\\{\\begin{array}{ll}1+\\log _{2}(2-x), & x<1 \\ 2^{x-1}, & x \\geqslant 1,\\end{array}\\right.$ 则 $f(-2)+f\\left(\\log _{2} 12\\right)=$ ( )\n从以下选项中选择:\n(A) 3\n(B) 6\n(C) 9\n(D) 12"
# question = "Let $0(0,0)，A(tfrac{1}{2},0)'$ and $B(0,\tfrac{\sqrt{3}}{2})$be points in the coordinate plane. Let $lmathcal{F}$ be the familof segments $\overlinefPO}$ of unit length lying in the firstquadrant with $P$ on the $x$-axis and $o$ on the $y$-axis. Thereis a unique point $c$ on $loverline{AB}$, distinct from $A$ and$B$,that does not belong to any segment from $\mathcal{F}$ otherthan $loverline{AB}$, Then $0c^2 =\tfrac{p}ig}$, where $p$ and$o$ are relatively prime positive integers. Find $p + g$."

# agent = OpenAI(
#     model="gpt-4o-2024-08-06",
#     base_url=base_url,
#     temperature=0,
#     max_tokens=2000,
#     top_p=1,
# )

CONTENT = '''
Please help me answer the following question in just a few words. 
You need to think step by step and show your reasoning. 
If you think it would help to solve a problem, please generate a mathematical query which contains all needed information enclosed by <query> </query> tags.
You are encouraged to divide the original problem into more basic subproblems and solving basic problems in sequence multiple times in order to answer, so I will allow you to make up to {} sequential queries before answering the question.
Please do not repeat queries you have already issued, as this is a waste of time.
I will provide results in the following format:
<information>INFORMATION</information>.
Once you have enough information, generate an answer that only contains the number or the math expression or the correct option enclosed by <answer> </answer> tags.
Please either issue a search query or answer the question, but not both.
The question is: {}
'''

JUDGE = '''
I need you to help me grade the answer to the following question: "{}".
The answer key says: {}, and my answer is {}. Am I correct?
Please explain your reasoning and then answer "YES" or "NO".
There are multiple ways to write the same answer. For example, "10", "10.00", "$10", and "$10.00" are all equivalent.
'''

question = CONTENT.format(5, question)

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": question}
]

open_api_key = "EMPTY"
vllm_client = OpenAI(
    base_url="http://localhost:8005/v1",
    api_key=open_api_key,
)

while True:
    response = vllm_client.chat.completions.create(
                messages=messages,
                n=3,
                model="/data2/share/hanxu/hyh/Qwen/Qwen2.5-7B-Instruct/",
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                stop=["</query>", '</answer>'],
            )
    results = [response.message.content.strip() for response in response.choices]

    for i, result in enumerate(results):
        print("RESULT {}:\n".format(i), result)
    idx = int(input("\n\n\ninput the result index\n\n\n"))
    result = results[idx]
    if "<answer>" in result:
        result += "</answer>"
        break
    elif "<query>" in result:
        result, query = result.split("<query>")
        result = "<think>" + result.strip() + "</think>"
        query = "<query>" + query + "</query>"
        print("RESULT:\n", result)
        print("QUERY:\n", query)
        inforamtion = input("\n\n\ninput the query answer\n\n\n")
        new_message = question + result + query + "<information>" + inforamtion + "</information>"
        print("NEW_MESSAGE:\n", new_message)
        messages[1] = {"role": "user", "content": CONTENT.format(5, new_message)}
        question = new_message
    else:
        print("failed to parse the result")
        print(result)
        messages[1] = {"role": "user", "content": CONTENT.format(5, question+result)}

print(result)

