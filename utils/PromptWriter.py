"""
Set of helper functions for writing prompts
"""
import utils
from transformers import PreTrainedTokenizer

Lean4_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

KIMINA_PROVER_SYS_PROMPT = """You are an expert in mathematics and Lean 4."""

OpenR1_SYS_PROMPT = """You are an expert in Lean4 theorem proving with exceptional strategic reasoning abilities. When solving problems, strictly follow this process:
1. First create a natural language proof draft explaining key insights and logical steps
2. Then analyze required Lean4 tactics, specifying exact syntax and their logical purpose"""

def formulate_prompt_existence_translation(
        QA_problem_to_transfer: str,
        tokenizer: PreTrainedTokenizer
) -> str:
    """
    This is the function for writing prompt to transform QA problem into existence problem.
    The default setting is to use Qwen3 tokenizer, other tokenizers may have problems
    """

    prompt = """@ Question-answering problem:
```md
Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
```

@ Existence problem:
```md
Prove the existence of the point in polar coordinates $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$ for the point $(0,3)$ in rectangular coordinates.
```

===

@ Question-answering problem:
```md
Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$
```

@ Existence problem:
```md
Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\] Prove that \[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\] can be represented in terms of $p$ and $q$.
```

===

@ Question-answering problem:
```md
If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.
```

@ Existence problem:
```md
Define $f(x) = \frac{3x-2}{x-2}$. Prove that there exists $ y = f(-2) + f(-1) + f(0)$ and $y$ can be expressed as a common fraction.
```

===""" + f"""

@ Question-answering problem:
```md
{QA_problem_to_transfer}
```

Based on the above examples, translate the QA problem into existence proble then, put it in a markdown code block as the examples."""
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return prompt
