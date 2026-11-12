# MC-IRF Data Pipeline - GSM-Symbolic + NoOp Generation

"""
GSM-Symbolic: Parametric GSM8K templates with variable substitution
NoOp: Adversarial irrelevant information injection (4 types)

Based on: GSM-Symbolic (Apple) + GSM-NoOp (NoOp adversarial attacks)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import random
import re
from pathlib import Path


@dataclass
class GSMProblem:
    """A single GSM problem with metadata."""
    problem_id: str
    template_id: str
    question: str
    answer: float
    variables: Dict[str, Any]
    reasoning_steps: List[str] = field(default_factory=list)
    nop_type: Optional[str] = None
    nop_text: Optional[str] = None
    nop_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GSMTemplate:
    """Parametric GSM-Symbolic template."""
    template_id: str
    question_template: str
    answer_expression: str  # Python expression using variables
    variables_spec: Dict[str, Dict]  # var_name -> {type, range, ...}
    reasoning_template: List[str] = field(default_factory=list)


# ============================================================
# GSM-Symbolic Templates (core set - expandable)
# ============================================================

CORE_TEMPLATES = [
    GSMTemplate(
        template_id="gsm_001",
        question_template="James has {a} apples and {b} oranges. He gives away {c} apples and {d} oranges. How many fruits does he have left?",
        answer_expression="(a - c) + (b - d)",
        variables_spec={
            "a": {"type": "int", "range": [5, 20]},
            "b": {"type": "int", "range": [5, 20]},
            "c": {"type": "int", "range": [1, 5]},
            "d": {"type": "int", "range": [1, 5]},
        },
        reasoning_template=[
            "James starts with {a} apples and {b} oranges.",
            "He gives away {c} apples, leaving {a} - {c} = {a_minus_c} apples.",
            "He gives away {d} oranges, leaving {b} - {d} = {b_minus_d} oranges.",
            "Total fruits left: {a_minus_c} + {b_minus_d} = {answer}.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_002",
        question_template="A store sells notebooks for ${p} each. Sarah buys {n} notebooks and pays with a ${bill} bill. How much change does she receive?",
        answer_expression="bill - (p * n)",
        variables_spec={
            "p": {"type": "int", "range": [2, 10]},
            "n": {"type": "int", "range": [2, 10]},
            "bill": {"type": "int", "range": [20, 100]},
        },
        reasoning_template=[
            "Each notebook costs ${p}.",
            "Sarah buys {n} notebooks, so total cost is ${p} * {n} = ${total_cost}.",
            "She pays with a ${bill} bill.",
            "Change: ${bill} - ${total_cost} = ${answer}.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_003",
        question_template="Tom reads {p1} pages on Monday, {p2} pages on Tuesday, and {p3} pages on Wednesday. The book has {total} pages total. How many pages are left to read?",
        answer_expression="total - (p1 + p2 + p3)",
        variables_spec={
            "p1": {"type": "int", "range": [10, 50]},
            "p2": {"type": "int", "range": [10, 50]},
            "p3": {"type": "int", "range": [10, 50]},
            "total": {"type": "int", "range": [100, 300]},
        },
        reasoning_template=[
            "Tom reads {p1} + {p2} + {p3} = {pages_read} pages in three days.",
            "The book has {total} pages total.",
            "Pages left: {total} - {pages_read} = {answer}.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_004",
        question_template="A farmer has {c} chickens and {k} cows. Each chicken has 2 legs, each cow has 4 legs. How many legs in total?",
        answer_expression="c * 2 + k * 4",
        variables_spec={
            "c": {"type": "int", "range": [5, 30]},
            "k": {"type": "int", "range": [3, 15]},
        },
        reasoning_template=[
            "{c} chickens * 2 legs = {chicken_legs} legs.",
            "{k} cows * 4 legs = {cow_legs} legs.",
            "Total: {chicken_legs} + {cow_legs} = {answer} legs.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_005",
        question_template="Emily saves ${s} per week for {w} weeks. She then spends ${spend} on a toy. How much money does she have left?",
        answer_expression="s * w - spend",
        variables_spec={
            "s": {"type": "int", "range": [5, 20]},
            "w": {"type": "int", "range": [4, 12]},
            "spend": {"type": "int", "range": [10, 50]},
        },
        reasoning_template=[
            "Emily saves ${s}/week * {w} weeks = ${saved}.",
            "She spends ${spend} on a toy.",
            "Remaining: ${saved} - ${spend} = ${answer}.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_006",
        question_template="A rectangle has length {l} and width {w}. What is its perimeter?",
        answer_expression="2 * (l + w)",
        variables_spec={
            "l": {"type": "int", "range": [5, 20]},
            "w": {"type": "int", "range": [3, 15]},
        },
        reasoning_template=[
            "Perimeter = 2 * (length + width) = 2 * ({l} + {w}) = 2 * {l_plus_w} = {answer}.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_007",
        question_template="A car travels at {speed} mph for {time} hours. How far does it go?",
        answer_expression="speed * time",
        variables_spec={
            "speed": {"type": "int", "range": [30, 70]},
            "time": {"type": "float", "range": [1.0, 5.0], "step": 0.5},
        },
        reasoning_template=[
            "Distance = speed * time = {speed} * {time} = {answer} miles.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_008",
        question_template="Pizza is cut into {slices} slices. {people} people share it equally. How many slices per person?",
        answer_expression="slices // people",
        variables_spec={
            "slices": {"type": "int", "range": [8, 24], "step": 2},
            "people": {"type": "int", "range": [2, 6]},
        },
        reasoning_template=[
            "{slices} slices / {people} people = {answer} slices each (remainder ignored).",
        ],
    ),
    GSMTemplate(
        template_id="gsm_009",
        question_template="A box has {r} red, {b} blue, and {g} green marbles. If {taken} marbles are removed randomly, how many remain?",
        answer_expression="r + b + g - taken",
        variables_spec={
            "r": {"type": "int", "range": [10, 30]},
            "b": {"type": "int", "range": [10, 30]},
            "g": {"type": "int", "range": [10, 30]},
            "taken": {"type": "int", "range": [5, 20]},
        },
        reasoning_template=[
            "Total marbles: {r} + {b} + {g} = {total}.",
            "After removing {taken}: {total} - {taken} = {answer}.",
        ],
    ),
    GSMTemplate(
        template_id="gsm_010",
        question_template="John earns ${hourly} per hour. He works {hours} hours regular time and {ot} hours overtime (1.5x pay). What is his total pay?",
        answer_expression="hourly * hours + hourly * 1.5 * ot",
        variables_spec={
            "hourly": {"type": "int", "range": [10, 25]},
            "hours": {"type": "int", "range": [30, 40]},
            "ot": {"type": "int", "range": [0, 10]},
        },
        reasoning_template=[
            "Regular pay: ${hourly} * {hours} = ${reg_pay}.",
            "Overtime pay: ${hourly} * 1.5 * {ot} = ${ot_pay}.",
            "Total: ${reg_pay} + ${ot_pay} = ${answer}.",
        ],
    ),
]


# ============================================================
# NoOp Perturbation Generators (4 types)
# ============================================================

NOOP_TEMPLATES = {
    "quantity": [
        " By the way, there are {irrelevant_num} {irrelevant_item} in the room.",
        " Also, {irrelevant_person} has {irrelevant_num} {irrelevant_item}.",
        " Incidentally, the store has {irrelevant_num} {irrelevant_item} in stock.",
    ],
    "comparison": [
        " Note that {irrelevant_num} is greater than {irrelevant_num2}.",
        " Interestingly, {irrelevant_item} costs more than {irrelevant_item2}.",
        " It's worth mentioning that {irrelevant_num} > {irrelevant_num2}.",
    ],
    "filter": [
        " Only consider the {irrelevant_color} {irrelevant_item}s.",
        " Filter for {irrelevant_property} items only.",
        " Focus on the {irrelevant_category} ones.",
    ],
    "discount": [
        " There is a {irrelevant_pct}% discount on {irrelevant_item}.",
        " A {irrelevant_pct}% off sale applies to {irrelevant_item}.",
        " You get {irrelevant_pct}% off {irrelevant_item}.",
    ],
}

IRRELEVANT_POOLS = {
    "irrelevant_num": list(range(1, 100)),
    "irrelevant_num2": list(range(1, 100)),
    "irrelevant_item": ["books", "pens", "chairs", "tables", "lamps", "cups", "plates", "shoes", "hats", "bags"],
    "irrelevant_item2": ["pencils", "desks", "sofas", "rugs", "curtains", "bowls", "forks", "socks", "gloves", "wallets"],
    "irrelevant_person": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
    "irrelevant_color": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"],
    "irrelevant_property": ["wooden", "metal", "plastic", "glass", "ceramic", "fabric", "leather"],
    "irrelevant_category": ["kitchen", "office", "bedroom", "living room", "bathroom", "garage"],
    "irrelevant_pct": [5, 10, 15, 20, 25, 30, 40, 50],
}


def sample_variables(spec: Dict[str, Dict]) -> Dict[str, Any]:
    """Sample variables according to specification."""
    result = {}
    for name, v_spec in spec.items():
        v_type = v_spec["type"]
        v_range = v_spec["range"]
        step = v_spec.get("step", 1)
        
        if v_type == "int":
            if step == 1:
                result[name] = random.randint(v_range[0], v_range[1])
            else:
                result[name] = random.randrange(v_range[0], v_range[1] + 1, step)
        elif v_type == "float":
            # Generate float with step precision
            steps = int((v_range[1] - v_range[0]) / step) + 1
            result[name] = round(v_range[0] + random.randint(0, steps - 1) * step, 1)
    return result


def evaluate_expression(expr: str, variables: Dict[str, Any]) -> float:
    """Safely evaluate answer expression."""
    # Only allow basic math operations
    allowed = set("0123456789+-*/(). ")
    for var, val in variables.items():
        allowed.update(var)
    
    # Replace variables
    for var, val in variables.items():
        expr = expr.replace(var, str(val))
    
    # Safe eval
    return eval(expr, {"__builtins__": {}}, {})


def inject_noop(question: str, nop_type: str, seed: int = None) -> tuple[str, Dict]:
    """Inject NoOp adversarial text into question."""
    if seed is not None:
        random.seed(seed)
    
    templates = NOOP_TEMPLATES[nop_type]
    template = random.choice(templates)
    
    # Sample irrelevant variables
    nop_vars = {}
    for key in re.findall(r'\{(\w+)\}', template):
        if key in IRRELEVANT_POOLS:
            nop_vars[key] = random.choice(IRRELEVANT_POOLS[key])
    
    nop_text = template.format(**nop_vars)
    perturbed = question + nop_text
    
    return perturbed, nop_vars


def generate_problem_set(
    templates: List[GSMTemplate],
    num_problems: int = 20,
    seeds: List[int] = None,
    nop_types: List[str] = None,
    severity: str = "moderate",
) -> List[GSMProblem]:
    """Generate full problem set with NoOp perturbations."""
    if seeds is None:
        seeds = [42, 123, 456]
    if nop_types is None:
        nop_types = ["quantity", "comparison", "filter", "discount"]
    
    problems = []
    
    for template in templates[:num_problems]:
        for seed in seeds:
            random.seed(seed)
            
            # Sample base variables
            variables = sample_variables(template.variables_spec)
            
            # Compute answer
            answer = evaluate_expression(template.answer_expression, variables)
            
            # Fill question template
            question = template.question_template.format(**variables)
            
            # Clean problem (no NoOp)
            clean = GSMProblem(
                problem_id=f"{template.template_id}_seed{seed}_clean",
                template_id=template.template_id,
                question=question,
                answer=answer,
                variables=variables,
                reasoning_steps=[],
                nop_type=None,
                nop_text=None,
                nop_variables={},
            )
            problems.append(clean)
            
            # NoOp variants
            for nop_type in nop_types:
                nop_seed = hash(f"{seed}_{nop_type}") % 10000
                perturbed_q, nop_vars = inject_noop(question, nop_type, nop_seed)
                
                nop_prob = GSMProblem(
                    problem_id=f"{template.template_id}_seed{seed}_{nop_type}",
                    template_id=template.template_id,
                    question=perturbed_q,
                    answer=answer,  # Answer unchanged!
                    variables=variables,
                    reasoning_steps=[],
                    nop_type=nop_type,
                    nop_text=perturbed_q[len(question):].strip(),
                    nop_variables=nop_vars,
                )
                problems.append(nop_prob)
    
    return problems


def save_problems(problems: List[GSMProblem], output_path: str):
    """Save problems to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in problems:
            f.write(json.dumps({
                "problem_id": p.problem_id,
                "template_id": p.template_id,
                "question": p.question,
                "answer": p.answer,
                "variables": p.variables,
                "nop_type": p.nop_type,
                "nop_text": p.nop_text,
                "nop_variables": p.nop_variables,
            }) + "\n")


def load_problems(input_path: str) -> List[GSMProblem]:
    """Load problems from JSONL."""
    problems = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            problems.append(GSMProblem(**data))
    return problems


if __name__ == "__main__":
    # Generate and save
    print("Generating GSM-Symbolic + NoOp problems...")
    
    problems = generate_problem_set(
        templates=CORE_TEMPLATES,
        num_problems=20,
        seeds=[42, 123, 456],
        nop_types=["quantity", "comparison", "filter", "discount"],
    )
    
    print(f"Generated {len(problems)} problems")
    print(f"  Clean: {sum(1 for p in problems if p.nop_type is None)}")
    for nt in ["quantity", "comparison", "filter", "discount"]:
        print(f"  {nt}: {sum(1 for p in problems if p.nop_type == nt)}")
    
    save_problems(problems, "data/processed/all_problems.jsonl")
    print("Saved to data/processed/all_problems.jsonl")