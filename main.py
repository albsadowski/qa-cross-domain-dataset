import json
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True, kw_only=True)
class Task:
    name: str
    path: Path
    question: str
    answers: list[tuple[str, str]]


def maud(task: str) -> Path:
    return Path("./legalbench/data") / f"maud_{task}"


MAUD_TASKS: Final[dict[str, Task]] = {
    "t1": Task(
        name="t1",
        path=maud("ability_to_consummate_concept_is_subject_to_mae_carveouts"),
        question='Is the "ability to consummate" concept subject to Material Adverse Effect (MAE) carveouts?',
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t2": Task(
        name="t2",
        path=maud("accuracy_of_fundamental_target_rws_bringdown_standard"),
        question="How accurate must the fundamental representations and warranties be according to the bring down provision?",
        answers=[
            ("A", "Accurate at another materiality standard (e.g., hybrid standard)"),
            ("B", "Accurate in all material respects"),
            ("C", "Accurate in all respects"),
        ],
    ),
    "t3": Task(
        name="t3",
        path=maud(
            "accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer"
        ),
        question="How accurate must the capitalization representations and warranties be according to the bring down provision?",
        answers=[
            ("A", "Accurate in all material respects"),
            ("B", "Accurate in all respects"),
            ("C", "Accurate in all respects with below-threshold carveout"),
            ("D", "Accurate in all respects with de minimis exception"),
        ],
    ),
    "t4": Task(
        name="t4",
        path=maud("accuracy_of_target_general_rw_bringdown_timing_answer"),
        question="When are representations and warranties required to be made according to the bring down provision?",
        answers=[
            ("A", "At Closing Only"),
            ("B", "At Signing & At Closing"),
        ],
    ),
    "t5": Task(
        name="t5",
        path=maud("additional_matching_rights_period_for_modifications_(cor)"),
        question="How long is the additional matching rights period for modifications in case the board changes its recommendation?",
        answers=[
            ("A", "2 business days or less"),
            ("B", "3 business days"),
            ("C", "3 days"),
            ("D", "4 business days"),
            ("E", "5 business days"),
            ("F", "> 5 business days"),
            ("G", "None"),
        ],
    ),
    "t6": Task(
        name="t6",
        path=maud(
            "application_of_buyer_consent_requirement_(negative_interim_covenant)"
        ),
        question="What negative covenants does the requirement of Buyer consent apply to?",
        answers=[
            ("A", "Applies only to specified negative covenants"),
            ("B", "Applies to all negative covenants"),
        ],
    ),
    "t7": Task(
        name="t7",
        path=maud("buyer_consent_requirement_(ordinary_course)"),
        question="In case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent?",
        answers=[
            (
                "A",
                "Yes. Consent may not be unreasonably withheld, conditioned or delayed.",
            ),
            ("B", "No."),
        ],
    ),
    "t8": Task(
        name="t8",
        path=maud("change_in_law__subject_to_disproportionate_impact_modifier"),
        question="Do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t9": Task(
        name="t9",
        path=maud(
            "changes_in_gaap_or_other_accounting_principles__subject_to_disproportionate_impact_modifier"
        ),
        question="Do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t10": Task(
        name="t10",
        path=maud("cor_permitted_in_response_to_intervening_event"),
        question="Is Change of Recommendation permitted in response to an intervening event?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t11": Task(
        name="t11",
        path=maud("cor_permitted_with_board_fiduciary_determination_only"),
        question="Is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t12": Task(
        name="t12",
        path=maud("cor_standard_(intervening_event)"),
        question="What standard should the board follow when determining whether to change its recommendation in response to an intervening event?",
        answers=[
            ("A", '"Breach" of fiduciary duties'),
            ("B", '"Inconsistent" with fiduciary duties'),
            ("C", '"Reasonably likely/expected breach" of fiduciary duties'),
            (
                "D",
                '"Reasonably likely/expected to be inconsistent" with fiduciary duties',
            ),
            ("E", '"Reasonably likely/expected violation" of fiduciary duties'),
            ("F", '"Required to comply" with fiduciary duties'),
            ("G", '"Violation" of fiduciary duties'),
            ("H", "More likely than not violate fiduciary duties"),
            ("I", "Other specified standard"),
        ],
    ),
    "t13": Task(
        name="t13",
        path=maud("cor_standard_(superior_offer)"),
        question="What standard should the board follow when determining whether to change its recommendation in connection with a superior offer?",
        answers=[
            ("A", '"Breach" of fiduciary duties'),
            ("B", '"Inconsistent" with fiduciary duties'),
            ("C", '"Reasonably likely/expected breach" of fiduciary duties'),
            (
                "D",
                '"Reasonably likely/expected to be inconsistent" with fiduciary duties',
            ),
            ("E", '"Reasonably likely/expected violation" of fiduciary duties'),
            ("F", '"Required to comply" with fiduciary duties'),
            ("G", '"Violation" of fiduciary duties'),
            ("H", "More likely than not violate fiduciary duties"),
            ("I", "None"),
            ("J", "Other specified standard"),
        ],
    ),
    "t14": Task(
        name="t14",
        path=maud("definition_contains_knowledge_requirement_-_answer"),
        question='What is the knowledge requirement in the definition of "Intervening Event"?',
        answers=[
            (
                "A",
                "Known, but consequences unknown or not reasonably foreseeable, at signing",
            ),
            ("B", "Known, but consequences unknown, at signing"),
            ("C", "Not known and not reasonably foreseeable at signing"),
            ("D", "Not known at signing"),
        ],
    ),
    "t15": Task(
        name="t15",
        path=maud("definition_includes_asset_deals"),
        question="What qualifies as a superior offer in terms of asset deals?",
        answers=[
            ("A", '"All or substantially all"'),
            ("B", "50%"),
            ("C", 'Greater than 50% but not "all or substantially all"'),
            ("D", "Less than 50%"),
        ],
    ),
    "t16": Task(
        name="t16",
        path=maud("definition_includes_stock_deals"),
        question="What qualifies as a superior offer in terms of stock deals?",
        answers=[
            ("A", '"All or substantially all"'),
            ("B", "50%"),
            ("C", 'Greater than 50% but not "all or substantially all"'),
            ("D", "Less than 50%"),
        ],
    ),
    "t17": Task(
        name="t17",
        path=maud("fiduciary_exception__board_determination_standard"),
        question="Under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision?",
        answers=[
            (
                "A",
                'If failure to take actions would lead to "breach" of fiduciary duties',
            ),
            (
                "B",
                'If failure to take actions would be "inconsistent" with fiduciary duties',
            ),
            (
                "C",
                'If failure to take actions would lead to "reasonably likely/expected breach" of fiduciary duties',
            ),
            (
                "D",
                'If failure to take actions would lead to "reasonably likely/expected to be inconsistent" with fiduciary duties',
            ),
            (
                "E",
                'If failure to take actions would lead to "reasonably likely/expected violation" of fiduciary duties',
            ),
            (
                "F",
                'If taking such actions is "required to comply" with fiduciary duties',
            ),
            (
                "G",
                'If failure to take actions would lead to "violation" of fiduciary duties',
            ),
            ("H", "Under no circumstances could the Board do so."),
            ("I", "Other circumstances"),
        ],
    ),
    "t18": Task(
        name="t18",
        path=maud("fiduciary_exception_board_determination_trigger_(no_shop)"),
        question="What type of offer could the Board take actions on notwithstanding the no-shop provision?",
        answers=[
            ("A", "Acquisition Proposal only"),
            (
                "B",
                "Superior Offer, or Acquisition Proposal reasonably likely/expected to result in a Superior Offer",
            ),
        ],
    ),
    "t19": Task(
        name="t19",
        path=maud("financial_point_of_view_is_the_sole_consideration"),
        question='Is "financial point of view" the sole consideration when determining whether an offer is superior?',
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t20": Task(
        name="t20",
        path=maud("fls_(mae)_standard"),
        question="What is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE)?",
        answers=[
            ("A", "'Could' (reasonably) be expected to"),
            ("B", "'Would'"),
            ("C", "'Would' (reasonably) be expected to"),
            ("D", "No"),
            ("E", "Other forward-looking standard"),
        ],
    ),
    "t21": Task(
        name="t21",
        path=maud(
            "general_economic_and_financial_conditions_subject_to_disproportionate_impact_modifier"
        ),
        question="Do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t22": Task(
        name="t22",
        path=maud("includes_consistent_with_past_practice"),
        question='Does the wording of the Efforts Covenant clause include "consistent with past practice"?',
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t23": Task(
        name="t23",
        path=maud("initial_matching_rights_period_(cor)"),
        question="How long is the initial matching rights period in case the board changes its recommendation?",
        answers=[
            ("A", "2 business days or less"),
            ("B", "3 business days"),
            ("C", "3 calendar days"),
            ("D", "4 business days"),
            ("E", "4 calendar days"),
            ("F", "5 business days"),
            ("G", "Greater than 5 business days"),
        ],
    ),
    "t24": Task(
        name="t24",
        path=maud("initial_matching_rights_period_(ftr)"),
        question="How long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR)?",
        answers=[
            ("A", "2 business days or less"),
            ("B", "3 business days"),
            ("C", "3 calendar days"),
            ("D", "4 business days"),
            ("E", "4 calendar days"),
            ("F", "5 business days"),
            ("G", "5 calendar days"),
            ("H", "Greater than 5 business days"),
        ],
    ),
    "t25": Task(
        name="t25",
        path=maud("intervening_event_-_required_to_occur_after_signing_-_answer"),
        question='Is an "Intervening Event" required to occur after signing?',
        answers=[
            ("A", "No. It may occur or arise prior to signing."),
            ("B", "Yes. It must occur or arise after signing."),
        ],
    ),
    "t26": Task(
        name="t26",
        path=maud("knowledge_definition"),
        question="What counts as Knowledge?",
        answers=[
            ("A", "Actual knowledge"),
            ("B", "Constructive knowledge"),
        ],
    ),
    "t27": Task(
        name="t27",
        path=maud(
            "liability_standard_for_no-shop_breach_by_target_non-do_representatives"
        ),
        question="What is the liability standard for no-shop breach by target non-DO representatives?",
        answers=[
            ("A", "Strict liability"),
            ("B", "Knowledge-based liability"),
        ],
    ),
    "t28": Task(
        name="t28",
        path=maud("ordinary_course_efforts_standard"),
        question="What is the efforts standard?",
        answers=[
            ("A", "Commercially reasonable efforts"),
            ("B", "Flat covenant (no efforts standard)"),
            ("C", "Reasonable best efforts"),
        ],
    ),
    "t29": Task(
        name="t29",
        path=maud(
            "pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier"
        ),
        question="Do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t30": Task(
        name="t30",
        path=maud(
            "pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures"
        ),
        question="Is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t31": Task(
        name="t31",
        path=maud("relational_language_(mae)_applies_to"),
        question="What carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?",
        answers=[
            ("A", "All MAE carveouts"),
            ("B", "No"),
            ("C", "Some MAE carveouts"),
        ],
    ),
    "t32": Task(
        name="t32",
        path=maud("specific_performance"),
        question="What is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach?",
        answers=[
            ("A", '"entitled to seek" specific performance'),
            ("B", '"entitled to" specific performance'),
        ],
    ),
    "t33": Task(
        name="t33",
        path=maud("tail_period_length"),
        question="How long is the Tail Period?",
        answers=[
            ("A", "12 months or longer"),
            ("B", "Other"),
            ("C", "within 12 months"),
            ("D", "within 6 months"),
            ("E", "within 9 months"),
        ],
    ),
    "t34": Task(
        name="t34",
        path=maud("type_of_consideration"),
        question="What type of consideration is specified in this agreement?",
        answers=[
            ("A", "All Cash"),
            ("B", "All Stock"),
            ("C", "Mixed Cash/Stock"),
            ("D", "Mixed Cash/Stock: Election"),
        ],
    ),
}


def format_table_as_text(table: list[list[str]]) -> str:
    if not table:
        return ""

    cleaned_table = []
    for row in table:
        cleaned_row = []
        for cell in row:
            cell_str = str(cell).strip()
            cleaned_row.append(cell_str)
        cleaned_table.append(cleaned_row)

    if not cleaned_table:
        return ""

    col_widths = []
    for col_idx in range(len(cleaned_table[0])):
        max_width = 0
        for row in cleaned_table:
            if col_idx < len(row):
                max_width = max(max_width, len(row[col_idx]))
        col_widths.append(max_width)

    formatted_lines = []
    for i, row in enumerate(cleaned_table):
        # Pad each cell to its column width
        padded_cells = []
        for j, cell in enumerate(row):
            if j < len(col_widths):
                padded_cells.append(cell.ljust(col_widths[j]))
            else:
                padded_cells.append(cell)

        formatted_line = " | ".join(padded_cells)
        formatted_lines.append(formatted_line)

        if i == 0 and len(cleaned_table) > 1:
            separator = " | ".join(["-" * width for width in col_widths])
            formatted_lines.append(separator)

    return "\n".join(formatted_lines)


def generate_plausible_answers(
    correct_answer: float, question: str
) -> list[tuple[str, str]]:
    variations = [
        round(correct_answer * 0.8, 1),
        round(correct_answer * 1.2, 1),
        round(correct_answer * 0.5, 1),
        round(correct_answer * 1.5, 1),
        round(correct_answer + 10, 1),
        round(correct_answer - 10, 1),
        round(correct_answer * -1, 1),
    ]
    variations = [v for v in set(variations) if v != correct_answer and v != 0]

    wrong_answers = random.sample(variations, min(3, len(variations)))

    while len(wrong_answers) < 3:
        if "percent" in question.lower() or "%" in question.lower():
            wrong_answers.append(round(correct_answer + random.uniform(-20, 20), 1))
        else:
            wrong_answers.append(round(correct_answer + random.uniform(-50, 50), 1))

    all_answers = [correct_answer] + wrong_answers[:3]
    random.shuffle(all_answers)

    letters = ["A", "B", "C", "D"]
    answer_choices = [(letters[i], str(all_answers[i])) for i in range(4)]

    correct_letter = None
    for letter, value in answer_choices:
        if float(value) == correct_answer:
            correct_letter = letter
            break

    return answer_choices, correct_letter


def process_finqa_record(record: dict[str, Any]) -> dict[str, Any] | None:
    pre_text = record.get("pre_text", [])
    post_text = record.get("post_text", [])
    table = record.get("table", [])
    qa = record.get("qa", {})

    text_parts = []

    if pre_text:
        text_parts.extend(pre_text)
        text_parts.append("")
    if table:
        text_parts.append(format_table_as_text(table))
        text_parts.append("")
    if post_text:
        text_parts.extend(post_text)

    formatted_text = "\n".join(text_parts)
    question = qa.get("question", "")

    try:
        if not qa["answer"]:
            correct_answer = qa["exe_ans"] * 100
            answer_choices, correct_letter = generate_plausible_answers(
                correct_answer, question
            )
        else:
            if qa["answer"] in ("yes", "no"):
                answer_choices = [("A", "yes"), ("B", "no")]
                correct_letter = 0 if qa["answer"] == "yes" else 1
            else:
                correct_answer = float(qa.get("answer", 0).replace("%", ""))
                answer_choices, correct_letter = generate_plausible_answers(
                    correct_answer, question
                )
    except Exception as e:
        print(f"WARN: {e}")
        return None

    return {
        "domain": "finance",
        "task_id": "finqa",
        "text": formatted_text,
        "question": question,
        "answers": json.dumps(answer_choices),
        "answer": correct_letter,
    }


def prep_legal() -> pd.DataFrame:
    dfs = []
    for task_name, task in MAUD_TASKS.items():
        df = pd.read_csv(task.path / "test.tsv", sep="\t", index_col="index")
        df["domain"] = "legal"
        df["task_id"] = f"maud:{task_name}"
        df["question"] = task.question
        df["answers"] = json.dumps(task.answers)
        dfs.append(df)

    maud_df = pd.concat(dfs, axis=0)
    maud_df.reset_index(drop=True, inplace=True)

    return maud_df


def prep_medical() -> pd.DataFrame:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    df = ds.to_pandas()[["pubid", "context", "question", "final_decision"]]

    def map_to_answer(decision: str) -> str:
        match decision:
            case "yes":
                return "A"
            case "no":
                return "B"
            case "maybe":
                return "C"
            case _:
                raise KeyError("Invalid decision")

    return pd.DataFrame(
        {
            "domain": "medical",
            "task_id": "pubmedqa",
            "text": df.apply(lambda row: "\n".join(row["context"]["contexts"]), axis=1),
            "question": df["question"],
            "answers": json.dumps([("A", "yes"), ("B", "no"), ("C", "maybe")]),
            "answer": df["final_decision"].map(map_to_answer),
        }
    )


def prep_financial() -> pd.DataFrame:
    with open("./FinQA/dataset/test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_records = []
    for record in data:
        processed_record = process_finqa_record(record)
        if processed_record is not None:
            processed_records.append(processed_record)

    return pd.DataFrame(processed_records)


def prep_reading_comprehension() -> pd.DataFrame:
    with open("./mctest/data/MCTest/mc500.test.tsv", "r", encoding="utf-8") as f:
        tsv_lines = f.read().strip().split("\n")
    with open("./mctest/data/MCTestAnswers/mc500.test.ans", "r", encoding="utf-8") as f:
        ans_lines = f.read().strip().split("\n")

        all_rows = []

    for tsv_line, ans_line in zip(tsv_lines, ans_lines):
        tsv_parts = tsv_line.split("\t")

        story = tsv_parts[2].replace("\\newline", "\n").replace("\\tab", "\t")

        questions_data = []
        for q_idx in range(4):
            start_idx = 3 + (q_idx * 5)
            if start_idx + 4 < len(tsv_parts):
                question_text = tsv_parts[start_idx]

                clean_question = question_text
                if question_text.startswith("one:"):
                    clean_question = question_text[4:].strip()
                elif question_text.startswith("multiple:"):
                    clean_question = question_text[9:].strip()

                answer_a = tsv_parts[start_idx + 1]
                answer_b = tsv_parts[start_idx + 2]
                answer_c = tsv_parts[start_idx + 3]
                answer_d = tsv_parts[start_idx + 4]

                questions_data.append(
                    {
                        "question": clean_question,
                        "choices": [
                            ("A", answer_a),
                            ("B", answer_b),
                            ("C", answer_c),
                            ("D", answer_d),
                        ],
                    }
                )

        correct_answers = ans_line.split("\t")

        for q_idx, question_data in enumerate(questions_data):
            if q_idx < len(correct_answers):
                correct_answer = correct_answers[q_idx].strip()

                row = {
                    "domain": "reading_comprehension",
                    "task_id": "mc500",
                    "text": story,
                    "question": question_data["question"],
                    "answers": json.dumps(question_data["choices"]),
                    "answer": correct_answer,
                }
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--target-size", default=600)
    parser.add_argument("--random-seed", default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    target_size = args.target_size
    random_seed = args.random_seed

    random.seed(random_seed)

    legal = prep_legal()
    medical = prep_medical()
    financial = prep_financial()
    rc = prep_reading_comprehension()

    balanced_dfs = []

    rc_balanced = rc.sample(n=target_size, replace=False, random_state=random_seed)
    balanced_dfs.append(rc_balanced)

    medical_balanced = medical.sample(
        n=target_size, replace=False, random_state=random_seed
    )
    balanced_dfs.append(medical_balanced)

    financial_balanced = financial.sample(
        n=target_size, replace=False, random_state=random_seed
    )
    balanced_dfs.append(financial_balanced)

    legal_tasks = legal["task_id"].unique()
    samples_per_task = target_size // len(legal_tasks)
    remainder = target_size % len(legal)

    legal_samples = []
    for i, task in enumerate(legal_tasks):
        task_df = legal[legal["task_id"] == task]
        n_samples = samples_per_task + (1 if i < remainder else 0)
        n_samples = min(n_samples, len(task_df))
        task_sample = task_df.sample(
            n=n_samples, replace=False, random_state=random_seed
        )
        legal_samples.append(task_sample)

    legal_balanced = pd.concat(legal_samples, ignore_index=True)
    balanced_dfs.append(legal_balanced)

    cross_domain_dataset = pd.concat(balanced_dfs, ignore_index=True)
    cross_domain_dataset = cross_domain_dataset.sample(
        frac=1, random_state=random_seed
    ).reset_index(drop=True)

    cross_domain_dataset.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
