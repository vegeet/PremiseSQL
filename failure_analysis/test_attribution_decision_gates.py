"""Regression checks for deterministic M01/M04 decision controls."""

import copy
import unittest

from .runner import validate_response


def assessment(label):
    return {
        "label_id": label,
        "verdict": "not_supported",
        "context_basis": "none",
        "target_requirement": "",
        "target_status": "not_applicable",
        "implementation_scope": "not_applicable",
        "replacement_status": "not_applicable",
        "transformation": "none",
        "evidence": [],
        "context_evidence": "",
        "explanation": "The positive test is not met.",
    }


def response():
    return {
        "recognition": "not_explicit",
        "response_behavior": "sql_generated",
        "label_assessments": [assessment(label) for label in ("M01", "M02", "M03", "M04")],
        "novel_candidate": None,
        "evidence_limitations": "",
    }


def select(value, label):
    return next(item for item in value["label_assessments"] if item["label_id"] == label)


class DecisionGateChecks(unittest.TestCase):
    sample = {
        "question": "Which current courses use ratings?",
        "detection": 0,
        "type_identification": 0,
        "sql_behavior": 0,
    }
    output = {"reason": "The query uses failed tests.", "sql": "SELECT course_id FROM enrolment"}

    def support(self, value, label, target):
        item = select(value, label)
        item.update(
            verdict="supported",
            target_requirement=target,
            target_status="complete_explicit",
            implementation_scope="remaining_implementation",
            replacement_status="no_replacement",
            evidence=[
                {"field": "question", "quote": target},
                {"field": "reason", "quote": self.output["reason"]},
            ],
            explanation="The current requirement disappears from the partial implementation.",
        )
        return item

    def test_valid_m04_survives(self):
        value = response()
        self.support(value, "M04", "ratings")
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], ["M04"])

    def test_m04_rejects_incomplete_target(self):
        value = response()
        item = self.support(value, "M04", "current courses")
        item["target_status"] = "unresolved_or_incomplete"
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("complete explicit", select(result, "M04")["explanation"])

    def test_m04_rejects_lexically_incomplete_targets(self):
        cases = (
            ("How many students did Harvard have in ?", "in ?"),
            ("Which orders arrived during the specified period?", "during the specified period"),
            ("Which races occurred on a given day?", "on a given day"),
            ("What is the venue of the competition?", "the competition"),
            ("How many Indian restaurants are there in that area?", "in that area"),
            ("What is the difference for the inspection type?", "the inspection type"),
            ("Show the weight for a single shipment.", "a single shipment"),
            ("List at least students with no payment due.", "at least"),
        )
        for question, target in cases:
            with self.subTest(target=target):
                value = response()
                self.support(value, "M04", target)
                sample = {**self.sample, "question": question}
                result = validate_response(value, self.output, sample)
                self.assertEqual(result["supported_label_ids"], [])
                self.assertIn("syntactically incomplete", select(result, "M04")["explanation"])

    def test_m04_rejects_complete_abstention(self):
        value = response()
        item = self.support(value, "M04", "ratings")
        item["implementation_scope"] = "complete_abstention"
        sample = {**self.sample, "sql_behavior": 2}
        result = validate_response(value, self.output, sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("complete abstention", select(result, "M04")["explanation"])

    def test_m04_rejects_replacement(self):
        value = response()
        item = self.support(value, "M04", "ratings")
        item["replacement_status"] = "concrete_replacement"
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])

    def test_redundant_replacement_scope_is_ignored(self):
        value = response()
        item = self.support(value, "M04", "ratings")
        item["replacement_scope"] = "no_replacement"
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], ["M04"])

    def test_m01_requires_concrete_replacement(self):
        value = response()
        item = self.support(value, "M01", "current courses")
        item["target_status"] = "unresolved_or_incomplete"
        item["replacement_status"] = "no_replacement"
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("concrete replacement", select(result, "M01")["explanation"])

    def test_valid_m01_survives(self):
        value = response()
        item = self.support(value, "M01", "current courses")
        item.update(
            target_status="unresolved_or_incomplete",
            replacement_status="concrete_replacement",
            explanation="Failed tests are used as a concrete proxy for the requested prediction.",
        )
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], ["M01"])

    def test_explicit_self_rejection_cannot_remain_supported(self):
        value = response()
        item = self.support(value, "M04", "ratings")
        item["explanation"] = "After checking the query, M04 is not supported."
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("explicitly rejects", select(result, "M04")["explanation"])

    def test_no_mechanism_self_rejection_cannot_remain_supported(self):
        value = response()
        item = self.support(value, "M01", "current courses")
        item.update(
            replacement_status="concrete_replacement",
            explanation="The query keeps the ambiguity broad. No M01 mechanism is present.",
        )
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("explicitly rejects", select(result, "M01")["explanation"])

    def test_m04_rejects_explanation_that_identifies_replacement(self):
        value = response()
        item = self.support(value, "M04", "ratings")
        item["explanation"] = (
            "The SQL does implement a concrete replacement for ratings, so deletion applies."
        )
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("identifies a replacement", select(result, "M04")["explanation"])

    def test_m04_rejects_explanation_that_maps_target_to_category(self):
        value = response()
        item = self.support(value, "M04", "current courses")
        item["explanation"] = (
            "The output maps this to the broader CurrentOrPast category but calls it deletion."
        )
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("identifies a replacement", select(result, "M04")["explanation"])

    def test_reference_only_m04_claim_is_rejected(self):
        for reference_name in ("paired valid question", "paired_valid_question"):
            with self.subTest(reference_name=reference_name):
                value = response()
                item = self.support(value, "M04", "ratings")
                item["explanation"] = (
                    f"The {reference_name} specifies a title constraint that was removed, "
                    "so the current SQL omits it."
                )
                result = validate_response(value, self.output, self.sample)
                self.assertEqual(result["supported_label_ids"], [])
                self.assertIn("paired reference", select(result, "M04")["explanation"])

    def test_long_reference_only_m04_claim_is_rejected(self):
        value = response()
        item = self.support(value, "M04", "ratings")
        item["explanation"] = (
            "The paired valid question contains a count of three. "
            + ("The current output discusses other implementation details. " * 12)
            + "The flawed question removes that number, so M04 applies."
        )
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], [])
        self.assertIn("paired reference", select(result, "M04")["explanation"])

    def test_distinct_nested_targets_can_coexist(self):
        value = response()
        m01 = self.support(value, "M01", "current courses")
        m01.update(
            replacement_status="concrete_replacement",
            explanation="Failed tests replace the dropout forecast.",
        )
        self.support(value, "M04", "ratings")
        result = validate_response(copy.deepcopy(value), self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], ["M01", "M04"])

    def test_later_m01_evidence_rejection_does_not_suppress_m04(self):
        value = response()
        m01 = self.support(value, "M01", "ratings")
        m01.update(
            replacement_status="concrete_replacement",
            evidence=[{"field": "question", "quote": "ratings"}],
            explanation="A failed-test proxy replaces ratings.",
        )
        self.support(value, "M04", "ratings")
        result = validate_response(value, self.output, self.sample)
        self.assertEqual(result["supported_label_ids"], ["M04"])
        self.assertEqual(select(result, "M01")["verdict"], "uncertain")

    def test_distinct_m01_m03_mechanisms_can_share_output_quote(self):
        value = response()
        m01 = self.support(value, "M01", "current courses")
        m01.update(
            target_status="unresolved_or_incomplete",
            replacement_status="concrete_replacement",
            explanation="The output positively classifies failed tests as a course proxy.",
        )
        m03 = select(value, "M03")
        m03.update(
            verdict="supported",
            context_basis="bounded_examples",
            evidence=[{"field": "reason", "quote": self.output["reason"]}],
            context_evidence="The displayed rows are bounded examples.",
            explanation="The output also treats the displayed failed tests as exhaustive.",
        )
        sample = {**self.sample, "context_evidence": "At most five example rows are shown."}
        result = validate_response(value, self.output, sample)
        self.assertEqual(result["supported_label_ids"], ["M01", "M03"])


if __name__ == "__main__":
    unittest.main()
