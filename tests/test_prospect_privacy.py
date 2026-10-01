import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


SENSITIVE_FIELDS = {
    "billing_qualification",
    "tax_id",
    "date_of_birth",
    "card_on_file",
    "credit_check_ref",
}


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def assert_no_sensitive_fields(self, value):
        serialized = json.dumps(value)
        for field in SENSITIVE_FIELDS:
            self.assertNotIn(field, serialized)

    def test_source_and_direct_lookup_are_sanitized(self):
        record = data_service.get_prospect_record("LEAD-12853")
        result = get_prospect.invoke({"prospect_id": "LEAD-12853"})

        self.assert_no_sensitive_fields(record)
        self.assert_no_sensitive_fields(result)
        self.assertEqual(record["annual_revenue"], 90000000)

    def test_cached_profiles_are_sanitized_on_save_and_read(self):
        data_service.save_profile_to_db(
            "LEAD-12853",
            {
                "prospect_id": "LEAD-12853",
                "annual_revenue": 90000000,
                "billing_qualification": {
                    "tax_id": "541-88-2307",
                    "date_of_birth": "1984-03-11",
                    "card_on_file": "4539882100437712",
                    "credit_check_ref": "EXPN-88213",
                },
            },
        )

        result = data_service.get_profile_from_db("LEAD-12853")

        self.assert_no_sensitive_fields(result)
        self.assert_no_sensitive_fields(data_service._PROFILES)

    def test_built_profiles_are_sanitized(self):
        result = build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})

        self.assert_no_sensitive_fields(result)
        self.assert_no_sensitive_fields(data_service._PROFILES)

    def test_scorer_receives_only_allowlisted_data(self):
        prospect = {
            "prospect_id": "LEAD-12853",
            "annual_revenue": 90000000,
            "tech_stack": ["Snowflake"],
            "segment": "Enterprise",
            "billing_qualification": {
                "tax_id": "541-88-2307",
                "date_of_birth": "1984-03-11",
                "card_on_file": "4539882100437712",
                "credit_check_ref": "EXPN-88213",
            },
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 50000000,
            "target_segment": "Enterprise",
            "description": "Sensitive offering detail that is not needed for scoring",
        }

        with patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value.model_dump.return_value = {"score": 90}
            score_prospect.invoke({"prospect": prospect, "offering": offering})

        messages = scoring_llm.invoke.call_args.args[0]
        prompt_text = json.dumps(messages)
        self.assert_no_sensitive_fields(messages)
        self.assertNotIn("prospect_id", prompt_text)
        self.assertNotIn("description", prompt_text)
        self.assertIn("annual_revenue", prompt_text)
        self.assertIn("target_segment", prompt_text)


if __name__ == "__main__":
    unittest.main()
