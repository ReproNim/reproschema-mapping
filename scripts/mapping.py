import json
import pandas as pd
from rs2nda.semantic_matcher import SemanticMatcher
from rs2nda.response_mapper import ResponseMapper
from rs2nda.utils import extract_reproschema_responses

def map_schema(cde_csv_path: str, reproschema_response_path: str, cde_template_path: str):
    # Load CDE definitions and template
    cde_definitions = pd.read_csv(cde_csv_path)
    # Read template CSV using the second row as headers
    template_df = pd.read_csv(cde_template_path, header=1)
    template_columns = template_df.columns.tolist()
    print(template_columns)

    # Load ReproSchema responses
    with open(reproschema_response_path) as f:
        response_data = json.load(f)

    # Extract ReproSchema responses
    reproschema_responses = extract_reproschema_responses(response_data)
    print(reproschema_responses)

    # In your mapping script
    semantic_matcher = SemanticMatcher()

    matched_mapping = semantic_matcher.match(
        cde_definitions=cde_definitions,
        reproschema_responses=reproschema_responses
    )

    print(matched_mapping)
    # Map responses
    response_mapper = ResponseMapper(cde_definitions)
    mapped_data = response_mapper.map_responses(reproschema_responses, matched_mapping)

    template_row = response_mapper.create_template_row(mapped_data, template_columns)

    # You can then write this to your output file
    # For example, using pandas:
    df = pd.DataFrame([template_row], columns=template_columns)
    df.to_csv("output_template.csv", index=False)

if __name__ == "__main__":
    cde_csv_path = "data/nda_cde/cde_dsm5crosspg01_definitions.csv"
    reproschema_response_path = "data/rs-response1/activity_1.jsonld"
    cde_template_path = "data/nda_cde/cde_dsm5crosspg01_template.csv"
    result_df = map_schema(cde_csv_path, reproschema_response_path, cde_template_path)
    print(result_df)