import asyncio
import json
import pandas as pd
from pathlib import Path
from rs2nda import SemanticMatcher, ResponseMapper, extract_reproschema_responses

async def map_schema(cde_csv_path: str, reproschema_response_path: str, cde_template_path: str):
    # Convert paths to Path objects for better path handling
    cde_csv = Path(cde_csv_path)
    response_path = Path(reproschema_response_path)
    template_path = Path(cde_template_path)
    
    # Check if files exist
    if not all(p.exists() for p in [cde_csv, response_path, template_path]):
        raise FileNotFoundError("One or more input files not found")

    # Load CDE definitions and template
    cde_definitions = pd.read_csv(cde_csv)
    template_df = pd.read_csv(template_path, header=1)
    template_columns = template_df.columns.tolist()

    # Load ReproSchema responses
    with open(response_path) as f:
        response_data = json.load(f)

    # Extract ReproSchema responses
    reproschema_responses = extract_reproschema_responses(response_data)
    # print("reproschema_responses:" , reproschema_responses)

    # Initialize semantic matcher and get matches
    semantic_matcher = SemanticMatcher()
    matched_mapping = semantic_matcher.match(
        cde_definitions=cde_definitions,
        reproschema_responses=reproschema_responses
    )
    # print("matched_mapping:" , matched_mapping)
    # Map responses
    response_mapper = ResponseMapper(cde_definitions)
    # print("response_mapper:" , response_mapper)
    mapped_data = await response_mapper.map_responses(reproschema_responses, matched_mapping)
    # Create template row
    template_row = response_mapper.create_template_row(mapped_data, template_columns)

    # Create output DataFrame
    df = pd.DataFrame([template_row], columns=template_columns)
    
    # Save to the same directory as the input files
    output_path = "output_template.csv"
    df.to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")
    
    return df

def main():
    cde_csv_path = "data/nda_cde/cde_dsm5crosspg01_definitions.csv"
    reproschema_response_path = "data/rs-response1/activity_1.jsonld"
    cde_template_path = "data/nda_cde/cde_dsm5crosspg01_template.csv"
    
    try:
        result_df = asyncio.run(map_schema(cde_csv_path, reproschema_response_path, cde_template_path))
        print("Mapping completed successfully")
        print(result_df)
    except Exception as e:
        print(f"Error during mapping: {e}")
        raise

if __name__ == "__main__":
    main()