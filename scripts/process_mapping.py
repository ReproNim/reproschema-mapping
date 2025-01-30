import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
from rs2nda import ResponseMapper, QuestionMatcher, extract_reproschema_responses
import logging

class ResponseProcessor:
    def __init__(self, response_dir: Path, cde_dir: Path, output_dir: Path):
        self.response_dir = Path(response_dir)
        self.cde_dir = Path(cde_dir)
        self.output_dir = Path(output_dir)
        self.subject_info = None
        self.processed_files: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Store subject data separately
        self.interview_age = None
        self.sex = None
        
        self.activity_to_cde_map = {
            "demo_schema": "ndar_subject01",
            "DSM5_crosscutting_adult_schema": "cde_dsm5crossad01",
            "dsm_5_parent_guardian_rated_level_1_crosscutting_s": "cde_dsm5crosspg01",
            "DSM5_crosscutting_youth_schema": "cde_dsm5crossyouth01",
            "WHODAS12_schema": "cde_whodas01",
            "PHQ9_schema": "cde_phq901",
            "GAD7_schema": "cde_gad701",
            "RCADS25_caregiver_administered_schema": "cde_rcads01",
            "RCADS25_youth_administered_schema": "cde_rcadsyouth01"
        }

    async def map_schema(self, cde_csv_path: str, reproschema_response_path: str, cde_template_path: str):
        """Map a single schema response to CDE format."""
        # Load CDE definitions and template
        cde_definitions = pd.read_csv(cde_csv_path)
        template_df = pd.read_csv(cde_template_path, header=1)
        template_columns = template_df.columns.tolist()

        # Load ReproSchema responses
        with open(reproschema_response_path) as f:
            response_data = json.load(f)

        # Extract the reproschema responses
        processed_responses = extract_reproschema_responses(response_data)

        # Get matches using semantic matcher - ADD await HERE
        question_matcher = QuestionMatcher()
        matched_mapping = await question_matcher.match(  # Add await here
            cde_definitions=cde_definitions,
            reproschema_responses=processed_responses
        )

        # Map responses using response mapper
        response_mapper = ResponseMapper(cde_definitions)
        mapped_data = await response_mapper.map_responses(processed_responses, matched_mapping)

        # Create template row
        template_row = response_mapper.create_template_row(mapped_data, template_columns)

        # Create output DataFrame
        return pd.DataFrame([template_row], columns=template_columns)

    def get_activity_name(self, response_file: Path) -> Optional[str]:
        """Extract activity name from response data's 'used' field."""
        with open(response_file) as f:
            response_data = json.load(f)
            
        if not response_data or 'used' not in response_data[0]:
            return None
        
        for used_path in response_data[0]['used']:
            for schema_name in self.activity_to_cde_map.keys():
                if schema_name in used_path:
                    return schema_name
        
        # Instead of raising an error, return None for unknown activities
        return None

    def should_combine_activities(self, activity_name: str) -> bool:
        """Check if this activity should be combined with another one."""
        return "dsm_5_parent_guardian_rated_level_1_crosscutting_s" in activity_name
    
    def _validate_subject_info(self, df: pd.DataFrame) -> None:
        """Validate extracted subject information."""
        if 'interview_age' in df.columns:
            age = df['interview_age'].iloc[0]
            try:
                age_val = float(age)
                if age_val < 0 or age_val > 150:  # reasonable age range
                    self.logger.warning(f"Suspicious interview_age value: {age}")
                    df['interview_age'] = '-9'
            except ValueError:
                self.logger.warning(f"Invalid interview_age value: {age}")
                df['interview_age'] = '-9'
                
        if 'sex' in df.columns:
            sex = df['sex'].iloc[0]
            valid_sex_values = {'M', 'F', 'NR'}
            if sex not in valid_sex_values:
                # Check if it's a schema.org value
                sex_mapping = {
                    "http://schema.org/Female": "F",
                    "http://schema.org/Male": "M",
                    "Female": "F",
                    "Male": "M"
                }
                mapped_sex = sex_mapping.get(sex)
                if mapped_sex:
                    df['sex'] = mapped_sex
                    self.logger.info(f"Mapped sex value from {sex} to {mapped_sex}")
                else:
                    self.logger.warning(f"Invalid sex value: {sex}")
                    df['sex'] = 'NR'

    async def process_demo_schema(self, response_file: Path) -> None:
        """Process demo_schema and store subject information."""
        self.logger.info(f"Processing demo schema from {response_file}")
        
        demo_def_path = self.cde_dir / "ndar_subject01_definitions.csv"
        demo_template_path = self.cde_dir / "ndar_subject01_template.csv"
        
        try:
            # Load CDE definitions to get mappings
            cde_definitions = pd.read_csv(demo_def_path)
            
            # Get CDE mappings
            response_mapper = ResponseMapper(cde_definitions)
            interview_age_mapping = response_mapper._get_cde_mapping('interview_age')
            
            # Map the schema
            self.subject_info = await self.map_schema(
                str(demo_def_path),
                str(response_file),
                str(demo_template_path)
            )
            
            self.logger.info("Initial mapped values:")
            self.logger.info(self.subject_info[['interview_age', 'sex']].iloc[0].to_dict())
            
            # Validate extracted information
            if 'interview_age' in self.subject_info.columns:
                age = self.subject_info['interview_age'].iloc[0]
                try:
                    age_val = float(age)
                    # Use min_value and max_value from mapping
                    if (interview_age_mapping.min_value <= age_val <= interview_age_mapping.max_value):
                        self.logger.info(f"Valid interview_age value: {age}")
                    else:
                        self.logger.warning(f"Interview_age {age} outside valid range ({interview_age_mapping.min_value}-{interview_age_mapping.max_value})")
                        self.subject_info['interview_age'] = '-9'
                except ValueError:
                    self.logger.warning(f"Invalid interview_age format: {age}")
                    self.subject_info['interview_age'] = '-9'
            else:
                self.logger.warning("Missing interview_age")
                
            # Log final extracted values
            self.logger.info(f"Extracted values after validation:")
            for field in ['interview_age', 'sex']:
                if field in self.subject_info.columns:
                    value = self.subject_info[field].iloc[0]
                    if value == '-9':
                        self.logger.warning(f"{field} is -9 in demo schema")
                    else:
                        self.logger.info(f"Successfully extracted {field}: {value}")
                
            self.demo_processed = True
            self.logger.info("Demo schema processed successfully")
            print("Stored subject info:", 
                self.subject_info[['interview_age', 'sex']].iloc[0].to_dict())
                
        except Exception as e:
            self.logger.error(f"Error processing demo schema: {str(e)}")
            raise

    async def process_single_response(self, response_file: Path) -> None:
        """Process a single response file."""
        activity_name = self.get_activity_name(response_file)
        if not activity_name:
            return

        if activity_name == "demo_schema":
            await self.process_demo_schema(response_file)
            return

        cde_name = self.activity_to_cde_map[activity_name]
        output_file = self.output_dir / f"{cde_name}_output.csv"

        try:
            # Process the response
            cde_def_path = self.cde_dir / f"{cde_name}_definitions.csv"
            cde_template_path = self.cde_dir / f"{cde_name}_template.csv"

            result_df = await self.map_schema(
                str(cde_def_path),
                str(response_file),
                str(cde_template_path)
            )

            # Add subject information and handle combined activities
            result_df = self._add_subject_info(result_df)
            
            if self.should_combine_activities(activity_name):
                result_df = self._handle_combined_activities(result_df, output_file)

            # Verify critical fields before saving
            for field in ['interview_age', 'sex']:
                if field not in result_df.columns or result_df[field].iloc[0] == '-9':
                    self.logger.warning(f"{field} is missing or -9 in {activity_name}")

            result_df.to_csv(output_file, index=False)
            self.processed_files.add(output_file)
            self.logger.info(f"Successfully processed {activity_name} to {output_file}")
                
        except Exception as e:
            self.logger.error(f"Error processing {activity_name}: {str(e)}")
            raise

    def _add_subject_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add subject information to the dataframe."""
        if self.subject_info is not None:
            subject_columns = ['subjectkey', 'src_subject_id', 'interview_age', 'interview_date', 'sex']
            for col in subject_columns:
                if col in self.subject_info.columns:
                    df[col] = self.subject_info[col].iloc[0]
                    if col in ['interview_age', 'sex']:
                        self.logger.info(f"Added {col} = {self.subject_info[col].iloc[0]}")
        else:
            self.logger.warning("No subject info available")
            # Add default -9 values for required fields
            for col in ['interview_age', 'sex']:
                df[col] = '-9'
                
        print("Current subject info:")
        print(df[['subjectkey', 'src_subject_id', 'interview_age', 'sex']].head())
        
        return df

    def _handle_combined_activities(self, new_df: pd.DataFrame, output_file: Path) -> pd.DataFrame:
        """Handle merging of combined activities."""
        if output_file.exists() and output_file in self.processed_files:
            existing_df = pd.read_csv(output_file)
            
            # Create merged DataFrame
            merged_df = pd.DataFrame()
            
            # First, copy subject information, preferring non-'-9' values
            subject_cols = ['subjectkey', 'src_subject_id', 'interview_age', 'interview_date', 'sex']
            for col in subject_cols:
                if col in new_df.columns and new_df[col].iloc[0] != '-9':
                    merged_df[col] = new_df[col]
                elif col in existing_df.columns and existing_df[col].iloc[0] != '-9':
                    merged_df[col] = existing_df[col]
                else:
                    merged_df[col] = '-9'
            
            # Then merge the non-subject columns from both dataframes
            for df in [existing_df, new_df]:
                for col in df.columns:
                    if col not in subject_cols and df[col].notna().any():
                        merged_df[col] = df[col]
            
            return merged_df
        return new_df

    async def process_responses(self) -> None:
        """Process all response files in the directory."""
        try:
            # First, find and process demo schema
            response_files = sorted(self.response_dir.glob("*.jsonld"))
            demo_file = next((f for f in response_files 
                            if "demo_schema" in self.get_activity_name(f)), None)
            
            if demo_file:
                self.logger.info("Found demo schema, processing it first")
                await self.process_single_response(demo_file)
                response_files = [f for f in response_files if f != demo_file]
            else:
                self.logger.warning("No demo schema found")
            
            # Then process the rest
            for response_file in response_files:
                await self.process_single_response(response_file)
                
        except Exception as e:
            self.logger.error(f"Error in process_responses: {str(e)}")
            raise

async def debug_mapping(self, response_file: Path) -> None:
    """Debug mapping for a single response file"""
    try:
        # 1. Load and validate response file
        with open(response_file) as f:
            response_data = json.load(f)
        print(f"Loaded response file: {response_file}")
        
        # 2. Extract activity name
        activity_name = self.get_activity_name(response_file)
        print(f"Activity name: {activity_name}")
        
        if not activity_name:
            print("Could not determine activity name")
            return
            
        # 3. Get corresponding CDE name
        cde_name = self.activity_to_cde_map.get(activity_name)
        print(f"CDE name: {cde_name}")
        
        if not cde_name:
            print("No CDE mapping found")
            return
            
        # 4. Load CDE definitions
        cde_def_path = self.cde_dir / f"{cde_name}_definitions.csv"
        cde_template_path = self.cde_dir / f"{cde_name}_template.csv"
        
        print(f"\nAttempting to load:")
        print(f"Definitions: {cde_def_path}")
        print(f"Template: {cde_template_path}")
        
        # 5. Process response
        result_df = await self.map_schema(
            str(cde_def_path),
            str(response_file),
            str(cde_template_path)
        )
        
        print("\nMapping results:")
        print(result_df.head())
        
    except Exception as e:
        print(f"Error in debug_mapping: {str(e)}")

async def debug_pipeline(response_dir: Path, cde_dir: Path, output_dir: Path, debug_mode: str = "full") -> None:
    """
    Debug the mapping pipeline with different levels of detail
    
    Args:
        response_dir: Directory containing response files
        cde_dir: Directory containing CDE files
        output_dir: Directory for output files
        debug_mode: Level of debugging ("full", "single", or "summary")
    """
    processor = ResponseProcessor(response_dir, cde_dir, output_dir)
    
    if debug_mode == "single":
        # Debug first response file only
        response_files = sorted(response_dir.glob("*.jsonld"))
        if response_files:
            print("\n=== Debug Single Response ===")
            await processor.debug_mapping(response_files[0])
    
    elif debug_mode == "full":
        # Debug all files with detailed logging
        print("\n=== Full Debug Mode ===")
        response_files = sorted(response_dir.glob("*.jsonld"))
        for response_file in response_files:
            print(f"\nProcessing: {response_file.name}")
            await processor.debug_mapping(response_file)
            
    elif debug_mode == "summary":
        # Process all files but only log summary statistics
        print("\n=== Summary Debug Mode ===")
        response_files = sorted(response_dir.glob("*.jsonld"))
        success_count = 0
        error_count = 0
        
        for response_file in response_files:
            try:
                await processor.process_single_response(response_file)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error processing {response_file.name}: {str(e)}")
                
        print(f"\nProcessing Summary:")
        print(f"Successful: {success_count}")
        print(f"Failed: {error_count}")
        print(f"Total: {len(response_files)}")
        
        # Print cache statistics if using URLCache
        if hasattr(processor, 'url_cache'):
            stats = processor.url_cache.get_stats()
            print("\nCache Statistics:")
            print(f"Cache size: {stats['size']}")
            print(f"Cache hits: {stats['hits']}")
            print(f"Cache misses: {stats['misses']}")
            print(f"Hit ratio: {stats['hit_ratio']:.2%}")

def main():
    """Main function with integrated debugging options"""
    response_dir = Path("data/rs-response1")
    cde_dir = Path("data/nda_cde")
    output_dir = Path("output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Set up argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Process ReproSchema responses to CDE format')
    parser.add_argument('--debug', choices=['full', 'single', 'summary', 'none'], 
                      default='none', help='Debug mode')
    parser.add_argument('--response-dir', type=Path, default=response_dir,
                      help='Directory containing response files')
    parser.add_argument('--cde-dir', type=Path, default=cde_dir,
                      help='Directory containing CDE files')
    parser.add_argument('--output-dir', type=Path, default=output_dir,
                      help='Directory for output files')
    
    args = parser.parse_args()
    
    # Configure logging based on debug mode
    log_level = logging.DEBUG if args.debug != 'none' else logging.INFO
    logging.basicConfig(level=log_level,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        if args.debug != 'none':
            # Run in debug mode
            asyncio.run(debug_pipeline(args.response_dir, args.cde_dir, args.output_dir, args.debug))
        else:
            # Run normal processing
            processor = ResponseProcessor(args.response_dir, args.cde_dir, args.output_dir)
            asyncio.run(processor.process_responses())
            
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()