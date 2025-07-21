import pandas as pd
import re
from typing import Dict, Set, Tuple, Optional, Any
import json
import os
from pathlib import Path
import requests
from openai import AzureOpenAI
import hashlib
import time
from collections import defaultdict
from dotenv import load_dotenv

class TRAzureDataAnonymizer:
    def __init__(self, workspace_id: str = None, model_name: str = None, asset_id: str = None):
        # Load environment variables
        load_dotenv()
        
        # Thomson Reuters Azure OpenAI configuration
        self.workspace_id = workspace_id or os.getenv("WORKSPACE_ID", "AnonymizerW5XR")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")
        self.asset_id = asset_id or os.getenv("ASSET_ID", "208321")
        
        # Initialize Azure OpenAI client
        self.client = None
        self.setup_azure_client()
        
        # Caching for consistency and efficiency
        self.llm_cache = {}
        self.mappings = defaultdict(dict)
        self.failed_requests = set()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = float(os.getenv("MIN_REQUEST_INTERVAL", "0.2"))
        
        # LLM settings
        self.max_tokens = int(os.getenv("MAX_TOKENS", "100"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        
        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        
    def setup_azure_client(self):
        """Setup Azure OpenAI client with Thomson Reuters credentials"""
        try:
            payload = {
                "workspace_id": self.workspace_id,
                "model_name": self.model_name
            }
            
            url = "https://aiplatform.gcs.int.thomsonreuters.com/v1/openai/token"
            OPENAI_BASE_URL = "https://eais2-use.int.thomsonreuters.com"
            
            print("Getting Azure OpenAI credentials...")
            resp = requests.post(url, json=payload)
            
            if resp.status_code != 200:
                print(f"Failed to get credentials. Status code: {resp.status_code}")
                return False
                
            credentials = json.loads(resp.content)
            
            if "openai_key" in credentials and "openai_endpoint" in credentials:
                OPENAI_API_KEY = credentials["openai_key"]
                OPENAI_DEPLOYMENT_ID = credentials["azure_deployment"]
                OPENAI_API_VERSION = credentials["openai_api_version"]
                token = credentials["token"]
                llm_profile_key = OPENAI_DEPLOYMENT_ID.split("/")[0]
                
                headers = {
                    "Authorization": f"Bearer {token}",
                    "api-key": OPENAI_API_KEY,
                    "Content-Type": "application/json",
                    "x-tr-chat-profile-name": "ai-platforms-chatprofile-prod",
                    "x-tr-userid": self.workspace_id,
                    "x-tr-llm-profile-key": llm_profile_key,
                    "x-tr-user-sensitivity": "true",
                    "x-tr-sessionid": OPENAI_DEPLOYMENT_ID,
                    "x-tr-asset-id": self.asset_id,
                    "x-tr-authorization": OPENAI_BASE_URL
                }
                
                # Initialize the AzureOpenAI client
                self.client = AzureOpenAI(
                    azure_endpoint=OPENAI_BASE_URL,
                    api_key=OPENAI_API_KEY,
                    api_version=OPENAI_API_VERSION,
                    azure_deployment=OPENAI_DEPLOYMENT_ID,
                    default_headers=headers
                )
                
                print("✅ Azure OpenAI client initialized successfully!")
                return True
            else:
                print("❌ Failed to retrieve OpenAI credentials. Please check your inputs.")
                return False
                
        except Exception as e:
            print(f"❌ Error setting up Azure OpenAI client: {e}")
            return False
    
    def create_anonymization_prompt(self, value: str, column_name: str, 
                                  data_type: str, context_samples: list = None) -> str:
        """Create a detailed prompt for LLM anonymization"""
        
        context_info = f"Context samples: {context_samples[:3]}" if context_samples else ""
        
        prompt = f"""
You are a professional data anonymization expert. Your task is to replace the given value with a semantically similar but completely anonymized alternative.

CRITICAL REQUIREMENTS:
1. The replacement must be semantically similar (same category/type as original)
2. The replacement must be completely fictional - no real names, companies, locations, or contact information
3. Maintain the same format, structure, and approximate length
4. Be consistent - same input should always produce same output
5. The replacement should be professional and business-appropriate
6. Avoid any identifiable information or real-world references
7. Change any numeric identifiers to new, unique values

INPUT DETAILS:
- Original value: "{value}"
- Column: "{column_name}"
- Data type: {data_type}
- {context_info}

ANONYMIZATION EXAMPLES:
- "John Smith" → "Alex Johnson"
- "Thomson Reuters" → "Legal Solutions Corp"
- "Toronto" → "Vancouver"
- "john.smith@tr.com" → "alex.johnson@company.com"
- "416-555-1234" → "604-555-5678"
- "Director of Legal Affairs" → "Head of Legal Operations"
- "PCO WL Digital Platforms" → "Digital Services Platform"
- "Westlaw" → "LegalPro"

IMPORTANT: Provide ONLY the anonymized replacement value. No quotes, explanations, or additional text.
"""
        
        return prompt
    
    def detect_data_type(self, value: str) -> str:
        """Detect the type of data to help with anonymization"""
        if pd.isna(value):
            return "empty"
        
        value_str = str(value).strip()
        
        # Email detection
        if re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', value_str):
            return "email"
        
        # Phone number detection
        if re.match(r'[\+]?[\d\s\-\(\)\.]{10,}', value_str):
            return "phone"
        
        # Name detection (First Last or First Middle Last)
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', value_str):
            return "person_name"
        
        # Company/Organization name
        if any(keyword in value_str.lower() for keyword in ['corp', 'inc', 'ltd', 'company', 'llc', 'corporation', 'group']):
            return "company_name"
        
        # Job title detection
        if any(keyword in value_str.lower() for keyword in ['director', 'manager', 'officer', 'head', 'lead', 'coordinator', 'specialist']):
            return "job_title"
        
        # Address detection
        if any(keyword in value_str.lower() for keyword in ['street', 'avenue', 'road', 'drive', 'suite', 'floor']):
            return "address"
        
        # City/Location detection
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', value_str) and len(value_str) < 30:
            return "location"
        
        # Check for common patterns
        if re.match(r'^\d+$', value_str):
            return "numeric_id"
        elif re.match(r'^[A-Z0-9_]+$', value_str):
            return "code_identifier"
        elif len(value_str) > 30:
            return "description"
        elif any(keyword in value_str.lower() for keyword in ['platform', 'system', 'software']):
            return "product_name"
        elif any(keyword in value_str.lower() for keyword in ['department', 'division', 'center', 'professional']):
            return "department"
        elif any(keyword in value_str.lower() for keyword in ['operations', 'management', 'core']):
            return "business_unit"
        else:
            return "general_text"
        
    def is_too_similar(self, original: str, anonymized: str, threshold: float = 0.8) -> bool:
        """
        Returns True if 'anonymized' is too similar to 'original'.
        Uses a simple similarity ratio and substring check.
        """
        if not original or not anonymized:
            return False
        original = str(original).lower()
        anonymized = str(anonymized).lower()
        if original == anonymized:
            return True
        if len(original) > 4 and original in anonymized:
            return True
        # Use SequenceMatcher for similarity ratio
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original, anonymized).ratio()
        return similarity > threshold
    
    def should_anonymize_column(self, column_name: str, sample_values: list) -> bool:
        """Determine if a column should be anonymized based on its name and content"""
        column_lower = column_name.lower()
        
        # Always anonymize these types of columns
        sensitive_keywords = [
            'name', 'email', 'phone', 'contact', 'address', 'title', 'position',
            'director', 'manager', 'officer', 'lead', 'coordinator', 'specialist',
            'city', 'location', 'region', 'country', 'entity', 'organization',
            'company', 'corp', 'business', 'unit', 'division', 'department'
        ]
        
        # Check column name
        if any(keyword in column_lower for keyword in sensitive_keywords):
            return True
        
        # Check sample values for sensitive content
        for value in sample_values[:5]:  # Check first 5 values
            if pd.notna(value):
                data_type = self.detect_data_type(str(value))
                if data_type in ['email', 'phone', 'person_name', 'company_name', 'job_title', 'address', 'location']:
                    return True
        
        return False
    
    def auto_detect_columns_to_anonymize(self, df: pd.DataFrame) -> list:
        """Automatically detect which columns need anonymization with improved logic"""
        columns_to_anonymize = []
        
        print("🔍 Analyzing columns for sensitive content...")
        
        for col in df.columns:
            # Skip unnamed columns and purely numeric columns
            if col.lower().startswith('unnamed') or col.lower() in ['id', 'index']:
                continue
            
            # Get sample values for analysis
            sample_values = df[col].dropna().head(10).tolist()
            
            if self.should_anonymize_column(col, sample_values):
                columns_to_anonymize.append(col)
                print(f"  ✅ {col}: Contains sensitive data")
            else:
                print(f"  ⏭️ {col}: Skipping (no sensitive content detected)")
        
        return columns_to_anonymize
    
    def interactive_column_selection(self, df: pd.DataFrame) -> list:
        """Allow user to interactively select columns to anonymize"""
        print("\n📋 Available columns:")
        for i, col in enumerate(df.columns, 1):
            sample_values = df[col].dropna().head(3).tolist()
            print(f"{i:2d}. {col:<30} (samples: {sample_values})")
        
        print("\nOptions:")
        print("  - Enter column numbers (comma-separated): e.g., 1,3,5")
        print("  - Enter 'all' to anonymize all columns")
        print("  - Enter 'auto' for automatic detection")
        print("  - Enter 'none' to skip anonymization")
        
        while True:
            selection = input("\nYour choice: ").strip().lower()
            
            if selection == 'none':
                return []
            elif selection == 'all':
                return df.columns.tolist()
            elif selection == 'auto':
                return self.auto_detect_columns_to_anonymize(df)
            else:
                try:
                    # Parse column numbers
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected_columns = [df.columns[i] for i in indices if 0 <= i < len(df.columns)]
                    return selected_columns
                except (ValueError, IndexError):
                    print("❌ Invalid input. Please try again.")
    
    def get_llm_anonymization(self, value: str, column_name: str, 
                            context_samples: list = None) -> str:
        """Get LLM anonymization using Azure OpenAI"""
        
        if not self.client:
            return self.fallback_anonymization(value, column_name)
        
        # Create cache key for consistency
        cache_key = hashlib.md5(f"{value}_{column_name}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.llm_cache:
            self.cache_hits += 1
            return self.llm_cache[cache_key]
        
        # Check if this request has failed before
        if cache_key in self.failed_requests:
            return self.fallback_anonymization(value, column_name)
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - self.last_request_time))
        
        try:
            data_type = self.detect_data_type(value)
            prompt = self.create_anonymization_prompt(value, column_name, data_type, context_samples)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional data anonymization assistant. Provide only the anonymized replacement value."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up the result (remove quotes if present)
            result = result.strip('"\'')
            
            # Validate result is not empty or too similar
            if (not result or result.lower() in ['null', 'none', 'n/a'] or 
                self.is_too_similar(value, result)):
                print(f"⚠️ Anonymized value too similar to original ('{value}' → '{result}'), using fallback.")
                result = self.fallback_anonymization(value, column_name)
            
            self.last_request_time = time.time()
            self.total_requests += 1
            
            # Cache the result
            self.llm_cache[cache_key] = result
            
            print(f"LLM: '{value}' → '{result}'")
            return result
            
        except Exception as e:
            print(f"⚠️ LLM request failed for '{value}': {e}")
            self.failed_requests.add(cache_key)
            return self.fallback_anonymization(value, column_name)
    
    def fallback_anonymization(self, value: str, column_name: str) -> str:
        """Fallback anonymization when LLM fails"""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        # Use existing mapping if available
        if value_str in self.mappings[column_name]:
            return self.mappings[column_name][value_str]
        
        # Generate fallback based on patterns
        data_type = self.detect_data_type(value_str)
        counter = len(self.mappings[column_name]) + 1
        
        if data_type == "email":
            result = f"user{counter:03d}@company.com"
        elif data_type == "phone":
            result = f"555-{counter:04d}"
        elif data_type == "person_name":
            result = f"Person {counter:03d}"
        elif data_type == "company_name":
            result = f"Company {counter:03d}"
        elif data_type == "job_title":
            result = f"Position {counter:03d}"
        elif data_type == "address":
            result = f"Address {counter:03d}"
        elif data_type == "location":
            result = f"Location {counter:03d}"
        elif data_type == "numeric_id":
            result = f"{counter:0{len(value_str)}d}"
        elif data_type == "code_identifier":
            result = f"CODE_{counter:04d}"
        elif data_type == "product_name":
            result = f"Product_{counter:03d}"
        elif data_type == "department":
            result = f"Department_{counter:03d}"
        elif data_type == "business_unit":
            result = f"BusinessUnit_{counter:03d}"
        else:
            result = f"Value_{counter:04d}"
        
        self.mappings[column_name][value_str] = result
        print(f"Fallback: '{value}' → '{result}'")
        return result
    
    def analyze_column_context(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """Analyze column to provide context for LLM"""
        column_data = df[column_name].dropna()
        
        return {
            'column_name': column_name,
            'sample_values': column_data.head(5).tolist(),
            'unique_count': column_data.nunique(),
            'total_count': len(column_data),
            'data_types': [self.detect_data_type(str(val)) for val in column_data.head(3)]
        }
    
    def anonymize_value_with_llm(self, value: Any, column_name: str, context_samples: list = None) -> Any:
        """Anonymize a single value using LLM"""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        # Skip if already anonymized
        if value_str in self.mappings[column_name]:
            return self.mappings[column_name][value_str]
        
        # Get LLM anonymization
        anonymized = self.get_llm_anonymization(value_str, column_name, context_samples)
        
        # Store mapping for consistency
        self.mappings[column_name][value_str] = anonymized
        
        return anonymized
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                          columns_to_anonymize: Optional[list] = None,
                          interactive: bool = True) -> pd.DataFrame:
        """Anonymize entire dataframe using Azure OpenAI"""
        
        if columns_to_anonymize is None:
            if interactive:
                columns_to_anonymize = self.interactive_column_selection(df)
            else:
                columns_to_anonymize = self.auto_detect_columns_to_anonymize(df)
        
        # Filter out columns that don't exist in the dataframe
        columns_to_anonymize = [col for col in columns_to_anonymize if col in df.columns]
        
        if not columns_to_anonymize:
            print("❌ No columns selected for anonymization.")
            return df
        
        print(f"🔍 Analyzing {len(columns_to_anonymize)} columns for LLM anonymization...")
        
        # Analyze each column for context
        column_contexts = {}
        total_unique_values = 0
        
        for col in columns_to_anonymize:
            column_contexts[col] = self.analyze_column_context(df, col)
            unique_count = column_contexts[col]['unique_count']
            total_unique_values += unique_count
            print(f"  📊 {col}: {unique_count} unique values")
        
        if interactive:
            print("\n" + "="*60)
            print("🚀 AZURE OPENAI ANONYMIZATION PLAN")
            print("="*60)
            print(f"🏢 Workspace ID: {self.workspace_id}")
            print(f"🤖 Model: {self.model_name}")
            print(f"📁 Asset ID: {self.asset_id}")
            print(f"📋 Columns to anonymize: {len(columns_to_anonymize)}")
            print(f"🔢 Total unique values to process: {total_unique_values}")
            
            print(f"\n📋 Columns to process:")
            for col in columns_to_anonymize:
                if col in column_contexts:
                    print(f"  • {col}: {column_contexts[col]['unique_count']} unique values")
            
            estimated_time = total_unique_values * self.min_request_interval / 60
            print(f"⏱️ Estimated processing time: ~{estimated_time:.1f} minutes")
            
            response = input(f"\n❓ Proceed with Azure OpenAI anonymization? (y/n): ").lower()
            if response != 'y':
                print("❌ Anonymization cancelled.")
                return df
        
        print(f"\n🔄 Processing {len(df)} rows with Azure OpenAI anonymization...")
        
        # Create anonymized dataframe
        anonymized_df = df.copy()
        
        for col in columns_to_anonymize:
            print(f"\n📝 Processing column: {col}")
            context_info = column_contexts.get(col, {})
            context_samples = context_info.get('sample_values', [])
            
            # Process each unique value in the column
            unique_values = df[col].dropna().unique()
            
            for i, value in enumerate(unique_values):
                if i % 5 == 0:  # Progress indicator
                    print(f"  📈 Progress: {i}/{len(unique_values)} ({(i/len(unique_values)*100):.1f}%)")
                
                anonymized_value = self.anonymize_value_with_llm(value, col, context_samples)
                
                # Replace all occurrences of this value in the column
                anonymized_df.loc[df[col] == value, col] = anonymized_value
        
        print(f"\n✅ Azure OpenAI anonymization complete!")
        print(f"📊 Statistics:")
        print(f"  • Total LLM requests made: {self.total_requests}")
        print(f"  • Cache hits: {self.cache_hits}")
        print(f"  • Failed requests: {len(self.failed_requests)}")
        print(f"  • Success rate: {((self.total_requests - len(self.failed_requests)) / max(self.total_requests, 1) * 100):.1f}%")
        
        return anonymized_df
    
    def process_csv_file(self, input_file: str, output_file: str = None, 
                        columns_to_anonymize: Optional[list] = None,
                        interactive: bool = True) -> str:
        """Process a CSV file with Azure OpenAI anonymization"""
        print(f"📂 Processing file: {input_file}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(input_file)
            print(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
        
        # Anonymize the data
        anonymized_df = self.anonymize_dataframe(df, columns_to_anonymize, interactive)
        
        # Generate output filename if not provided
        if output_file is None:
            path = Path(input_file)
            output_file = path.parent / f"{path.stem}_azure_anonymized{path.suffix}"
        
        # Save anonymized data
        anonymized_df.to_csv(output_file, index=False)
        print(f"💾 Azure OpenAI anonymized data saved to: {output_file}")
        
        # Save mappings and cache
        mappings_file = str(Path(output_file).parent / f"{Path(output_file).stem}_azure_mappings.json")
        self.save_mappings_and_cache(mappings_file)
        
        return str(output_file)
    
    def save_mappings_and_cache(self, mappings_file: str):
        """Save mappings and LLM cache"""
        data = {
            'mappings': dict(self.mappings),
            'llm_cache': self.llm_cache,
            'failed_requests': list(self.failed_requests),
            'workspace_id': self.workspace_id,
            'model_name': self.model_name,
            'asset_id': self.asset_id,
            'statistics': {
                'total_requests': self.total_requests,
                'cache_hits': self.cache_hits,
                'failed_requests': len(self.failed_requests)
            }
        }
        
        with open(mappings_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Azure OpenAI mappings and cache saved to: {mappings_file}")
    
    def load_mappings_and_cache(self, mappings_file: str):
        """Load mappings and LLM cache"""
        try:
            with open(mappings_file, 'r') as f:
                data = json.load(f)
            
            self.mappings = defaultdict(dict, data.get('mappings', {}))
            self.llm_cache = data.get('llm_cache', {})
            self.failed_requests = set(data.get('failed_requests', []))
            
            # Load statistics
            stats = data.get('statistics', {})
            self.total_requests = stats.get('total_requests', 0)
            self.cache_hits = stats.get('cache_hits', 0)
            
            print(f"📂 Azure OpenAI mappings and cache loaded from: {mappings_file}")
            print(f"📊 Loaded {len(self.llm_cache)} cached responses")
        except FileNotFoundError:
            print("📂 No existing Azure OpenAI mappings found, starting fresh")

def main():
    """Main function to run the Azure OpenAI anonymizer"""
    
    # Initialize the Azure OpenAI anonymizer (credentials from .env file)
    anonymizer = TRAzureDataAnonymizer()
    
    # Load existing cache if available
    anonymizer.load_mappings_and_cache("azure_anonymizer_mappings.json")
    
    # Process your sample file
    input_file = "Overall Key Legal Entity Contacts -  FY 2025_Canada.csv"
    
    # Let the system auto-detect or allow manual selection
    columns_to_anonymize = None  # This will trigger interactive selection
    
    try:
        result = anonymizer.process_csv_file(
            input_file, 
            columns_to_anonymize=columns_to_anonymize,
            interactive=True
        )
        
        print(f"\n🎉 Processing complete! Output saved to: {result}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    main()