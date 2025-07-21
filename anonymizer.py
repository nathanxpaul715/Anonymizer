import pandas as pd
import re
from typing import Dict, Set, Tuple, Optional, Any, List
import json
import os
from pathlib import Path
import requests
from openai import AzureOpenAI
import hashlib
import time
from collections import defaultdict
from dotenv import load_dotenv
import random

class TRAzureDataAnonymizer:
    def __init__(self, workspace_id: str = None, model_name: str = None, asset_id: str = None, 
                 entity_identifying_columns: Optional[List[str]] = None): # NEW PARAM
        # Load environment variables
        load_dotenv()
        
        # Thomson Reuters Azure OpenAI configuration
        self.workspace_id = workspace_id or os.getenv("WORKSPACE_ID", "AnonymizerW5XR")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")
        self.asset_id = asset_id or os.getenv("ASSET_ID", "208321")
        self.entity_identifying_columns = entity_identifying_columns # NEW
        
        # Initialize Azure OpenAI client
        self.client = None
        self.setup_azure_client()
        
        # Caching for consistency and efficiency
        self.llm_cache = {}
        self.mappings = defaultdict(dict) # Per-value mapping (for fallback or non-entity columns)
        self.entity_profile_mappings = {} # NEW: Entity-level mapping
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
                                  data_type: str, context_samples: list = None,
                                  llm_specific_context: dict = None) -> str: # NEW PARAM
        """Create a detailed prompt for LLM anonymization based on data type and column context."""
        
        context_info_str = ""
        if context_samples:
            context_info_str = f"- Other representative values from this column: {context_samples}"
        
        # Determine specific instructions based on detected data_type and column_name
        specific_instruction = "" # Initialize specific_instruction
        
        if data_type == "email":
            # Check for specific context like an anonymized name for consistency
            if llm_specific_context and 'anonymized_person_name' in llm_specific_context:
                anonymized_name_for_email = llm_specific_context['anonymized_person_name']
                # Attempt to derive plausible first/last from anonymized name
                parts = anonymized_name_for_email.split()
                anonymized_first = parts[0].lower() if parts else "anon"
                anonymized_last = parts[-1].lower() if len(parts) > 1 else "user"
                
                specific_instruction = (
                    f"Generate a random but plausible email address. "
                    f"The anonymized name for this entity is '{anonymized_name_for_email}'. "
                    f"Preferably, integrate the anonymized name (e.g., '{anonymized_first}.{anonymized_last}@fictiousdomain.net') "
                    f"into the email to maintain consistency. "
                    "Maintain the typical email format (e.g., firstname.lastname@randomdomain.com). "
                    "Ensure it is distinctly different from the original and sounds entirely fictional."
                )
            else:
                specific_instruction = (
                    "Generate a random but plausible email address. "
                    "Maintain the typical email format (e.g., firstname.lastname@randomdomain.com). "
                    "Ensure it is distinctly different from the original and sounds entirely fictional. "
                    "For example, if the original was 'john.doe@example.com', a good anonymized value would be 'alex.fictional@fictiousdomain.net'."
                )
        elif data_type == "phone":
            specific_instruction = (
                "Generate a random but plausible phone number of a similar length and common format (e.g., international, national). "
                "It must not be a real phone number. Example: if original is '+1-555-123-4567', anonymized could be '+1-999-876-5432'."
            )
        elif data_type == "person_name":
            specific_instruction = (
                "Generate a plausible, but completely fictional, human name (e.g., a first and last name, or full name). "
                "Ensure it sounds natural for a person but is not a real person's name. Example: 'Jane Smith' becomes 'Elara Vance'."
            )
        elif data_type == "company_name":
            specific_instruction = (
                "Generate a plausible, but completely fictional, company or organization name. "
                "Maintain semantic plausibility and common company suffixes (e.g., 'Solutions', 'Group', 'Inc.', 'Ltd.', 'GmbH'). "
                "Example: 'Acme Corp' becomes 'Luminary Innovations Inc.'."
            )
        elif data_type == "job_title":
            specific_instruction = (
                "Generate a plausible, but completely fictional, job title. "
                "It should reflect a similar professional level or domain as the original, "
                "but not be easily traceable to specific roles. Example: 'Senior Marketing Manager' becomes 'Lead Brand Strategist'."
            )

        elif data_type == "numeric_text_combo":
            specific_instruction = (
                "The original value consists of a numerical prefix followed by text (e.g., '12345 Westland'). "
                "Your task is to generate a new anonymized value that maintains this structure: a numerical part "
                "followed by a textual part. "
                "1. Generate a random number of the *same length* as the original numerical prefix. "
                "2. Generate a plausible, fictional, and contextually appropriate text for the suffix. "
                "3. Combine the new number and new text with a single space in between, mirroring the original format. "
                "Example: if original is '12345 Westland', anonymized could be '98765 Northfield'."
            )

        elif data_type == "address":
            specific_instruction = (
                "Generate a plausible, but completely fictional, street address (e.g., street name, number, building details). "
                "It should resemble a real address but refer to a non-existent location. Example: '123 Main St, Anytown' becomes '789 Elm Blvd, Fictionalburg'."
            )
        elif data_type == "location": # This can cover City, Country, State/Province etc.
            specific_instruction = (
                "Generate a plausible, but completely fictional, geographical location name (e.g., city, country, region, state). "
                "It should maintain the general type and feel of the original, without being real. "
                "Example: 'Canada' becomes 'Celestia'; 'London' becomes 'Aethelburg'."
            )
        elif data_type == "numeric_id":
            specific_instruction = (
                "Generate a random numeric identifier of the same length and general format as the original. "
                "The number should be unique and not easily guessable or reversible. Example: '123456789' becomes '987654321'."
            )
        elif data_type == "code_identifier":
            specific_instruction = (
                "Generate a random alphanumeric code or identifier of the same length and format. "
                "It should be unique and not easily guessable. Example: 'USR_ABC_001' becomes 'XYZ_PQR_789'."
            )
        elif data_type == "description": # For long text that might contain PII
            specific_instruction = (
                "Analyze the provided text. Identify and replace any specific personal or sensitive information (e.g., names, specific dates, unique numbers, precise locations) "
                "with generic placeholders or plausible fictional equivalents. "
                "Maintain the overall length, tone, and non-sensitive descriptive content of the original. "
                "If no sensitive info is found, provide a slightly rephrased version of the original to obscure it subtly."
            )
        elif data_type == "product_name":
            specific_instruction = (
                "Generate a plausible, fictional product or service name. "
                "It should retain a similar industry feel or type if applicable. Example: 'ExcelSuite Pro' becomes 'DataGenius Hub'."
            )
        elif data_type == "department" or data_type == "business_unit":
            specific_instruction = (
                "Generate a plausible, fictional department or business unit name. "
                "It should sound similar in nature to the original but be entirely new. Example: 'Sales & Marketing' becomes 'Client Engagement Division'."
            )
        else: # Covers 'general_text', 'empty', or other uncategorized types.
            specific_instruction = (
                "Generate a plausible, random replacement value for this general text. "
                "It should be distinctly different from the original but maintain similar characteristics (e.g., length, word count, general topic if discernable). "
                "If it's a number that wasn't caught as an ID, generate a random number of similar magnitude."
            )

        # General instructions for the LLM
        prompt_template = """
        You are a highly specialized data anonymization assistant. Your task is to provide a single, anonymized replacement value for the given input.

        GENERAL GUIDELINES FOR ANONYMIZATION:
        - The anonymized value must be plausible and maintain the same format, structure, and general data type as the original.
        - The anonymized value must contain NO original personal or sensitive information, directly or indirectly.
        - The anonymized value should be distinct and non-reversible from the original, preventing easy re-identification.
        - All generated names, places, and identifiers should be entirely fictional.

        INPUT DETAILS FOR ANONYMIZATION:
        - Original Value to Anonymize: "{value}"
        - Column Name this value came from: "{column_name}"
        - Automatically Detected Data Type for this value: {data_type}
        {context_info_str}

        SPECIFIC INSTRUCTION FOR ANONYMIZING THIS VALUE:
        {specific_instruction}

        IMPORTANT:
        Provide ONLY the anonymized replacement value. Do NOT include any explanations, quotes around the value, or additional text.
        Your response should be *only* the anonymized data point.
        """
        
        return prompt_template.format(
            value=value,
            column_name=column_name,
            data_type=data_type,
            context_info_str=context_info_str,
            specific_instruction=specific_instruction
        )
    
    def detect_data_type(self, value: str) -> str:
        """Detect the type of data to help with anonymization (improved version)"""
        if pd.isna(value):
            return "empty"
        value_str = str(value).strip()
        lower_val = value_str.lower()

        # Email detection
        if re.match(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b", value_str):
            return "email"

        # Phone (US, Canada, Europe, with or without country code)
        phone_patterns = [
            r"^\+?(\d[\d\-\.\s\(\)]{7,}\d)$",                 # +1 555-555-5555, +44 20 7123 1234
            r"^\(?\d{3,4}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4}$" # (555) 555-5555, 020 7123 1234
        ]
        if any(re.match(p, value_str) for p in phone_patterns):
            return "phone"

        # International names: support accented characters, common non-English names
        # Check for at least two words, first letters capitalized
        if re.match(r"^[A-ZÀ-ÿ][a-zà-ÿ'-]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ'-]+)+$", value_str):
            return "person_name"

        # Company/Organization names (expanded keywords)
        company_keywords = [
            'corp', 'inc', 'ltd', 'company', 'llc', 'corporation', 'group',
            'gmbh', 'sarl', 'sa', 'plc', 'pte', 'ag', 'spa', 'oy', 'ab'
        ]
        if any(kw in lower_val for kw in company_keywords) or \
           (' inc' in lower_val or ' llc' in lower_val or ' ltd' in lower_val): # Added common standalone suffixes
            return "company_name"

        # Job title detection (expanded)
        job_keywords = [
            'director', 'manager', 'officer', 'head', 'lead', 'coordinator', 'specialist',
            'president', 'chief', 'analyst', 'consultant', 'engineer', 'developer', 'vice', 'vp',
            'executive', 'assistant', 'admin', 'sales', 'marketing', 'finance', 'hr', 'support',
            'cfo', 'ceo', 'cto', 'coo', 'ciso'
        ]
        if any(kw in lower_val.split() for kw in job_keywords): # split() for whole words
            return "job_title"
        
        # Captures one or more digits at the beginning, followed by a space, and then any text.
        match_numeric_text = re.match(r"^(\d+)\s+(.+)$", value_str)
        if match_numeric_text:
            num_part = match_numeric_text.group(1)
            text_part = match_numeric_text.group(2)
            # Ensure the text part is not purely numeric itself to prevent misclassification
            if not re.match(r"^\d+$", text_part):
                return "numeric_text_combo"

        # Address detection (expanded)
        address_keywords = [
            'street', 'st.', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr', 'suite', 'floor',
            'blvd', 'lane', 'ln', 'way', 'building', 'block', 'apartment', 'apt', 'unit',
            'plaza', 'circle', 'court', 'place', 'highway', 'hwy', 'parkway', 'pkwy'
        ]
        if any(kw in lower_val for kw in address_keywords) and \
           re.search(r'\d+\s+', value_str): # require a number followed by space
            return "address"

        # City/Location detection (allow some special characters, support longer names)
        # Check if it looks like a place name (e.g., capitalised, not too long, no numbers/special chars)
        if re.match(r"^[A-ZÀ-ÿ][a-zà-ÿ'\-\s.]{2,70}$", value_str) and \
           not any(char.isdigit() for char in value_str) and \
           ' ' in value_str: # More likely to be a city if multiple words
            return "location"

        # Numeric ID - Must be purely numeric and often has a fixed length or structure
        if re.match(r"^\d{4,20}$", value_str) and len(value_str) > 5: # Assume IDs are usually longer
            return "numeric_id"
        # Alphanumeric ID / Code Identifier - often has underscores, hyphens, fixed patterns
        elif re.match(r"^[A-Z0-9_-]{5,30}$", value_str):
            return "code_identifier"
        
        # Longer texts often imply descriptions
        elif len(value_str) > 50: # Arbitrary threshold for longer text
            return "description"
        
        # Product name detection (expanded)
        elif any(kw in lower_val for kw in ['platform', 'system', 'software', 'solution', 'service', 'suite', 'pro']):
            return "product_name"
        
        # Department/Business Unit detection
        elif any(kw in lower_val for kw in ['department', 'division', 'center', 'professional', 
                                              'operations', 'management', 'sales', 'marketing',
                                              'finance', 'human resources', 'legal', 'it', 'hr']):
            return "department" # Generalizing to 'department'

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
        if len(original) > 4 and original in anonymized: # e.g., "John Doe" -> "John Doe Anonymized"
            return True
        # Use SequenceMatcher for similarity ratio
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original, anonymized).ratio()
        return similarity > threshold
    
    def should_anonymize_column(self, column_name: str, sample_values: list) -> bool:
        """Determine if a column should be anonymized based on its name and content"""
        column_lower = column_name.lower()
        
        # Always anonymize these types of columns (expanded keywords)
        sensitive_keywords = [
            'name', 'email', 'phone', 'contact', 'address', 'title', 'position', 'role',
            'city', 'location', 'region', 'country', 'state', 'province', 'zip', 'postcode',
            'entity', 'organization', 'company', 'corp', 'business', 'unit', 'division', 'department',
            'id', 'account', 'number', 'ssn', 'sin', 'tax' # common identifier keywords
        ]
        
        # Check column name
        if any(keyword in column_lower for keyword in sensitive_keywords):
            return True
        
        # Check sample values for sensitive content
        for value in sample_values[:5]:  # Check first 5 values for efficiency
            if pd.notna(value):
                data_type = self.detect_data_type(str(value))
                if data_type in ['email', 'phone', 'person_name', 'company_name', 'job_title', 'address', 'location',
                                 'numeric_id', 'code_identifier', 'description']: # Consider description if long text
                    return True
        
        return False
    
    def auto_detect_columns_to_anonymize(self, df: pd.DataFrame) -> list:
        """Automatically detect which columns need anonymization with improved logic"""
        columns_to_anonymize = []
        
        print("🔍 Analyzing columns for sensitive content...")
        
        for col in df.columns:
            # Skip unnamed columns and purely numeric columns (unless they contain IDs)
            if col.lower().startswith('unnamed') or col.lower() in ['id', 'index'] and df[col].dtype.kind in 'iuf': # skip generic IDs if numeric
                continue
            
            # Get sample values for analysis
            sample_values = df[col].dropna().head(10).tolist()
            
            if self.should_anonymize_column(col, sample_values):
                columns_to_anonymize.append(col)
                print(f"  ✅ {col}: Contains sensitive data")
            else:
                print(f"  ⏭️ {col}: Skipping (no sensitive content detected or already a simple ID column)")
        
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
                            context_samples: list = None,
                            llm_specific_context: dict = None) -> str: # NEW PARAM
        """Get LLM anonymization using Azure OpenAI"""
        
        if not self.client:
            return self.fallback_anonymization(value, column_name)
        
        # Create cache key for consistency (include specific context if relevant)
        cache_key_components = [value, column_name]
        if llm_specific_context:
            for k in sorted(llm_specific_context.keys()): # Sort keys for consistent hashing
                cache_key_components.append(f"{k}:{llm_specific_context[k]}")
        cache_key = hashlib.md5("_".join(cache_key_components).encode()).hexdigest()
        
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
            prompt = self.create_anonymization_prompt(value, column_name, data_type, context_samples, llm_specific_context) # Pass llm_specific_context
            
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
            
            # Update the per-value mappings for consistency across contexts
            self.mappings[column_name][value] = result
            
            print(f"LLM: '{value}' → '{result}'")
            return result
            
        except Exception as e:
            print(f"⚠️ LLM request failed for '{value}': {e}")
            self.failed_requests.add(cache_key)
            return self.fallback_anonymization(value, column_name)
    
    def fallback_anonymization(self, value: str, column_name: str) -> str:
        """Fallback anonymization when LLM fails or is too similar"""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        # Use existing mapping if available
        if value_str in self.mappings[column_name]:
            return self.mappings[column_name][value_str]
        
        # Generate fallback based on patterns
        data_type = self.detect_data_type(value_str)
        counter = len(self.mappings[column_name]) + 1
        
        # Make fallback slightly more generic to prevent collision if LLM output varies
        # but also to be less recognizable
        if data_type == "email":
            result = f"anon.user.{counter:03d}@example.com"
        elif data_type == "phone":
            result = f"+1-555-0{counter:06d}"
        elif data_type == "person_name":
            result = f"Fictional Person {counter:03d}"
        elif data_type == "company_name":
            result = f"Anonymized Corp {counter:03d}"
        elif data_type == "job_title":
            result = f"Generic Role {counter:03d}"
        # NEW: Fallback for Numeric-Text Combination
        elif data_type == "numeric_text_combo":
            match = re.match(r"^(\d+)\s+(.+)$", value_str)
            if match:
                num_part_len = len(match.group(1))
                # Generate a random number of the same length
                new_num_digits = [str(random.randint(0, 9)) for _ in range(num_part_len)]
                # Ensure the first digit is not '0' unless original was '0' and only one digit
                if num_part_len > 1 and new_num_digits[0] == '0' and match.group(1)[0] != '0':
                    new_num_digits[0] = str(random.randint(1, 9))
                new_num = "".join(new_num_digits)

                # Generate a generic, plausible text part
                generic_text_options = ["Eastland", "Northfield", "Riverbend", "Greenwood", "Lakeview", "Southridge", "Westwood"]
                new_text = random.choice(generic_text_options)
                result = f"{new_num} {new_text}"
            else: # Fallback if regex somehow fails (shouldn't happen if detect_data_type returned this type)
                result = f"NUM_TEXT_ANON_{counter:04d}"
        elif data_type == "address":
            result = f"999 Anonymized St, Anytown {counter:03d}"
        elif data_type == "location":
            result = f"Fictionalia {counter:03d}"
        elif data_type == "numeric_id":
            # Preserve length if possible for numeric IDs
            len_val = len(value_str)
            result = str(10**(len_val -1) + counter).zfill(len_val) if len_val > 0 else str(counter)
        elif data_type == "code_identifier":
            result = f"CODE_ANON_{counter:04d}"
        elif data_type == "product_name":
            result = f"ProductX_{counter:03d}"
        elif data_type == "department":
            result = f"Dept_Anon_{counter:03d}"
        elif data_type == "business_unit":
            result = f"Unit_Anon_{counter:03d}"
        else:
            result = f"AnonymizedValue_{counter:04d}"
        
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
            'data_types': [self.detect_data_type(str(val)) for val in column_data.head(min(len(column_data), 3))] # Avoid error on small series
        }
    
    def anonymize_value_with_llm(self, value: Any, column_name: str, 
                                context_samples: list = None,
                                llm_specific_context: dict = None) -> Any: # NEW PARAM
        """Anonymize a single value using LLM. Used for fallback path or specific entity fields."""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        # Check per-value mapping cache first
        if value_str in self.mappings[column_name]:
            return self.mappings[column_name][value_str]
        
        # Get LLM anonymization
        anonymized = self.get_llm_anonymization(value_str, column_name, context_samples, llm_specific_context) # Pass llm_specific_context
        
        # Store mapping for consistency (get_llm_anonymization also stores it)
        # self.mappings[column_name][value_str] = anonymized # This is now done inside get_llm_anonymization
        
        return anonymized
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                          columns_to_anonymize: Optional[list] = None,
                          interactive: bool = True,
                          entity_identifying_columns: Optional[List[str]] = None) -> pd.DataFrame: # NEW PARAM
        """Anonymize entire dataframe using Azure OpenAI"""
        
        # Update instance's entity identifying columns if provided
        if entity_identifying_columns is not None:
            self.entity_identifying_columns = entity_identifying_columns

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
        
        anonymized_df = df.copy()

        # --- Entity-level Anonymization Logic (New) ---
        if self.entity_identifying_columns and all(col in df.columns for col in self.entity_identifying_columns):
            print(f"\n✨ Initiating entity-level anonymization using columns: {self.entity_identifying_columns}")
            
            # Create a temporary unique key for each entity based on identifying columns
            # Ensure NaN values are handled consistently (e.g., converted to a string 'NaN_VAL')
            df['__entity_key__'] = df.apply(
                lambda row: tuple(str(row[col]) if pd.isna(row[col]) else row[col] for col in self.entity_identifying_columns),
                axis=1
            )
            
            unique_entity_keys = df['__entity_key__'].unique()
            print(f"Detected {len(unique_entity_keys)} unique entities.")

            # --- Pre-generate Anonymized Entity Profiles ---
            print("🔄 Pre-generating anonymized profiles for unique entities...")
            for i, entity_key in enumerate(unique_entity_keys):
                if i % 50 == 0 and i > 0: # Progress indicator
                    print(f"  Processed {i}/{len(unique_entity_keys)} entities ({(i/len(unique_entity_keys)*100):.1f}%)")

                if entity_key not in self.entity_profile_mappings:
                    # Find one original row that matches this entity key to get its original PII values
                    original_row_for_entity = df[df['__entity_key__'] == entity_key].iloc[0]
                    
                    anonymized_profile_for_this_entity = {}
                    
                    # Define a processing order for sensitive columns to handle dependencies (e.g., Name before Email)
                    # This is a heuristic and can be improved based on specific data patterns.
                    # Prioritize person_name, then email, then others.
                    ordered_sensitive_cols_for_entity = []
                    # First, add person names
                    for col_name in columns_to_anonymize:
                        if self.detect_data_type(original_row_for_entity[col_name]) == 'person_name':
                            if col_name not in ordered_sensitive_cols_for_entity:
                                ordered_sensitive_cols_for_entity.append(col_name)
                    # Then, add emails (potentially using the anonymized name)
                    for col_name in columns_to_anonymize:
                        if self.detect_data_type(original_row_for_entity[col_name]) == 'email':
                             if col_name not in ordered_sensitive_cols_for_entity:
                                ordered_sensitive_cols_for_entity.append(col_name)
                    # Finally, add remaining sensitive columns
                    for col_name in columns_to_anonymize:
                        if col_name not in ordered_sensitive_cols_for_entity:
                            ordered_sensitive_cols_for_entity.append(col_name)
                    
                    for col_name in ordered_sensitive_cols_for_entity:
                        original_value = original_row_for_entity[col_name]
                        
                        # Prepare LLM specific context for cross-column consistency
                        llm_context = {}
                        if self.detect_data_type(original_value) == 'email':
                            # Pass anonymized name if available from the same profile being built
                            for p_col, p_val in anonymized_profile_for_this_entity.items():
                                if self.detect_data_type(p_val) == 'person_name':
                                    llm_context['anonymized_person_name'] = p_val
                                    break
                        
                        anonymized_value = self.anonymize_value_with_llm(
                            value=original_value,
                            column_name=col_name,
                            llm_specific_context=llm_context # Pass context
                        )
                        anonymized_profile_for_this_entity[col_name] = anonymized_value
                    
                    self.entity_profile_mappings[entity_key] = anonymized_profile_for_this_entity
            
            print(f"  Finished pre-generating {len(self.entity_profile_mappings)} entity profiles.")

            # --- Apply Anonymized Entity Profiles to DataFrame ---
            print("Applying anonymized profiles to the DataFrame...")
            # Use a dictionary mapping for efficiency instead of apply on columns one by one
            for col_name in columns_to_anonymize:
                # Only apply if the column was part of the entity profile (i.e., processed above)
                # Check if at least one profile contains this column to avoid KeyError
                if any(col_name in profile for profile in self.entity_profile_mappings.values()):
                    # Create a temporary series of anonymized values for the column
                    anonymized_series = df['__entity_key__'].map(
                        lambda k: self.entity_profile_mappings.get(k, {}).get(col_name, pd.NA)
                    )
                    # Fill original values where entity key not found or column not in profile (shouldn't happen if logic is tight)
                    anonymized_series = anonymized_series.fillna(df[col_name])
                    
                    anonymized_df[col_name] = anonymized_series.astype(df[col_name].dtype, errors='ignore') # Maintain dtype
                else:
                    # If column was marked for anonymization but not part of entity logic, fall back to per-value
                    print(f"  Column '{col_name}' was not processed as part of entity; falling back to per-value anonymization.")
                    unique_values = df[col_name].dropna().unique()
                    for i, value in enumerate(unique_values):
                         anonymized_value = self.anonymize_value_with_llm(value, col_name)
                         anonymized_df.loc[df[col_name] == value, col_name] = anonymized_value

            # Drop the temporary entity key column
            anonymized_df = anonymized_df.drop(columns=['__entity_key__'])
            print("✅ Entity-level anonymization complete.")

        # --- Fallback to per-value Anonymization Logic (Existing) ---
        else: 
            print("\n⚠️ Entity identifying columns not specified or invalid. Falling back to per-value anonymization.")
            print(f"\n🔄 Processing {len(df)} rows with Azure OpenAI anonymization...")
            
            # Analyze each column for context (needed for `get_llm_anonymization` if not entity-level)
            column_contexts = {}
            total_unique_values = 0
            
            for col in columns_to_anonymize:
                column_contexts[col] = self.analyze_column_context(df, col)
                unique_count = column_contexts[col]['unique_count']
                total_unique_values += unique_count
                print(f"  📊 {col}: {unique_count} unique values")
            
            if interactive:
                print("\n" + "="*60)
                print("🚀 AZURE OPENAI ANONYMIZATION PLAN (PER-VALUE MODE)")
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
            
            for col in columns_to_anonymize:
                print(f"\n📝 Processing column: {col}")
                context_info = column_contexts.get(col, {})
                context_samples = context_info.get('sample_values', [])
                
                # Process each unique value in the column
                unique_values = df[col].dropna().unique()
                
                for i, value in enumerate(unique_values):
                    if i % 50 == 0 and i > 0:  # Progress indicator
                        print(f"  📈 Progress: {i}/{len(unique_values)} ({(i/len(unique_values)*100):.1f}%)")
                    
                    anonymized_value = self.anonymize_value_with_llm(value, col, context_samples)
                    
                    # Replace all occurrences of this value in the column
                    # Use .loc with boolean indexing for robust replacement, handle potential type changes
                    anonymized_df.loc[df[col] == value, col] = anonymized_value
            
            print(f"\n✅ Per-value anonymization complete!")
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total LLM requests made: {self.total_requests}")
        print(f"  • Cache hits: {self.cache_hits}")
        print(f"  • Failed requests: {len(self.failed_requests)}")
        print(f"  • Success rate: {((self.total_requests - len(self.failed_requests)) / max(self.total_requests, 1) * 100):.1f}%")
        
        return anonymized_df
    
    def process_csv_file(self, input_file: str, output_file: str = None, 
                        columns_to_anonymize: Optional[list] = None,
                        interactive: bool = True,
                        entity_identifying_columns: Optional[List[str]] = None) -> str: # NEW PARAM
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
        anonymized_df = self.anonymize_dataframe(df, columns_to_anonymize, interactive, entity_identifying_columns) # Pass new param
        
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
            'entity_profile_mappings': self.entity_profile_mappings, # NEW
            'failed_requests': list(self.failed_requests),
            'workspace_id': self.workspace_id,
            'model_name': self.model_name,
            'asset_id': self.asset_id,
            'entity_identifying_columns': self.entity_identifying_columns, # NEW
            'statistics': {
                'total_requests': self.total_requests,
                'cache_hits': self.cache_hits,
                'failed_requests': len(self.failed_requests)
            }
        }
        
        # Custom JSON encoder for tuple keys in entity_profile_mappings
        class TupleEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, tuple):
                    return {'__tuple__': True, 'items': list(obj)}
                return json.JSONEncoder.default(self, obj)

        with open(mappings_file, 'w') as f:
            json.dump(data, f, indent=2, cls=TupleEncoder) # Use custom encoder
        
        print(f"💾 Azure OpenAI mappings and cache saved to: {mappings_file}")
    
    def load_mappings_and_cache(self, mappings_file: str):
        """Load mappings and LLM cache"""
        try:
            # Custom JSON decoder for tuple keys
            def tuple_decoder(dct):
                if '__tuple__' in dct:
                    return tuple(dct['items'])
                return dct

            with open(mappings_file, 'r') as f:
                data = json.load(f, object_hook=tuple_decoder) # Use custom decoder
            
            self.mappings = defaultdict(dict, data.get('mappings', {}))
            self.llm_cache = data.get('llm_cache', {})
            self.entity_profile_mappings = data.get('entity_profile_mappings', {}) # NEW
            self.failed_requests = set(data.get('failed_requests', []))
            self.entity_identifying_columns = data.get('entity_identifying_columns', None) # NEW
            
            # Load statistics
            stats = data.get('statistics', {})
            self.total_requests = stats.get('total_requests', 0)
            self.cache_hits = stats.get('cache_hits', 0)
            
            print(f"📂 Azure OpenAI mappings and cache loaded from: {mappings_file}")
            print(f"📊 Loaded {len(self.llm_cache)} cached responses, {len(self.entity_profile_mappings)} entity profiles") # Updated message
        except FileNotFoundError:
            print("📂 No existing Azure OpenAI mappings found, starting fresh")
        except json.JSONDecodeError as e:
            print(f"❌ Error loading mappings from {mappings_file}: {e}. Starting fresh.")
            # Reset states if loading fails
            self.mappings = defaultdict(dict)
            self.llm_cache = {}
            self.entity_profile_mappings = {}
            self.failed_requests = set()
            self.total_requests = 0
            self.cache_hits = 0
            self.entity_identifying_columns = None


def main():
    """Main function to run the Azure OpenAI anonymizer"""
    
    # Initialize the Azure OpenAI anonymizer (credentials from .env file)
    # Define columns that together uniquely identify a person/entity for cross-column consistency
    # IMPORTANT: Adjust these to match the actual column names in your CSV file that identify unique entities.
    # For example, if you have 'Full Name' and 'Email Address' that always belong to the same person:
    # entity_cols = ['Full Name', 'Email Address']
    # If your dataset has a 'Person ID' column, that's often the best single identifier:
    # entity_cols = ['Person ID']
    # If you want NO entity-level anonymization, set it to None or an empty list.
    
    # Example: Adjust based on your CSV's actual column names.
    # For 'Overall Key Legal Entity Contacts - FY 2025_Canada.csv', likely 'Full Name' and 'Email Address'
    entity_cols = ['Full Name', 'Email Address'] 
    
    anonymizer = TRAzureDataAnonymizer(entity_identifying_columns=entity_cols)
    
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
            interactive=True,
            entity_identifying_columns=entity_cols # Pass to process_csv_file
        )
        
        print(f"\n🎉 Processing complete! Output saved to: {result}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    main()