import streamlit as st
import pandas as pd
import io
import json
from pathlib import Path
import tempfile
import os
from anonymizer import TRAzureDataAnonymizer

# Set page config
st.set_page_config(
    page_title="Azure OpenAI Data Anonymizer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    color: #1e3a8a;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 1rem 0;
    color: #3b82f6;
}
.info-box {
    background-color: #f0f9ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'anonymizer' not in st.session_state:
        st.session_state.anonymizer = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'columns_to_anonymize' not in st.session_state:
        st.session_state.columns_to_anonymize = []
    if 'anonymized_df' not in st.session_state:
        st.session_state.anonymized_df = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">🔒 Azure OpenAI Data Anonymizer</div>', unsafe_allow_html=True)
    st.markdown("Securely anonymize sensitive data using Thomson Reuters Azure OpenAI")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sub-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # Azure OpenAI Configuration
        workspace_id = st.text_input(
            "Workspace ID", 
            value=os.getenv("WORKSPACE_ID", "AnonymizerW5XR"),
            help="Your Thomson Reuters workspace ID"
        )
        
        model_name = st.text_input(
            "Model Name", 
            value=os.getenv("MODEL_NAME", "gpt-4o"),
            help="Azure OpenAI model to use"
        )
        
        asset_id = st.text_input(
            "Asset ID", 
            value=os.getenv("ASSET_ID", "208321"),
            help="Thomson Reuters asset ID"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            min_request_interval = st.slider(
                "Request Interval (seconds)", 
                min_value=0.1, 
                max_value=2.0, 
                value=0.2, 
                step=0.1,
                help="Minimum time between API requests"
            )
            
            max_tokens = st.slider(
                "Max Tokens", 
                min_value=50, 
                max_value=500, 
                value=100,
                help="Maximum tokens for LLM response"
            )
            
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.0,
                help="LLM creativity level (0 = deterministic)"
            )
    
    # Initialize anonymizer with current settings
    if (st.session_state.anonymizer is None or 
        st.session_state.anonymizer.workspace_id != workspace_id or
        st.session_state.anonymizer.model_name != model_name or
        st.session_state.anonymizer.asset_id != asset_id):
        
        with st.spinner("🔄 Initializing Azure OpenAI connection..."):
            try:
                # Set environment variables for the anonymizer
                os.environ["MIN_REQUEST_INTERVAL"] = str(min_request_interval)
                os.environ["MAX_TOKENS"] = str(max_tokens)
                os.environ["TEMPERATURE"] = str(temperature)
                
                st.session_state.anonymizer = TRAzureDataAnonymizer(
                    workspace_id=workspace_id,
                    model_name=model_name,
                    asset_id=asset_id
                )
                st.success("✅ Azure OpenAI connection established!")
            except Exception as e:
                st.error(f"❌ Failed to initialize Azure OpenAI: {e}")
                return
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🔍 Column Selection", "🚀 Anonymization"])
    
    with tab1:
        st.markdown('<div class="sub-header">📁 Upload Your Data</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload the CSV file you want to anonymize"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                st.session_state.df = pd.read_csv(uploaded_file)
                
                # Display file info
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"**📊 File loaded successfully!**")
                st.write(f"• Rows: {len(st.session_state.df):,}")
                st.write(f"• Columns: {len(st.session_state.df.columns)}")
                st.write(f"• File size: {uploaded_file.size:,} bytes")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show preview
                st.markdown("**📋 Data Preview (first 5 rows):**")
                st.dataframe(st.session_state.df.head(), use_container_width=True)
                
                # Reset processing state when new file is uploaded
                st.session_state.processing_complete = False
                st.session_state.anonymized_df = None
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with tab2:
        if st.session_state.df is not None:
            st.markdown('<div class="sub-header">🔍 Select Columns to Anonymize</div>', unsafe_allow_html=True)
            
            # Auto-detection option
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Auto-Detect Sensitive Columns", use_container_width=True):
                    with st.spinner("🔍 Analyzing columns for sensitive content..."):
                        auto_detected = st.session_state.anonymizer.auto_detect_columns_to_anonymize(st.session_state.df)
                        st.session_state.columns_to_anonymize = auto_detected
                        st.success(f"✅ Auto-detected {len(auto_detected)} columns")
            
            with col2:
                if st.button("🗑️ Clear Selection", use_container_width=True):
                    st.session_state.columns_to_anonymize = []
                    st.success("✅ Selection cleared")
            
            # Manual column selection
            st.markdown("**📋 Column Analysis:**")
            
            # Create column analysis
            column_analysis = []
            for col in st.session_state.df.columns:
                sample_values = st.session_state.df[col].dropna().head(3).tolist()
                should_anonymize = st.session_state.anonymizer.should_anonymize_column(col, sample_values)
                unique_count = st.session_state.df[col].nunique()
                
                column_analysis.append({
                    'Column': col,
                    'Sample Values': str(sample_values),
                    'Unique Count': unique_count,
                    'Auto-Detected': '🔴 Sensitive' if should_anonymize else '🟢 Safe',
                    'Selected': col in st.session_state.columns_to_anonymize
                })
            
            analysis_df = pd.DataFrame(column_analysis)
            
            # Interactive column selection
            selected_columns = st.multiselect(
                "Select columns to anonymize:",
                options=st.session_state.df.columns.tolist(),
                default=st.session_state.columns_to_anonymize,
                help="Choose which columns contain sensitive data that needs anonymization"
            )
            
            st.session_state.columns_to_anonymize = selected_columns
            
            # Display analysis table
            st.dataframe(analysis_df, use_container_width=True)
            
            if selected_columns:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"**🎯 Selected {len(selected_columns)} columns for anonymization:**")
                for col in selected_columns:
                    unique_count = st.session_state.df[col].nunique()
                    st.write(f"• **{col}**: {unique_count} unique values")
                
                total_unique_values = sum(st.session_state.df[col].nunique() for col in selected_columns)
                estimated_time = total_unique_values * min_request_interval / 60
                st.write(f"• **⏱️ Estimated processing time**: ~{estimated_time:.1f} minutes")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📁 Please upload a CSV file first")
    
    with tab3:
        if st.session_state.df is not None and st.session_state.columns_to_anonymize:
            st.markdown('<div class="sub-header">🚀 Anonymization Process</div>', unsafe_allow_html=True)
            
            # Processing summary
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("**📋 Processing Summary:**")
            st.write(f"• **File**: {uploaded_file.name if uploaded_file else 'Unknown'}")
            st.write(f"• **Rows**: {len(st.session_state.df):,}")
            st.write(f"• **Columns to anonymize**: {len(st.session_state.columns_to_anonymize)}")
            st.write(f"• **Model**: {model_name}")
            st.write(f"• **Workspace**: {workspace_id}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Start anonymization
            if st.button("🚀 Start Anonymization", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process the anonymization
                    with st.spinner("🔄 Processing data with Azure OpenAI..."):
                        st.session_state.anonymized_df = st.session_state.anonymizer.anonymize_dataframe(
                            st.session_state.df,
                            columns_to_anonymize=st.session_state.columns_to_anonymize,
                            interactive=False
                        )
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Anonymization completed successfully!")
                    st.session_state.processing_complete = True
                    
                    # Show statistics
                    stats = st.session_state.anonymizer
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔄 Total Requests", stats.total_requests)
                    with col2:
                        st.metric("💾 Cache Hits", stats.cache_hits)
                    with col3:
                        st.metric("❌ Failed Requests", len(stats.failed_requests))
                    with col4:
                        success_rate = ((stats.total_requests - len(stats.failed_requests)) / max(stats.total_requests, 1) * 100)
                        st.metric("✅ Success Rate", f"{success_rate:.1f}%")
                    
                except Exception as e:
                    st.error(f"❌ Error during anonymization: {e}")
                    progress_bar.empty()
                    status_text.empty()
            
            # Show results and download options
            if st.session_state.processing_complete and st.session_state.anonymized_df is not None:
                st.markdown("---")
                st.markdown('<div class="sub-header">📊 Results</div>', unsafe_allow_html=True)
                
                # Before/After comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📋 Original Data (sample):**")
                    st.dataframe(st.session_state.df[st.session_state.columns_to_anonymize].head(), use_container_width=True)
                
                with col2:
                    st.markdown("**🔒 Anonymized Data (sample):**")
                    st.dataframe(st.session_state.anonymized_df[st.session_state.columns_to_anonymize].head(), use_container_width=True)
                
                # Download options
                st.markdown("---")
                st.markdown('<div class="sub-header">💾 Download Results</div>', unsafe_allow_html=True)
                
                # Prepare download data
                csv_buffer = io.StringIO()
                st.session_state.anonymized_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # Prepare mappings data
                mappings_data = {
                    'mappings': dict(st.session_state.anonymizer.mappings),
                    'llm_cache': st.session_state.anonymizer.llm_cache,
                    'statistics': {
                        'total_requests': st.session_state.anonymizer.total_requests,
                        'cache_hits': st.session_state.anonymizer.cache_hits,
                        'failed_requests': len(st.session_state.anonymizer.failed_requests)
                    }
                }
                mappings_json = json.dumps(mappings_data, indent=2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Anonymized CSV",
                        data=csv_data,
                        file_name=f"anonymized_{uploaded_file.name if uploaded_file else 'data.csv'}",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download Mappings (JSON)",
                        data=mappings_json,
                        file_name="anonymization_mappings.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Full anonymized data preview
                with st.expander("🔍 View Full Anonymized Data"):
                    st.dataframe(st.session_state.anonymized_df, use_container_width=True)
        
        elif st.session_state.df is None:
            st.info("📁 Please upload a CSV file first")
        else:
            st.info("🔍 Please select columns to anonymize first")

if __name__ == "__main__":
    main()