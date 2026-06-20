import streamlit as st
# Import your custom AI brain from your existing file
from summarize import summarize_article 

# Set up the webpage
st.set_page_config(page_title="Malayalam AI Summarizer", page_icon="📰", layout="centered")

st.title("📰 Malayalam Extractive Summarizer")
st.markdown("**Powered by LaBSE and Custom Neural Network (M5-Trained)**")
st.divider()

# Input area for the user
raw_text = st.text_area("Paste Malayalam News Article Here:", height=250)

# Controls for the AI
col1, col2 = st.columns(2)
with col1:
    k_sentences = st.slider("Number of Sentences to Extract", min_value=1, max_value=10, value=3)
with col2:
    diversity = st.slider("MMR Diversity Penalty (Redundancy Filter)", min_value=0.0, max_value=1.0, value=0.3)

# The Run Button
if st.button("Summarize Article", type="primary"):
    if raw_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Neural Network analyzing text..."):
            try:
                # Call your exact AI pipeline
                summary_text, extracted_list = summarize_article(
                    raw_text, 
                    k=k_sentences, 
                    diversity=diversity
                )
                
                st.success("Analysis Complete!")
                st.subheader("✨ Final Summary")
                
                # Print the Malayalam text perfectly using bullet points
                for sentence in extracted_list:
                    st.markdown(f"- {sentence}")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")


