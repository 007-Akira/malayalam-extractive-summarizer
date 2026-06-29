import streamlit as st
# Import your custom AI brain from your existing backend
from summarize import summarize_article 

# Set up the webpage
st.set_page_config(page_title="Malayalam AI Summarizer", page_icon="📰", layout="centered")

st.title("📰 Malayalam Extractive Summarizer")
st.markdown("**Powered by Neuro-Symbolic Dual-Path AI & Dynamic MMR**")
st.divider()

# Input area for the user
raw_text = st.text_area("Paste Malayalam News Article Here:", height=250)

# Controls for the AI
st.subheader("⚙️ AI Tuning Parameters")
col1, col2 = st.columns(2)

with col1:
    k_sentences = st.slider("Number of Sentences to Extract", min_value=1, max_value=10, value=3)

with col2:
    # Add a toggle for the new Dynamic MMR feature!
    use_dmmr = st.checkbox("🤖 Enable Dynamic MMR (Auto-Tune)", value=True, help="Automatically calculates the semantic variance of the article to set the perfect redundancy penalty.")
    
    if use_dmmr:
        diversity_param = "auto"
        st.info("Diversity penalty will be auto-calculated by the AI.")
    else:
        diversity_param = st.slider("Manual Diversity Penalty", min_value=0.0, max_value=1.0, value=0.3)

# The Run Button
if st.button("Summarize Article", type="primary"):
    if raw_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Neuro-Symbolic Hybrid Network analyzing text..."):
            try:
                # Call your exact AI pipeline
                summary_text, extracted_list = summarize_article(
                    raw_text, 
                    k=k_sentences, 
                    diversity=diversity_param
                )
                
                if "Article is too short" in summary_text:
                    st.warning("⚠️ " + summary_text)
                else:
                    st.success("Analysis Complete!")
                    st.subheader("✨ Final Summary")
                    
                    # Print the Malayalam text perfectly using bullet points
                    for sentence in extracted_list:
                        st.markdown(f"- {sentence}")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")