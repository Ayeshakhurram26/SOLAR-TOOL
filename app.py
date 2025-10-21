import streamlit as st
from solar_pv_tool.rag import load_documents, create_vector_store, create_rag_chain

st.set_page_config(page_title="☀️ Solar PV Compliance Tool", layout="wide")

st.title("☀️ Solar PV Compliance Tool")
st.write("Upload your solar PV system files below to check compliance with standards.")

uploaded_files = st.file_uploader("Upload files (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.info("🔍 Processing uploaded files...")
    documents = load_documents(uploaded_files)
    vector_store = create_vector_store(documents)
    rag_chain = create_rag_chain(vector_store)

    query = st.text_input("Enter a compliance check query", "Does this design meet PV connection standards?")
    if st.button("Check Compliance"):
        with st.spinner("Analyzing..."):
            result = rag_chain.invoke({"input": query})
            st.success("✅ Analysis Complete")
            st.write(result["answer"])
else:
    st.warning("Please upload files to start the analysis.")
