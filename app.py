import streamlit as st
from solar_pv_tool.rag import load_documents, create_vector_store, create_rag_chain

st.title("☀️ Solar PV Compliance Checker")

uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file.name)

    with st.spinner("Processing your files..."):
        docs = load_documents(file_paths)
        db = create_vector_store(docs)
        qa = create_rag_chain(db)

    st.success("Documents processed successfully!")

    query = st.text_input("Ask a compliance question:")
    if query:
        with st.spinner("Analyzing..."):
            answer = qa.run(query)
        st.write("### ✅ Result:")
        st.write(answer)
