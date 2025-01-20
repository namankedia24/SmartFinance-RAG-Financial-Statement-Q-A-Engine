import time
import streamlit as st
from system import RAGSystem

# TODO: develop a Streamlit-based web application that enables
# users to ask questions about a company's financial statements
# for a specific year. The app should include dropdown menus for
# selecting the company ticker and year, alongside a text box for
# entering queries. Responses will be generated using a prebuilt
# Retrieval-Augmented Generation (RAG) pipeline.

# The application must be robust and should never crash, handling
# all errors gracefully with user-friendly messages. It should
# ensure a smooth user experience, even when unexpected issues
# occur during execution.

rag_system = RAGSystem()

# Streamlit app title
st.title("Financial Statement Q&A")

# Dropdown menu for selecting company ticker
tickers = ["BA", "BIIB", "D", "FFIV", "HLT", "HRL", "IDXX", "STZ", "TSLA"]
selected_ticker = st.selectbox("Select a Company Ticker:", tickers)

# Dropdown menu for selecting year
years = [str(year) for year in range(2010, 2020)]  # Years from 2010 to 2020
selected_year = st.selectbox("Select a Year:", years)

# Text input for entering the query
query = st.text_input("Enter your question:")

# Button to submit the query
if st.button("Submit"):
    if not query:
        st.warning("Please enter a question before submitting.")
    else:
        try:
            # Get the response and relevant node from the RAG system
            begin = time.time()
            response, relevant_node = rag_system.respond(query, selected_ticker, selected_year)
            end = time.time()
            # print("Total time taken in : ", func.__name__, end - begin)
            
            # Display the response
            st.subheader("Response:")
            st.write(response)

            # Display the relevant node (context)
            if relevant_node:
                st.subheader("Relevant Context:")
                st.write(relevant_node.text)
            else:
                st.info("No relevant context found.")
            
            st.subheader(f"Approx Response Time: {end - begin:.2f} seconds")
        except Exception as e:
            # Handle errors gracefully
            st.error(f"An error occurred: {str(e)}")
