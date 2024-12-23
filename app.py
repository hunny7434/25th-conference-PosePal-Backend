import streamlit as st
from pages.first_page import first_page
from pages.second_page import second_page
from utils.model.Rocket import RocketTransformerClassifier

def main():

    st.title("PosePal 💪")

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 1

    # Page navigation
    if st.session_state.page == 1:
        first_page()
    elif st.session_state.page == 2:
        second_page()

if __name__ == "__main__":
    main()
