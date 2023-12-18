import streamlit as st

def reset_session_state():
    for key in st.session_state:
        del st.session_state[key]

def fill_suggested_question():
    st.session_state.q = st.session_state.selected_question


