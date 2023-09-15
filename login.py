import streamlit as st
def authenticate(username, password):
    # Replace with your authentication logic
    if username == "your_username" and password == "your_password":
        return True
    return False
st.subheader("Login")
username = st.text_input("Username")
password = st.text_input("Password", type="password")
if st.button("Login", key="login_button_key"):
    if authenticate(username, password):
        st.success("Login successful!")
    else:
        st.error("Authentication failed. Please try again.")
if st.button("Home_Login"):
    st.session_state.page = "Home"
if st.button("Register_Login"):
    st.session_state.page = "Register"