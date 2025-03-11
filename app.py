import streamlit as st

page1 = st.Page(
    page = "pages/1.py",
    title = "Data & Algorithms",
    icon = ":material/dataset:",
    default = True,
)

page2 = st.Page(
    page = "pages/2.py",
    title = "Model Development Guidelines",
    icon = ":material/deployed_code_history:",
)

page3 = st.Page(
    page = "pages/machine.py",
    title = "Machine Learning Model",
    icon = ":material/model_training:",
)

page4 = st.Page(
    page = "pages/neural.py",
    title = "Neural Network Model",
    icon = ":material/neurology:",
)

pg = st.navigation(
    {
        "Introduction": [page1, page2],
        "Model": [page3, page4]
    }
)

st.sidebar.markdown(
    """
    <style>
        .custom-text {
            text-align: center;
            font-family: 'RSU';
            font-size: 16px;
            font-weight: bold;
            color: #
        }
    </style>
    <p class='custom-text'>KANANAD SANGSAK | 6604062610250</p>
    <p class='custom-text'></p>
    """,
    unsafe_allow_html=True
)

pg.run()