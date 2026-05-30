"""Admin dashboard needs Streamlit >= 1.37 (st.navigation + st.fragment run_every)."""

from importlib.metadata import version

from packaging.version import Version


def test_streamlit_at_least_1_37():
    assert Version(version("streamlit")) >= Version("1.37"), (
        "Admin dashboard nav (st.navigation) + heartbeat (st.fragment run_every) need Streamlit >= 1.37"
    )
