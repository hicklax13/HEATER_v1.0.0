"""Floating chat window chrome: top-right 'AI Chat' launcher + a draggable,
resizable, minimizable, closable window. All chrome behaviors are client-side
(no Streamlit rerun); only sending a message reruns the chat fragment.

The Streamlit chat widgets render inside a streamlit-float container with id
CONTAINER_ID; this module injects the launcher button + the JS that wires drag/
resize/minimize/close and restores position/size/open-state from localStorage.
"""

from __future__ import annotations

import streamlit as st

CONTAINER_ID = "heater-ai-window"
LAUNCHER_ID = "heater-ai-launcher"
LAUNCHER_LABEL = "AI Chat"


def window_frame_css() -> str:
    """Frame styling for the floated chat window — declarations only (no <style>).

    Passed to streamlit-float's float_parent(css=...) so it is scoped to ONLY the
    chat container via that library's unique per-container class. It must NOT be a
    global `div:has(... anchor)` rule: a descendant `:has()` matches every ANCESTOR
    that contains the anchor, including the page's top-level content block, which
    pins the whole page into this 380px box and blanks the main column.
    """
    return (
        "position: fixed; bottom: 24px; right: 24px; width: 380px;"
        # Fixed-height flex column: header + controls sit at the top, the transcript
        # flexes to fill the middle and scrolls internally, and the chat input is
        # pinned at the bottom (the descendant rules live in float_window_css,
        # scoped to .heater-ai-window). Window itself never scrolls.
        "display: flex; flex-direction: column;"
        "height: min(560px, 82vh); min-height: 360px;"
        "min-width: 300px; max-width: 90vw;"
        "resize: both; overflow: hidden; z-index: 99999;"
        "background: #fff; border: 1px solid rgba(0,0,0,.18); border-radius: 12px;"
        "box-shadow: 0 10px 40px rgba(0,0,0,.18);"
    )


def float_window_css() -> str:
    """CSS for the launcher button + window-header affordances ONLY.

    The window FRAME positioning lives in window_frame_css() and is applied
    per-container through float_parent(css=...) — see that function's note on why
    a global :has() frame rule cannot be used here.
    """
    return f"""
    <style>
      #{LAUNCHER_ID} {{
        position: fixed; top: 12px; right: 16px; z-index: 100000;
        background: var(--heater-primary, #ff6d00); color: #fff; border: none;
        border-radius: 8px; padding: 7px 12px; font-weight: 600; cursor: pointer;
        font-family: inherit; display: inline-flex; gap: 6px; align-items: center;
      }}
      #{CONTAINER_ID}-header {{ cursor: move; user-select: none; }}
      .heater-ai-hidden {{ display: none !important; }}
      /* Internal layout (scoped to the JS-tagged .heater-ai-window): header +
         controls at the top, transcript flexes to fill the middle and scrolls,
         chat input pinned at the bottom. The window itself is a flex column
         (set in window_frame_css). */
      .heater-ai-window > div[data-testid="stLayoutWrapper"] {{
        flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column;
      }}
      .heater-ai-window > div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"] {{
        flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; gap: .45rem;
      }}
      .heater-ai-window div[data-testid="stVerticalBlockBorderWrapper"] {{
        flex: 1 1 auto !important; min-height: 0 !important; height: auto !important;
      }}
      .heater-ai-window div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        height: 100% !important; max-height: none !important; overflow-y: auto !important;
      }}
      .heater-ai-window div[data-testid="stChatInput"] {{ flex: 0 0 auto; }}
    </style>
    """


def _shell_script(container_id: str = CONTAINER_ID, launcher_id: str = LAUNCHER_ID) -> str:
    """JS (runs in a components iframe; reaches parent doc same-origin) that wires
    the launcher + drag/resize/minimize/close and restores state from localStorage."""
    return f"""
    <script>
    (function() {{
      const doc = window.parent.document;
      const KEY = 'heaterAiWindowV2';
      function el(id) {{ return doc.getElementById(id); }}
      function winBlock() {{
        const anchor = el('{container_id}-anchor');
        return anchor ? anchor.closest('div[data-testid="stVerticalBlock"]') : null;
      }}
      function save(state) {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch(e) {{}} }}
      function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch(e) {{ return {{}}; }} }}

      function apply() {{
        const w = winBlock(); if (!w) return;
        w.classList.add('heater-ai-window');  // stable hook for the layout rules
        const s = load();
        if (s.left != null) {{ w.style.left = s.left + 'px'; w.style.right = 'auto'; }}
        if (s.top  != null) {{ w.style.top  = s.top  + 'px'; w.style.bottom = 'auto'; }}
        // Size is NOT restored: CSS owns it (auto-height fits the conversation,
        // width:380px default). Live resize still works; it just resets on reload —
        // persisting size froze the window at its initial 540px and left dead space.
        // Default closed: the window is open ONLY when explicitly opened (s.open===true).
        // On first load (s.open undefined) the launcher shows and the window stays hidden,
        // so the two are never visible at once.
        const isOpen = s.open === true;
        w.classList.toggle('heater-ai-hidden', !isOpen);
        const launcher = el('{launcher_id}');
        if (launcher) launcher.style.display = isOpen ? 'none' : 'inline-flex';
      }}

      function wireLauncher() {{
        const launcher = el('{launcher_id}'); if (!launcher || launcher.dataset.wired) return;
        launcher.dataset.wired = '1';
        launcher.addEventListener('click', function() {{
          const s = load(); s.open = true; save(s); apply();
        }});
      }}

      function wireHeader() {{
        const header = el('{container_id}-header'); if (!header || header.dataset.wired) return;
        header.dataset.wired = '1';
        const w = winBlock();
        header.querySelectorAll('[data-ai-act]').forEach(function(btn) {{
          btn.addEventListener('click', function(ev) {{
            ev.stopPropagation();
            const s = load();
            if (btn.dataset.aiAct === 'close' || btn.dataset.aiAct === 'minimize') {{
              s.open = false; save(s); apply();
            }}
          }});
        }});
        let drag = null;
        header.addEventListener('mousedown', function(ev) {{
          if (ev.target.closest('[data-ai-act]')) return;
          const r = w.getBoundingClientRect();
          drag = {{ dx: ev.clientX - r.left, dy: ev.clientY - r.top }};
          ev.preventDefault();
        }});
        doc.addEventListener('mousemove', function(ev) {{
          if (!drag) return;
          w.style.left = (ev.clientX - drag.dx) + 'px'; w.style.right = 'auto';
          w.style.top  = (ev.clientY - drag.dy) + 'px'; w.style.bottom = 'auto';
        }});
        doc.addEventListener('mouseup', function() {{
          if (!drag) return; drag = null;
          const r = w.getBoundingClientRect();
          const s = load(); s.left = r.left; s.top = r.top; save(s);
        }});
      }}

      function tick() {{ wireLauncher(); wireHeader(); apply(); }}
      tick();
      new MutationObserver(tick).observe(doc.body, {{ childList: true, subtree: true }});
    }})();
    </script>
    """


def render_launcher_and_shell() -> None:
    """Inject the launcher button + CSS + behavior JS. Call once per page render."""
    st.markdown(float_window_css(), unsafe_allow_html=True)
    st.markdown(f'<button id="{LAUNCHER_ID}">{LAUNCHER_LABEL}</button>', unsafe_allow_html=True)
    st.components.v1.html(_shell_script(), height=0)
