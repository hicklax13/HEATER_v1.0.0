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


def float_window_css() -> str:
    """CSS for the floated container: window frame, resizable, draggable header."""
    return f"""
    <style>
      #{LAUNCHER_ID} {{
        position: fixed; top: 12px; right: 16px; z-index: 100000;
        background: var(--heater-primary, #ff6d00); color: #fff; border: none;
        border-radius: 8px; padding: 7px 12px; font-weight: 600; cursor: pointer;
        font-family: inherit; display: inline-flex; gap: 6px; align-items: center;
      }}
      div[data-testid="stVerticalBlock"]:has(> div #{CONTAINER_ID}-anchor) {{
        position: fixed; bottom: 24px; right: 24px; width: 380px; height: 540px;
        min-width: 300px; min-height: 240px; max-width: 90vw; max-height: 90vh;
        resize: both; overflow: auto; z-index: 99999;
        background: #fff; border: 1px solid rgba(0,0,0,.18); border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0,0,0,.18);
      }}
      #{CONTAINER_ID}-header {{ cursor: move; user-select: none; }}
      .heater-ai-hidden {{ display: none !important; }}
    </style>
    """


def _shell_script(container_id: str = CONTAINER_ID, launcher_id: str = LAUNCHER_ID) -> str:
    """JS (runs in a components iframe; reaches parent doc same-origin) that wires
    the launcher + drag/resize/minimize/close and restores state from localStorage."""
    return f"""
    <script>
    (function() {{
      const doc = window.parent.document;
      const KEY = 'heaterAiWindow';
      function el(id) {{ return doc.getElementById(id); }}
      function winBlock() {{
        const anchor = el('{container_id}-anchor');
        return anchor ? anchor.closest('div[data-testid="stVerticalBlock"]') : null;
      }}
      function save(state) {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch(e) {{}} }}
      function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch(e) {{ return {{}}; }} }}

      function apply() {{
        const w = winBlock(); if (!w) return;
        const s = load();
        if (s.left != null) {{ w.style.left = s.left + 'px'; w.style.right = 'auto'; }}
        if (s.top  != null) {{ w.style.top  = s.top  + 'px'; w.style.bottom = 'auto'; }}
        if (s.width)  w.style.width  = s.width + 'px';
        if (s.height) w.style.height = s.height + 'px';
        w.classList.toggle('heater-ai-hidden', s.open === false);
        const launcher = el('{launcher_id}');
        if (launcher) launcher.style.display = (s.open === false || s.open == null) ? 'inline-flex' : 'none';
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
          const s = load(); s.left = r.left; s.top = r.top; s.width = r.width; s.height = r.height; save(s); apply();
        }});
        new ResizeObserver(function() {{
          const r = w.getBoundingClientRect();
          const s = load(); s.width = r.width; s.height = r.height; save(s);
        }}).observe(w);
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
