"""Keep the copy buttons working when the app is not served from a secure origin.

Every copy affordance Streamlit draws — the button on each ``st.code`` block, the
cell copy in a dataframe — goes through ``navigator.clipboard``. Browsers expose that
API only in a *secure context*: HTTPS, or a ``localhost`` address. Opened at the
machine's LAN address, which is the second URL Streamlit prints on start-up and the
only one that works from another device, ``navigator.clipboard`` is undefined and
every one of those buttons silently does nothing — no copy, no error, no feedback.

The pre-Clipboard-API path — select a hidden ``<textarea>``, run
``document.execCommand("copy")`` — carries no such restriction and is still honoured
by every current browser. Installing it under the name Streamlit already calls fixes
the buttons where they are, rather than adding a second set beside them.

Only ``writeText`` is filled in. Streamlit's richer paths (``clipboard.write`` with a
``ClipboardItem``, used when a dataframe selection is copied as text *and* HTML)
already degrade to ``writeText`` when ``write`` is absent, so leaving it absent is
what routes them here.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

# Runs in the sandboxed iframe Streamlit creates, and patches the page that hosts it.
# The sandbox grants ``allow-same-origin``, so reaching the parent is allowed; the
# guard is there for the day it is not.
_FALLBACK = """
<script>
(function () {
    var host, doc, nav;
    try {
        host = window.parent;
        doc = host.document;
        nav = host.navigator;
    } catch (err) {
        return;  // Cross-origin: leave the page exactly as it was.
    }

    // The iframe is a delivery mechanism, not an element of the page.
    var frame = window.frameElement;
    var block = frame && frame.closest('[data-testid="stElementContainer"]');
    if (block) { block.style.display = "none"; }

    // Every rerun re-renders this component; the patch is installed once.
    if (host.__nl2sqlClipboardFallback) { return; }
    if (nav.clipboard && nav.clipboard.writeText) { return; }  // Secure origin.

    // Must stay synchronous: browsers honour execCommand only while the click that
    // asked for the copy is still being handled, so nothing here may await.
    function copyText(text) {
        var area = doc.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        // Off-screen would scroll the page on focus; a transparent 1px box does not.
        area.style.cssText =
            "position:fixed;top:0;left:0;width:1px;height:1px;" +
            "padding:0;border:0;opacity:0;";
        doc.body.appendChild(area);

        var selection = doc.getSelection();
        var previous = selection && selection.rangeCount
            ? selection.getRangeAt(0)
            : null;
        var focused = doc.activeElement;

        var copied = false;
        try {
            area.focus();
            area.select();
            area.setSelectionRange(0, area.value.length);
            copied = doc.execCommand("copy");
        } catch (err) {
            copied = false;
        } finally {
            doc.body.removeChild(area);
            // Whatever the user had selected or focused is theirs, not ours.
            if (previous && selection) {
                selection.removeAllRanges();
                selection.addRange(previous);
            }
            if (focused && focused.focus) { focused.focus(); }
        }
        return copied;
    }

    if (!nav.clipboard) {
        Object.defineProperty(nav, "clipboard", {value: {}, configurable: true});
    }
    nav.clipboard.writeText = function (text) {
        // Rejecting on failure keeps the contract: Streamlit shows the tick only
        // once the promise resolves, so a refused copy must not look like a copy.
        return copyText(String(text))
            ? Promise.resolve()
            : Promise.reject(new Error("The browser refused the copy."));
    };

    host.__nl2sqlClipboardFallback = true;
})();
</script>
"""


def install() -> None:
    """Install the clipboard fallback. Safe to call on every rerun."""
    # ``st.iframe`` superseded ``components.html`` and the older name is scheduled
    # for removal; the declared floor is 1.40, which predates the rename. One pixel
    # rather than none: ``st.iframe`` rejects a height of zero, and the script hides
    # its own container anyway.
    if hasattr(st, "iframe"):
        st.iframe(_FALLBACK, height=1)
    else:
        components.html(_FALLBACK, height=1)
