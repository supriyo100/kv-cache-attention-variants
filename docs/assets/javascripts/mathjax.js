// MathJax 3 config for pymdownx.arithmatex (generic mode), re-typeset on instant navigation.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "none"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  chtml: { scale: 1.0, matchFontHeight: false }
};

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (!window.MathJax || !MathJax.startup || !MathJax.typesetPromise) return;
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}
