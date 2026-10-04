// Load the live Excalidraw viewer only when its <details> is first opened.
document.addEventListener("toggle", function (event) {
  var details = event.target;
  if (!details.classList || !details.classList.contains("xd-live") || !details.open) return;
  var frame = details.querySelector("iframe[data-src]");
  if (frame) {
    frame.src = frame.getAttribute("data-src");
    frame.removeAttribute("data-src");
  }
}, true);
