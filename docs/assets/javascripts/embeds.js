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

// Excalidraw posters (hooks/embeds.py). Two states:
//   fit    - the whole drawing at column width, no inner scrolling; a click zooms in there.
//   zoomed - the drawing at a fixed scale inside a pannable stage (drag, scroll, arrow keys).
// Section chips, +/-, Ctrl+wheel and "#sheet-<name>@<n>" links all move between them.
(function () {
  var LEVELS = [0.35, 0.5, 0.7, 1, 1.4];

  function setup(fig) {
    if (fig.dataset.ready) return;
    fig.dataset.ready = "1";
    var stage = fig.querySelector(".xd-stage");
    var img = stage && stage.querySelector("img");
    if (!img) return;
    var readZoom = +fig.dataset.zoom || 0.7;
    var level = fig.querySelector(".xd-level");
    var fullBtn = fig.querySelector('[data-act="full"]');
    var zoom = "fit";

    function natural() { return +img.getAttribute("width") || img.naturalWidth || 1; }
    function fitScale() { return stage.clientWidth / natural(); }
    function label() {
      var s = zoom === "fit" ? fitScale() : zoom;
      level.textContent = (zoom === "fit" ? "Fit · " : "") + Math.round(s * 100) + "%";
    }

    // Set the zoom, keeping image point (fx, fy) (fractions) at stage point (sx, sy) (pixels).
    function setZoom(z, fx, fy, sx, sy) {
      if (z !== "fit" && z <= fitScale() * 1.02 && !document.fullscreenElement) z = "fit";
      zoom = z;
      fig.classList.toggle("zoomed", z !== "fit");
      img.style.width = z === "fit" ? "" : Math.round(natural() * z) + "px";
      if (z !== "fit" && fx !== undefined) {
        stage.scrollLeft = fx * img.clientWidth - sx;
        stage.scrollTop = fy * img.clientHeight - sy;
      }
      label();
    }
    function step(dir, fx, fy, sx, sy) {
      var cur = zoom === "fit" ? fitScale() : zoom;
      var next = dir > 0 ? LEVELS.find(function (l) { return l > cur * 1.02; })
                         : LEVELS.slice().reverse().find(function (l) { return l < cur * 0.98; });
      if (fx === undefined) {  // keep the centre of the view
        sx = stage.clientWidth / 2;
        sy = Math.min(stage.clientHeight, window.innerHeight) / 2;
        fx = (stage.scrollLeft + sx) / img.clientWidth;
        fy = (stage.scrollTop + sy) / img.clientHeight;
      }
      setZoom(next === undefined ? (dir > 0 ? LEVELS[LEVELS.length - 1] : "fit") : next, fx, fy, sx, sy);
    }

    fig.querySelector(".xd-bar").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-act]");
      if (!b) return;
      var act = b.dataset.act;
      if (act === "in") step(1);
      else if (act === "out") step(-1);
      else if (act === "fit") setZoom("fit");
      else if (act === "full") {
        if (document.fullscreenElement) document.exitFullscreen();
        else fig.requestFullscreen().catch(function () {});
      }
    });
    if (!fig.requestFullscreen) fullBtn.hidden = true;
    fig.addEventListener("fullscreenchange", function () {
      var on = document.fullscreenElement === fig;
      fullBtn.textContent = on ? "Exit full screen" : "Full screen";
      if (!on && zoom !== "fit" && zoom <= fitScale()) setZoom("fit");
      label();
    });

    function jump(b) {
      var go = function () {
        setZoom(Math.max(readZoom, zoom === "fit" ? 0 : zoom), +b.dataset.x, +b.dataset.y, 24, 16);
        if (!document.fullscreenElement) {
          var top = fig.getBoundingClientRect().top;
          if (top < 0 || top > window.innerHeight * 0.4) fig.scrollIntoView({ block: "start", behavior: "smooth" });
        }
      };
      img.loading = "eager";
      if (img.complete && img.naturalWidth) go(); else img.addEventListener("load", go, { once: true });
      fig.querySelectorAll(".xd-sections button").forEach(function (o) { o.classList.toggle("on", o === b); });
    }
    fig.querySelectorAll(".xd-sections button").forEach(function (b) {
      b.addEventListener("click", function () { jump(b); });
    });
    fig.jump = function (n) {
      var b = fig.querySelectorAll(".xd-sections button")[n];
      if (b) jump(b); else fig.scrollIntoView({ block: "start" });
    };

    // Pointer: drag pans when zoomed; a click without movement zooms in at that point.
    var drag = null;
    stage.addEventListener("pointerdown", function (e) {
      if (e.button !== 0 || (e.pointerType !== "mouse" && zoom !== "fit")) return;
      drag = { x: e.clientX, y: e.clientY, l: stage.scrollLeft, t: stage.scrollTop, moved: false, id: e.pointerId };
    });
    stage.addEventListener("pointermove", function (e) {
      if (!drag || zoom === "fit" || e.pointerType !== "mouse") return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) > 4) {
        drag.moved = true;
        stage.classList.add("dragging");
        stage.setPointerCapture(drag.id);
      }
      if (drag.moved) {
        stage.scrollLeft = drag.l - dx;
        stage.scrollTop = drag.t - dy;
      }
    });
    stage.addEventListener("pointerup", function (e) {
      var d = drag;
      drag = null;
      stage.classList.remove("dragging");
      if (!d || d.moved || zoom !== "fit") return;
      var r = img.getBoundingClientRect();
      var fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
      var sr = stage.getBoundingClientRect();
      // the zoomed stage is shorter than the fitted image, so put the clicked point mid-stage
      setZoom(readZoom, fx, fy, e.clientX - sr.left, Math.min(stage.clientHeight || 400, 420) / 2);
      if (sr.top < 0) fig.scrollIntoView({ block: "start" });
    });
    stage.addEventListener("pointercancel", function () { drag = null; stage.classList.remove("dragging"); });

    stage.addEventListener("wheel", function (e) {
      if (!e.ctrlKey) return;
      e.preventDefault();
      var r = img.getBoundingClientRect(), sr = stage.getBoundingClientRect();
      step(e.deltaY < 0 ? 1 : -1, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height,
           e.clientX - sr.left, e.clientY - sr.top);
    }, { passive: false });

    stage.addEventListener("keydown", function (e) {
      if (e.key === "+" || e.key === "=") step(1);
      else if (e.key === "-" || e.key === "_") step(-1);
      else if (e.key === "0") setZoom("fit");
      else if (e.key === "f" || e.key === "F") fullBtn.click();
      else if (e.key === "Escape" && zoom !== "fit" && !document.fullscreenElement) setZoom("fit");
      else return;
      e.preventDefault();
    });

    if (window.ResizeObserver) new ResizeObserver(label).observe(stage);
    img.addEventListener("load", label);
    label();
  }

  function init() {
    document.querySelectorAll(".xd-poster").forEach(setup);
    document.querySelectorAll('a[href*="#sheet-"]').forEach(function (a) {
      if (a.dataset.sheet) return;
      a.dataset.sheet = "1";
      a.addEventListener("click", function (e) {
        var m = a.getAttribute("href").match(/#sheet-([\w-]+)@(\d+)$/);
        var img = m && document.querySelector('.xd-poster img[src$="/' + m[1] + '.svg"]');
        if (!img) return;
        e.preventDefault();
        img.closest(".xd-poster").jump(+m[2]);
      });
    });
    // Arriving from another page with "#sheet-<name>@<n>" in the URL.
    var h = location.hash.match(/^#sheet-([\w-]+)@(\d+)$/);
    var target = h && document.querySelector('.xd-poster img[src$="/' + h[1] + '.svg"]');
    if (target && !target.dataset.arrived) {
      target.dataset.arrived = "1";
      target.closest(".xd-poster").jump(+h[2]);
    }
  }
  // document$ (Material instant navigation) re-runs init after page swaps; it does not emit when the
  // site is opened from file://, so also run on first load. init() is idempotent.
  if (window.document$) window.document$.subscribe(init);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
