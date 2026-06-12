<script>
(function () {
  function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === "function") {
      return window.CSS.escape(value);
    }
    return value.replace(/["\\]/g, "\\$&");
  }

  function targetFor(anchor) {
    var href = anchor.getAttribute("href") || "";
    if (!href.startsWith("#")) {
      return null;
    }
    var id = decodeURIComponent(href.slice(1));
    return document.getElementById(id) || document.querySelector('[name="' + cssEscape(id) + '"]');
  }

  function pageFor(element, title) {
    if (window.REPORT_TOC_PAGES && window.REPORT_TOC_PAGES[title]) {
      return window.REPORT_TOC_PAGES[title];
    }

    var pageHeightPx = 11 * 96;
    var top = element.getBoundingClientRect().top + window.scrollY;
    return Math.max(1, Math.floor(top / pageHeightPx) + 1);
  }

  function decorateToc() {
    var toc = document.getElementById("TOC");
    if (!toc) {
      return;
    }

    toc.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
      var target = targetFor(anchor);
      if (!target) {
        return;
      }

      var existingTitle = anchor.querySelector(".toc-title");
      var title = existingTitle ? existingTitle.textContent.trim() : anchor.textContent.trim();
      title = title.replace(/\s+/g, " ").trim();
      anchor.textContent = "";

      var titleSpan = document.createElement("span");
      titleSpan.className = "toc-title";
      titleSpan.textContent = title;

      var leaderSpan = document.createElement("span");
      leaderSpan.className = "toc-leader";
      leaderSpan.setAttribute("aria-hidden", "true");

      var pageSpan = document.createElement("span");
      pageSpan.className = "toc-page";
      pageSpan.textContent = String(pageFor(target, title));

      anchor.appendChild(titleSpan);
      anchor.appendChild(leaderSpan);
      anchor.appendChild(pageSpan);
    });
  }

  decorateToc();

  window.addEventListener("load", function () {
    requestAnimationFrame(function () {
      requestAnimationFrame(decorateToc);
    });
  });

  window.setTimeout(decorateToc, 250);
})();
</script>
