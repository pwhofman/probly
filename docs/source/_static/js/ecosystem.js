/* Pointer behaviour of the landing page ecosystem ring.

   The ring lays itself out in CSS; this only decides how the marks react to each
   other. Hovering (or tabbing to) a mark marks it active and gives every other mark a
   --push between -1 and 1, so its neighbours slide along the ring away from it and
   marks further around the ring stay where they are. */

(function () {
  "use strict";

  // How far around the ring the nudge reaches, in node slots.
  var REACH = 2.5;

  function setup(ring) {
    var nodes = Array.prototype.slice.call(ring.querySelectorAll(".probly-ecosystem__node"));
    var core = ring.querySelector(".probly-ecosystem__core");
    if (!nodes.length) {
      return;
    }

    function focus(index) {
      ring.classList.add("is-active");
      nodes.forEach(function (node, other) {
        // Signed distance around the ring, wrapped into [-count/2, count/2].
        var delta = other - index;
        if (delta > nodes.length / 2) {
          delta -= nodes.length;
        } else if (delta < -nodes.length / 2) {
          delta += nodes.length;
        }
        var push = delta === 0 ? 0 : Math.sign(delta) * Math.max(0, 1 - Math.abs(delta) / REACH);
        node.style.setProperty("--push", push.toFixed(3));
        node.classList.toggle("is-hovered", other === index);
      });
    }

    function clear() {
      ring.classList.remove("is-active");
      nodes.forEach(function (node) {
        node.style.setProperty("--push", "0");
        node.classList.remove("is-hovered");
      });
    }

    nodes.forEach(function (node, index) {
      node.addEventListener("pointerenter", function () {
        focus(index);
      });
      node.addEventListener("pointerleave", clear);
      node.addEventListener("focus", function () {
        focus(index);
      });
      node.addEventListener("blur", clear);
    });

    if (core) {
      core.addEventListener("pointerenter", function () {
        ring.classList.add("is-core-active");
      });
      core.addEventListener("pointerleave", function () {
        ring.classList.remove("is-core-active");
      });
    }
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll(".probly-ecosystem"), setup);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
