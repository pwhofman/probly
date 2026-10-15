/* The probly wordmark in the middle of the landing page's ecosystem ring is a spring.

   Hovering the wordmark sets it wobbling, and a mouse can pull it by either end and let
   go. Each end of the word is held in place by a spring and a third spring joins the two
   ends, so the word sways, tilts, and stretches (thinning as it stretches) before it comes
   to rest. The sidebar logo stays still: it is on every page, where a wobble on every pass
   of the mouse would distract. */

(function () {
  "use strict";

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return;
  }

  var ANCHOR = 140; // stiffness of the springs that hold the ends in place (1/s^2)
  var LINK = 45; // stiffness of the spring between the ends (1/s^2)
  var DAMPING = 6; // a few visible swings, at rest after about a second (1/s)
  var KICK = 1.5; // speed of the wobble on hover (word widths/s)
  var MAX_PULL = 0.3; // how far a pulled end can get from its rest position (word widths)
  var DRAG_SLOP = 4; // pointer travel (px) after which a press is a pull rather than a click

  function setup(core) {
    var images = Array.prototype.slice.call(core.querySelectorAll("img"));
    // Displacement (px) of the left and the right end of the word from rest, and velocity.
    var ends = [
      { x: 0, y: 0, vx: 0, vy: 0 },
      { x: 0, y: 0, vx: 0, vy: 0 },
    ];
    var width = 0;
    var held = null;
    var grab = null;
    var pulled = false;
    var running = false;
    var lastKick = 0;

    function render(transform) {
      images.forEach(function (image) {
        image.style.transform = transform;
      });
    }

    // At rest the ends sit at -width/2 and +width/2 around the center of the word.
    function span() {
      return { x: width + ends[1].x - ends[0].x, y: ends[1].y - ends[0].y };
    }

    function step(dt) {
      var s = span();
      var length = Math.hypot(s.x, s.y);
      var force = (LINK * (length - width)) / length;
      ends.forEach(function (end, index) {
        var toward = index === 0 ? 1 : -1;
        var ax = -ANCHOR * end.x - DAMPING * end.vx + toward * force * s.x;
        var ay = -ANCHOR * end.y - DAMPING * end.vy + toward * force * s.y;
        if (end === held) {
          return;
        }
        end.vx += ax * dt;
        end.vy += ay * dt;
        end.x += end.vx * dt;
        end.y += end.vy * dt;
      });
    }

    function animate() {
      if (running) {
        return;
      }
      running = true;
      var last = performance.now();
      function frame(now) {
        var dt = Math.min((now - last) / 1000, 1 / 30);
        last = now;
        for (var i = 0; i < 4; i += 1) {
          step(dt / 4);
        }
        var moving = ends.some(function (end) {
          return Math.hypot(end.x, end.y) > 0.15 || Math.hypot(end.vx, end.vy) > 1.5;
        });
        if (!moving && !held) {
          ends.forEach(function (end) {
            end.x = end.y = end.vx = end.vy = 0;
          });
          render("");
          running = false;
          return;
        }
        var s = span();
        var stretch = Math.hypot(s.x, s.y) / width;
        render(
          "translate(" + (ends[0].x + ends[1].x) / 2 + "px, " + (ends[0].y + ends[1].y) / 2 + "px) " +
            "rotate(" + Math.atan2(s.y, s.x) + "rad) scale(" + stretch + ", " + 1 / stretch + ")"
        );
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    }

    core.addEventListener("pointerenter", function () {
      if (held || performance.now() - lastKick < 400) {
        return;
      }
      lastKick = performance.now();
      width = core.offsetWidth;
      ends.forEach(function (end) {
        var angle = Math.random() * 2 * Math.PI;
        var speed = KICK * width * (0.6 + 0.4 * Math.random());
        end.vx += speed * Math.cos(angle);
        end.vy += speed * Math.sin(angle);
      });
      animate();
    });

    // Touch keeps scrolling the page; only a mouse or a pen pulls the word.
    core.addEventListener("pointerdown", function (event) {
      if (event.pointerType === "touch" || event.button !== 0) {
        return;
      }
      event.preventDefault(); // no text selection
      var rect = core.getBoundingClientRect();
      width = core.offsetWidth;
      held = event.clientX < rect.left + rect.width / 2 ? ends[0] : ends[1];
      grab = { x: event.clientX, y: event.clientY, endX: held.x, endY: held.y };
      pulled = false;
      core.setPointerCapture(event.pointerId);
      animate();
    });

    core.addEventListener("pointermove", function (event) {
      if (!held) {
        return;
      }
      var dx = event.clientX - grab.x;
      var dy = event.clientY - grab.y;
      if (!pulled && Math.hypot(dx, dy) < DRAG_SLOP) {
        return;
      }
      pulled = true;
      core.classList.add("is-pulling");
      dx += grab.endX;
      dy += grab.endY;
      // Rubber band: the end follows the pointer at first, then ever more reluctantly.
      var far = Math.hypot(dx, dy);
      var reach = MAX_PULL * width;
      var scale = far > 0 ? (reach * Math.tanh(far / reach)) / far : 1;
      held.x = dx * scale;
      held.y = dy * scale;
      held.vx = held.vy = 0;
    });

    function release() {
      held = null;
      core.classList.remove("is-pulling");
      animate();
    }

    core.addEventListener("pointerup", release);
    core.addEventListener("pointercancel", function () {
      pulled = false;
      release();
    });

    // A pull that ends on the wordmark must not follow its link.
    core.addEventListener("click", function (event) {
      if (pulled) {
        event.preventDefault();
        pulled = false;
      }
    });
    core.addEventListener("dragstart", function (event) {
      event.preventDefault();
    });
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll(".probly-ecosystem__core"), setup);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
