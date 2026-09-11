/* Automation Miner - ingress UI.
   No framework and no CDN: this has to work on an offline instance. */
(function () {
  "use strict";

  var base = window.AMMINER_BASE || "";
  var toast = document.getElementById("toast");
  var toastTimer = null;

  function notify(message, isError) {
    if (!toast) { return; }
    toast.textContent = message;
    toast.hidden = false;
    toast.style.background = isError ? "#c62828" : "";
    toast.style.color = isError ? "#fff" : "";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toast.hidden = true; }, 4000);
  }

  function post(path, body) {
    return fetch(base + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body)
    }).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (body) {
        if (!response.ok) {
          throw new Error(body.detail || body.error || ("HTTP " + response.status));
        }
        return body;
      });
    });
  }

  function removeCard(id) {
    var card = document.getElementById("s-" + id);
    if (card) { card.remove(); return; }
    var button = document.querySelector('[data-id="' + id + '"]');
    if (button && button.closest("li")) { button.closest("li").remove(); }
  }

  var handlers = {
    dismiss: function (id, button) {
      if (!window.confirm("Dismiss this permanently? It will not be suggested again.")) {
        return null;
      }
      return post("/api/suggestions/" + id + "/dismiss").then(function () {
        notify("Dismissed. It will not come back.");
        removeCard(id);
      });
    },
    restore: function (id) {
      return post("/api/suggestions/" + id + "/restore").then(function () {
        notify("Restored to the suggestion list.");
        removeCard(id);
      });
    },
    shadow: function (id) {
      return post("/api/suggestions/" + id + "/shadow").then(function () {
        notify("Shadow-testing: it will be logged when it would fire, but will not act.");
      });
    },
    apply: function (id, confirmed) {
      if (!confirmed &&
          !window.confirm("Write this automation to Home Assistant and reload automations?")) {
        return null;
      }
      var url = "/api/suggestions/" + id + "/apply" + (confirmed ? "?confirm=true" : "");
      return post(url).then(function (result) {
        if (result.ok) {
          notify("Applied as " + (result.automation_id || "a new automation") +
                 (result.reloaded ? " and reloaded." : ". Reload automations manually."));
          return;
        }
        if (result.needs_confirmation) {
          // Not a failure: this rule fights an automation they already have,
          // and they have not been asked about that specifically yet.
          var lines = (result.conflicts || []).map(function (c) { return "\u2022 " + c.message; });
          if (window.confirm(
                "This rule conflicts with an automation you already have:\n\n" +
                lines.join("\n") + "\n\nApply it anyway?")) {
            return handlers.apply(id, true);
          }
          notify("Not applied.");
          return;
        }
        notify((result.errors || ["Apply failed"]).join(" "), true);
      });
    },
    "preference-off": function (id) {
      return post("/api/preferences/" + id + "/off").then(function () {
        notify("Switched off. Anything it was hiding is back.");
        window.location.reload();
      });
    },
    "preference-on": function (id) {
      return post("/api/preferences/" + id + "/on").then(function () {
        notify("Switched on. It applies from the next analysis.");
        window.location.reload();
      });
    },
    "preference-draft": function (id) {
      // The model only ever fills in the box. Saving stays a deliberate act,
      // because a rule that hides suggestions should never be written by
      // anything but the person it hides them from.
      var input = id
        ? document.querySelector('[data-rule-for="' + id + '"]')
        : document.getElementById("new-preference");
      if (!input) { return null; }
      var instruction = window.prompt(
        "What should change about this preference?\n\n" +
        "For example: \"only the lamp, not the whole room\", or " +
        "\"this should not apply at weekends\".",
        "");
      if (!instruction || !instruction.trim()) { return null; }
      notify("Asking\u2026");
      return post("/api/preferences/draft", {
        preference: id || undefined,
        rule: input.value,
        instruction: instruction
      }).then(function (result) {
        input.value = result.rule;
        input.focus();
        notify("Drafted. Check the wording, then press Save - nothing is stored yet.");
      });
    },
    "preference-save": function (id) {
      var input = document.querySelector('[data-rule-for="' + id + '"]');
      if (!input || !input.value.trim()) {
        notify("A preference needs a rule.", true);
        return null;
      }
      return post("/api/preferences/" + id, { rule: input.value }).then(function () {
        notify("Saved. That wording is yours now - no later run will change it.");
        window.location.reload();
      });
    },
    "preference-delete": function (id) {
      if (!window.confirm(
            "Delete this preference? Anything it was hiding comes back.\n\n" +
            "If it was learned and the dismissals behind it are still on record, " +
            "a later run may learn something like it again. Switching it off instead " +
            "is permanent.")) {
        return null;
      }
      return post("/api/preferences/" + id + "/delete").then(function () {
        notify("Deleted.");
        window.location.reload();
      });
    },
    "preference-add": function () {
      var input = document.getElementById("new-preference");
      if (!input || !input.value.trim()) {
        notify("A preference needs a rule.", true);
        return null;
      }
      return post("/api/preferences", { rule: input.value }).then(function () {
        notify("Added. It applies from the next analysis.");
        window.location.reload();
      });
    },
    "dismiss-gap": function (id) {
      return post("/api/gaps/" + id + "/dismiss").then(function () {
        notify("Hidden.");
        var card = document.querySelector('[data-id="' + id + '"]');
        if (card && card.closest("article")) { card.closest("article").remove(); }
      });
    }
  };

  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-action]");
    if (!button) { return; }
    var handler = handlers[button.getAttribute("data-action")];
    if (!handler) { return; }
    event.preventDefault();
    var id = button.getAttribute("data-id");
    button.disabled = true;
    var promise = handler(id, button);
    if (!promise) { button.disabled = false; return; }
    promise.catch(function (error) {
      notify(error.message, true);
    }).then(function () {
      button.disabled = false;
    });
  });

  var runButton = document.getElementById("run-now");
  if (runButton) {
    runButton.addEventListener("click", function () {
      runButton.disabled = true;
      runButton.textContent = "Analysing…";
      post("/api/run").then(function (result) {
        if (result.status === "already_running") {
          notify("An analysis is already running.");
        } else {
          notify("Analysis started. This page will refresh when it finishes.");
          pollUntilDone();
        }
      }).catch(function (error) {
        notify(error.message, true);
        runButton.disabled = false;
        runButton.textContent = "Analyse now";
      });
    });
  }

  function pollUntilDone() {
    var attempts = 0;
    var timer = setInterval(function () {
      attempts += 1;
      fetch(base + "/api/health").then(function (r) { return r.json(); }).then(function (health) {
        if (!health.running) {
          clearInterval(timer);
          window.location.reload();
        } else if (attempts > 300) {
          clearInterval(timer);
          notify("Analysis is taking a long time; check the Status page.", true);
          if (runButton) {
            runButton.disabled = false;
            runButton.textContent = "Analyse now";
          }
        }
      }).catch(function () { /* transient during a restart - keep polling */ });
    }, 2000);
  }
})();
