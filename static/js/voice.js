/* static/js/voice.js
   Voice mode using Web Speech API
   - Starts/stops recognition
   - Streams transcript to #voice-transcript
*/

const Voice = (function () {
  let recognition = null;
  let running = false;
  const transcriptElId = "voice-transcript";
  const logElId = "voice-logs";

  function init() {
    // populate device list (if desired)
    if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
      appendLog("Speech recognition not supported in this browser");
      return;
    }
    const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new Rec();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (ev) => {
      let interim = "";
      let final = "";
      for (let i = ev.resultIndex; i < ev.results.length; ++i) {
        const r = ev.results[i];
        if (r.isFinal) final += r[0].transcript;
        else interim += r[0].transcript;
      }
      const el = document.getElementById(transcriptElId);
      if (el) el.textContent = final + interim;
    };

    recognition.onerror = (e) => {
      appendLog("Recognition error: " + e.error);
    };

    recognition.onend = () => {
      running = false;
      appendLog("Recognition stopped");
    };

    appendLog("Voice initialized");
  }

  function appendLog(msg) {
    const el = document.getElementById(logElId);
    if (!el) return;
    const d = document.createElement("div");
    d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.prepend(d);
  }

  function start() {
    if (!recognition) init();
    try {
      recognition.start();
      running = true;
      appendLog("Listening...");
    } catch (e) {
      appendLog("Start error: " + e.message);
    }
  }

  function stop() {
    if (!recognition) return;
    recognition.stop();
    running = false;
    appendLog("Stopped");
  }

  return {
    init, start, stop
  };
})();

window.Voice = Voice;
