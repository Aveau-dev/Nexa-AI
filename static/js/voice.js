window.Voice = (function () {
  let recognition = null;
  let listening = false;

  function onViewLoaded() {
    setStatus("Microphone: ready (browser support required).");
  }

  function setStatus(msg) {
    const el = document.getElementById("voice-status");
    if (el) el.textContent = msg;
  }

  function start() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return alert("SpeechRecognition not supported in this browser.");

    recognition = new SR();
    recognition.lang = "en-US";
    recognition.interimResults = true;

    recognition.onresult = (e) => {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
      const ta = document.getElementById("voice-transcript");
      if (ta) ta.value = text;
    };

    recognition.onerror = () => setStatus("Microphone: error.");
    recognition.onend = () => {
      listening = false;
      setStatus("Microphone: stopped.");
    };

    listening = true;
    recognition.start();
    setStatus("Microphone: listening...");
  }

  function stop() {
    if (recognition && listening) recognition.stop();
  }

  function sendToChat() {
    const text = (document.getElementById("voice-transcript")?.value || "").trim();
    if (!text) return alert("Transcript is empty.");
    Router.go("chat").then(() => Chat.useSuggestion(text));
  }

  return { onViewLoaded, start, stop, sendToChat };
})();
