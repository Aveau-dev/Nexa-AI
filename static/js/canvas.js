window.Canvas = (function () {
  let ctx = null;
  let drawing = false;
  let brushSize = 6;
  let brushColor = "#19c37d";
  const history = [];

  function onViewLoaded() {
    const canvas = document.getElementById("main-canvas");
    if (!canvas) return;

    ctx = canvas.getContext("2d");
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    canvas.addEventListener("pointerdown", (e) => {
      drawing = true;
      ctx.beginPath();
      const p = getPos(canvas, e);
      ctx.moveTo(p.x, p.y);
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!drawing) return;
      const p = getPos(canvas, e);
      ctx.strokeStyle = brushColor;
      ctx.lineWidth = brushSize;
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    });

    const end = () => {
      if (!drawing) return;
      drawing = false;
      saveState(canvas);
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointerleave", end);

    saveState(canvas);
    setStatus("Ready.");
  }

  function getPos(canvas, e) {
    const r = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (canvas.width / r.width),
      y: (e.clientY - r.top) * (canvas.height / r.height),
    };
  }

  function saveState(canvas) {
    try {
      history.push(canvas.toDataURL("image/png"));
      if (history.length > 30) history.shift();
    } catch (_) {}
  }

  function setBrush(v) {
    brushSize = Number(v) || brushSize;
    setStatus(`Brush: ${brushSize}px`);
  }

  function setColor(v) {
    brushColor = v || brushColor;
    setStatus(`Color: ${brushColor}`);
  }

  function clear() {
    const canvas = document.getElementById("main-canvas");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    saveState(canvas);
    setStatus("Cleared.");
  }

  async function undo() {
    const canvas = document.getElementById("main-canvas");
    if (!canvas || !ctx) return;

    if (history.length <= 1) return setStatus("Nothing to undo.");
    history.pop();

    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      setStatus("Undo.");
    };
    img.src = history[history.length - 1];
  }

  function exportPng() {
    const canvas = document.getElementById("main-canvas");
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = "canvas.png";
    a.click();
  }

  function setStatus(msg) {
    const el = document.getElementById("canvas-status");
    if (el) el.textContent = msg;
  }

  return { onViewLoaded, setBrush, setColor, clear, undo, exportPng };
})();
