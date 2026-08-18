import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
type Channel = "chat" | "email" | "notes";

type Result = {
  interaction_id: string;
  raw_text: string;
  polished_text: string;
  rewrite_mode: string;
  total_milliseconds: number;
  timings: { name: string; milliseconds: number }[];
};

export function App() {
  const [channel, setChannel] = useState<Channel>("email");
  const [raw, setRaw] = useState("um send the api report to wisper tomorrow morning");
  const [result, setResult] = useState<Result | null>(null);
  const [finalText, setFinalText] = useState("");
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("Ready");
  const recorder = useRef<MediaRecorder | null>(null);
  const socket = useRef<WebSocket | null>(null);

  useEffect(() => () => socket.current?.close(), []);

  async function polish() {
    setStatus("Polishing…");
    const response = await fetch(`${API}/v1/polish`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ user_id: "demo", transcript: raw, channel }),
    });
    const payload = (await response.json()) as Result;
    setResult(payload);
    setFinalText(payload.polished_text);
    setStatus("Ready for feedback");
  }

  async function sendFeedback(accepted: boolean) {
    if (!result) return;
    await fetch(`${API}/v1/feedback`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        user_id: "demo",
        interaction_id: result.interaction_id,
        raw_text: result.raw_text,
        suggested_text: result.polished_text,
        final_text: finalText,
        accepted,
        channel,
      }),
    });
    setStatus("Preference learned locally");
  }

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ws = new WebSocket(`${API.replace("http", "ws")}/v1/stream/demo`);
    socket.current = ws;
    await new Promise<void>((resolve) => ws.addEventListener("open", () => resolve(), { once: true }));
    ws.send(`channel:${channel}`);
    ws.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data) as Result;
      setResult(payload);
      setRaw(payload.raw_text);
      setFinalText(payload.polished_text);
      setStatus("Ready for feedback");
    });
    const mediaRecorder = new MediaRecorder(stream);
    recorder.current = mediaRecorder;
    mediaRecorder.addEventListener("dataavailable", async (event) => {
      if (event.data.size && ws.readyState === WebSocket.OPEN) ws.send(await event.data.arrayBuffer());
    });
    mediaRecorder.start(200);
    setRecording(true);
    setStatus("Listening…");
  }

  function stopRecording() {
    recorder.current?.stop();
    recorder.current?.stream.getTracks().forEach((track) => track.stop());
    setTimeout(() => socket.current?.send("commit"), 250);
    setRecording(false);
    setStatus("Transcribing…");
  }

  return (
    <main>
      <header>
        <div className="mark">VA</div>
        <div>
          <p className="eyebrow">LOCAL VOICE INTELLIGENCE</p>
          <h1>Say it roughly. Send it like you.</h1>
          <p className="subtitle">
            VoxAdapt measures every stage, learns from your corrections, and keeps your profile local.
          </p>
        </div>
      </header>

      <section className="workspace">
        <div className="controls">
          <label>
            Destination
            <select value={channel} onChange={(event) => setChannel(event.target.value as Channel)}>
              <option value="chat">Chat</option>
              <option value="email">Email</option>
              <option value="notes">Notes</option>
            </select>
          </label>
          <button className={recording ? "record active" : "record"} onClick={recording ? stopRecording : startRecording}>
            <span /> {recording ? "Stop & polish" : "Record"}
          </button>
        </div>

        <label className="editor">
          Rough transcript
          <textarea value={raw} onChange={(event) => setRaw(event.target.value)} rows={5} />
        </label>
        <button className="primary" onClick={polish}>Polish transcript</button>

        {result && (
          <div className="result">
            <div className="result-heading">
              <span>Personalized result</span>
              <span className="mode">{result.rewrite_mode}</span>
            </div>
            <textarea value={finalText} onChange={(event) => setFinalText(event.target.value)} rows={5} />
            <div className="feedback">
              <button onClick={() => sendFeedback(finalText === result.polished_text)}>Accept</button>
              <button onClick={() => sendFeedback(false)}>Save edit</button>
            </div>
            <div className="timings">
              {result.timings.map((timing) => (
                <span key={timing.name}><b>{timing.name}</b> {timing.milliseconds.toFixed(2)} ms</span>
              ))}
              <span><b>total</b> {result.total_milliseconds.toFixed(2)} ms</span>
            </div>
          </div>
        )}
      </section>
      <footer><span className="pulse" /> {status}</footer>
    </main>
  );
}
