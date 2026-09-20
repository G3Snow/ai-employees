/* Chainlit stores uploads on the live websocket session. The paperclip can
   fire POST /project/file before that session exists (cold start, reconnect,
   or attaching the moment the page loads), which the server reports as
   "Session not found". Wait for the socket, then send. */
(function () {
  let socketOpenAt = 0;

  function socketReady() {
    return socketOpenAt > 0 && Date.now() - socketOpenAt >= 400;
  }

  const NativeWebSocket = window.WebSocket;
  class TrackedWebSocket extends NativeWebSocket {
    constructor(url, protocols) {
      super(url, protocols);
      if (String(url).includes("socket.io")) {
        this.addEventListener("open", function () {
          socketOpenAt = Date.now();
        });
        this.addEventListener("close", function () {
          socketOpenAt = 0;
        });
      }
    }
  }
  window.WebSocket = TrackedWebSocket;

  const xhrOpen = XMLHttpRequest.prototype.open;
  const xhrSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (method, url) {
    this.__aiTeamUpload =
      String(method || "").toUpperCase() === "POST" &&
      String(url || "").includes("/project/file");
    return xhrOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function () {
    if (!this.__aiTeamUpload || socketReady()) {
      return xhrSend.apply(this, arguments);
    }
    const xhr = this;
    const args = arguments;
    const started = Date.now();
    const wait = function () {
      if (socketReady() || Date.now() - started > 20000) {
        xhrSend.apply(xhr, args);
        return;
      }
      setTimeout(wait, 200);
    };
    wait();
  };
})();
