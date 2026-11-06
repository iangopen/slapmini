// A port for a verify script's own dev server: VERIFY_PORT if set, else one the system says is free.
// Other dev servers on the machine hold 5173 and neighbours, so no fixed default.
import { createServer } from "node:net";

export function freePort() {
  if (process.env.VERIFY_PORT) return Promise.resolve(Number(process.env.VERIFY_PORT));
  return new Promise((res, rej) => {
    const s = createServer();
    s.on("error", rej);
    s.listen(0, () => {
      const { port } = s.address();
      s.close(() => res(port));
    });
  });
}
