// Pure drawing: reads a State + params, never steps the sim.

import type { P, Sample, State } from "./types";

/**
 * s = pixels per meter, (cx, cy) = screen position of center ice.
 * rot = false: x right, y up (lab default). rot = true: "game view", the real game's camera: +x up
 * (toward the net at the top), +y left, i.e. the default view rotated 90 deg counterclockwise. Both keep
 * x then y counterclockwise on screen, which is the measure/ coordinate convention.
 */
export interface View { s: number; cx: number; cy: number; rot: boolean }

export function fitView(canvasW: number, canvasH: number, p: P, rot = false, marginPx = 24): View {
  const [along, across] = rot ? [canvasH, canvasW] : [canvasW, canvasH];
  const s = Math.min((along - 2 * marginPx) / p.rink_length, (across - 2 * marginPx) / p.rink_width);
  return { s, cx: canvasW / 2, cy: canvasH / 2, rot };
}

export const toScreen = (v: View, x: number, y: number): [number, number] =>
  v.rot ? [v.cx - y * v.s, v.cy - x * v.s] : [v.cx + x * v.s, v.cy - y * v.s];
export const toWorld = (v: View, sx: number, sy: number): [number, number] =>
  v.rot ? [(v.cy - sy) / v.s, (v.cx - sx) / v.s] : [(sx - v.cx) / v.s, -(sy - v.cy) / v.s];

/** Minimum on-screen puck radius in pixels (display only; physics radius is unchanged). */
const PUCK_MIN_PX = 6;

const TEAM = ["#4aa3ff", "#ff7a5c"];

/** Cool (blue) on the ice → warm (red) at 2 m and above. */
export function puckColor(z: number): string {
  const t = Math.max(0, Math.min(1, z / 2));
  const hue = 200 - 190 * t;
  return `hsl(${hue}, 85%, ${55 + 5 * t}%)`;
}

function rinkPath(ctx: CanvasRenderingContext2D, v: View, p: P): void {
  const [ax, ay] = toScreen(v, -p.rink_length / 2, p.rink_width / 2);
  const [bx, by] = toScreen(v, p.rink_length / 2, -p.rink_width / 2);
  const x0 = Math.min(ax, bx);
  const y0 = Math.min(ay, by);
  const w = Math.abs(bx - ax);
  const h = Math.abs(by - ay);
  const r = Math.min(p.rink_corner_radius * v.s, w / 2, h / 2);
  ctx.beginPath();
  ctx.roundRect(x0, y0, w, h, r);
}

export interface Overlay {
  targets: number[]; // stick target angle per player
  spawn: { x: number; y: number; z: number } | null;
  target: { x: number; y: number } | null;
  ghost: Sample[] | null;
  ghostIndex: number;
  forehand: { x: number; y: number } | null;
}

export function draw(ctx: CanvasRenderingContext2D, v: View, p: P, st: State, o: Overlay): void {
  const { width, height } = ctx.canvas;
  ctx.fillStyle = "#0f1419";
  ctx.fillRect(0, 0, width, height);
  const line = (x0: number, y0: number, x1: number, y1: number) => {
    const [a, b] = toScreen(v, x0, y0);
    const [c, d] = toScreen(v, x1, y1);
    ctx.beginPath();
    ctx.moveTo(a, b);
    ctx.lineTo(c, d);
    ctx.stroke();
  };
  const circle = (x: number, y: number, r: number) => {
    const [a, b] = toScreen(v, x, y);
    ctx.beginPath();
    ctx.arc(a, b, Math.max(0.5, r * v.s), 0, Math.PI * 2);
  };

  // Ice + markings, clipped to the rink outline.
  ctx.save();
  rinkPath(ctx, v, p);
  ctx.fillStyle = "#e9eff4";
  ctx.fill();
  ctx.clip();
  const hw = p.rink_width / 2;
  ctx.lineWidth = Math.max(2, 0.3 * v.s);
  ctx.strokeStyle = "#d44a4a";
  line(0, -hw, 0, hw);
  ctx.lineWidth = Math.max(1, 0.1 * v.s);
  line(p.net_x, -hw, p.net_x, hw);
  line(-p.net_x, -hw, -p.net_x, hw);
  ctx.lineWidth = Math.max(2, 0.3 * v.s);
  ctx.strokeStyle = "#3d6fd1";
  line(p.blue_line_x, -hw, p.blue_line_x, hw);
  line(-p.blue_line_x, -hw, -p.blue_line_x, hw);
  ctx.lineWidth = Math.max(1, 0.08 * v.s);
  ctx.strokeStyle = "#3d6fd1";
  circle(0, 0, p.faceoff_circle_radius);
  ctx.stroke();
  ctx.strokeStyle = "#d44a4a";
  ctx.fillStyle = "#d44a4a";
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    circle(sx * p.faceoff_x, sy * p.faceoff_y, p.faceoff_circle_radius);
    ctx.stroke();
    circle(sx * p.faceoff_x, sy * p.faceoff_y, 0.3);
    ctx.fill();
  }
  ctx.restore();

  // Boards.
  rinkPath(ctx, v, p);
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#8a96a3";
  ctx.stroke();

  // Nets (sides + back), mouth facing center.
  ctx.lineWidth = Math.max(2, 0.12 * v.s);
  ctx.strokeStyle = "#c33";
  for (const sign of [1, -1]) {
    const x0 = sign * p.net_x;
    const x1 = sign * (p.net_x + p.net_depth);
    const w = p.net_width / 2;
    const [a, b] = toScreen(v, x0, w);
    const [c, d] = toScreen(v, x1, w);
    const [e, f] = toScreen(v, x1, -w);
    const [g, h] = toScreen(v, x0, -w);
    ctx.fillStyle = "rgba(200,40,40,0.12)";
    ctx.beginPath();
    ctx.moveTo(a, b);
    ctx.lineTo(c, d);
    ctx.lineTo(e, f);
    ctx.lineTo(g, h);
    ctx.fill();
    ctx.stroke();
  }

  // Ghost path (faded) + scrub marker.
  if (o.ghost && o.ghost.length > 1) {
    ctx.strokeStyle = "rgba(140, 60, 200, 0.5)";
    ctx.lineWidth = Math.max(2, p.puck_radius * v.s);
    ctx.lineJoin = "round";
    // Break the path where samples are missing for a while, or where the puck would have to move
    // faster than 3x the largest speed cap (a measured track's teleports / practice-tool resets).
    const dts = o.ghost.slice(1).map((g, i) => g.t - o.ghost![i].t).sort((a, b) => a - b);
    const gapT = 3 * (dts[Math.floor(dts.length / 2)] || 0);
    const vMax = 3 * Math.max(p.max_puck_speed_ice_dash, p.max_puck_speed_ice_normal, p.max_puck_speed_air);
    ctx.beginPath();
    o.ghost.forEach((g, i) => {
      const [a, b] = toScreen(v, g.x, g.y);
      const q = o.ghost![i - 1];
      const dt = q ? g.t - q.t : 0;
      const jump = q ? Math.hypot(g.x - q.x, g.y - q.y, g.z - q.z) > vMax * Math.max(dt, 1e-9) : false;
      if (i === 0 || dt > gapT || jump) ctx.moveTo(a, b);
      else ctx.lineTo(a, b);
    });
    ctx.stroke();
    const g = o.ghost[Math.min(o.ghostIndex, o.ghost.length - 1)];
    circle(g.x, g.y, Math.max(p.puck_radius, PUCK_MIN_PX / v.s) * 1.2);
    ctx.fillStyle = "rgba(140, 60, 200, 0.45)";
    ctx.fill();
  }

  // Practice markers.
  if (o.spawn) {
    const [a, b] = toScreen(v, o.spawn.x, o.spawn.y);
    ctx.strokeStyle = "#2b8a3e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a - 10, b - 10);
    ctx.lineTo(a - 5, b - 5);
    ctx.moveTo(a + 5, b + 5);
    ctx.lineTo(a + 10, b + 10);
    ctx.moveTo(a + 10, b - 10);
    ctx.lineTo(a + 5, b - 5);
    ctx.moveTo(a - 5, b + 5);
    ctx.lineTo(a - 10, b + 10);
    ctx.stroke();
    if (o.spawn.z > 0) {
      ctx.fillStyle = "#2b8a3e";
      ctx.font = "11px system-ui";
      ctx.fillText(`spawn z ${o.spawn.z.toFixed(2)} m`, a + 12, b - 12);
    }
  }
  if (o.target) {
    ctx.strokeStyle = "#e0a84a";
    ctx.lineWidth = 2;
    circle(o.target.x, o.target.y, 0.6);
    ctx.stroke();
    circle(o.target.x, o.target.y, 0.15);
    ctx.stroke();
  }
  if (o.forehand) {
    circle(o.forehand.x, o.forehand.y, 0.08);
    ctx.fillStyle = "rgba(43,138,62,0.5)";
    ctx.fill();
  }

  // Players and sticks.
  st.players.forEach((pl, i) => {
    const col = TEAM[pl.team % TEAM.length];
    circle(pl.x, pl.y, p.player_radius);
    ctx.fillStyle = col;
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    // facing tick
    ctx.strokeStyle = "#102030";
    ctx.lineWidth = 2;
    line(pl.x, pl.y, pl.x + Math.cos(pl.facing) * p.player_radius, pl.y + Math.sin(pl.facing) * p.player_radius);
    // target stick (faint) to show lag
    const ta = o.targets[i];
    if (ta !== undefined) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(20,30,40,0.3)";
      ctx.lineWidth = 2;
      line(pl.x, pl.y, pl.x + Math.cos(ta) * p.stick_length, pl.y + Math.sin(ta) * p.stick_length);
      ctx.setLineDash([]);
    }
    // actual stick
    ctx.strokeStyle = "#1b232c";
    ctx.lineWidth = Math.max(3, p.stick_thickness * 2 * v.s);
    ctx.lineCap = "round";
    const a = pl.stick_angle;
    line(pl.x + Math.cos(a) * p.stick_inner, pl.y + Math.sin(a) * p.stick_inner, pl.x + Math.cos(a) * p.stick_length, pl.y + Math.sin(a) * p.stick_length);
    ctx.lineCap = "butt";
  });

  // Puck: shadow grows and offsets with height; color goes cool → warm.
  // Display radius has a pixel floor so the puck stays readable at minimap scale.
  const pk = st.puck;
  const z = Math.max(0, pk.z);
  const off = z * 0.6;
  const rDisp = Math.max(p.puck_radius, PUCK_MIN_PX / v.s);
  circle(pk.x + off, pk.y - off, rDisp * (1 + 0.6 * z));
  ctx.fillStyle = `rgba(0,0,0,${0.45 / (1 + z)})`;
  ctx.fill();
  circle(pk.x, pk.y, rDisp * (1 + 0.08 * z));
  ctx.fillStyle = puckColor(z);
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#0b0f13";
  ctx.stroke();
}
