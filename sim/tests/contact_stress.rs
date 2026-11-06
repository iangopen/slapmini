//! w5a contact guarantees: 100,000 steps of random swings, dashes, raised and lowered sticks, with the puck at rest,
//! moving, and against boards and nets. Run with --nocapture for the numbers.

use sim::{step_with_stats, stick_puck_gap, stick_reaches, ContactStats, Params, PlayerInput, State};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
}

/// Puck placements that cycle through the hard cases; players are put right next to it.
fn scenario(s: &mut State, p: &Params, r: &mut Rng, kind: u64) {
    let hx = p.rink_length * 0.5 - p.puck_radius - 0.02;
    let hy = p.rink_width * 0.5 - p.puck_radius - 0.02;
    let (x, y) = match kind % 5 {
        0 => (r.range(-15.0, 15.0), r.range(-10.0, 10.0)),            // open ice
        1 => (r.range(-15.0, 15.0), if r.f() < 0.5 { hy } else { -hy }), // side board
        2 => (if r.f() < 0.5 { hx - 9.0 } else { 9.0 - hx }, r.range(-5.0, 5.0)), // beside the nets
        3 => (p.net_x - 0.6, p.net_width * 0.5 + p.puck_radius + 0.01), // on a post
        _ => (r.range(-10.0, 10.0), r.range(-10.0, 10.0)),
    };
    s.puck.x = x;
    s.puck.y = y;
    s.puck.z = 0.0;
    s.puck.vz = 0.0;
    let v = if kind % 2 == 0 { 0.0 } else { r.range(0.0, 30.0) };
    let a = r.range(-3.14, 3.14);
    s.puck.vx = v * a.cos();
    s.puck.vy = v * a.sin();
    s.puck.in_net = None;
    for pl in s.players.iter_mut() {
        let a = r.range(-3.14, 3.14);
        let d = r.range(1.0, 2.5);
        pl.x = (x + d * a.cos()).clamp(-p.rink_length * 0.5 + 1.0, p.rink_length * 0.5 - 1.0);
        pl.y = (y + d * a.sin()).clamp(-p.rink_width * 0.5 + 1.0, p.rink_width * 0.5 - 1.0);
        // Clamped against the boards it could land on the puck: go toward centre ice instead (a test reset, not physics).
        if (pl.x - x).hypot(pl.y - y) < 1.0 {
            let l = x.hypot(y).max(1e-9);
            pl.x = x - x / l * d;
            pl.y = y - y / l * d;
        }
        pl.vx = 0.0;
        pl.vy = 0.0;
        pl.stick_angle = r.range(-3.14, 3.14);
        pl.stick_omega = 0.0;
    }
}

#[test]
fn contact_stress() {
    let p = Params::default();
    let mut s = State::new_default(&p, 2);
    let mut r = Rng(0x5EED);
    let mut stats = ContactStats::default();
    let mut inputs = vec![PlayerInput::default(); 2];
    let (mut max_pen, mut end_overlaps, mut contact_steps, mut hits) = (0.0f64, 0u64, 0u64, 0u64);
    let (mut blocks, mut body_blocks, mut outside, mut raised) = (0u64, 0u64, 0u64, 0u64);
    let (mut ghosts, mut ghost_steps, mut was_ghost) = (0u64, 0u64, [false; 2]);
    let mut run = [0u32; 2];
    let (mut longest, mut longest_pinned, mut pinned, mut settle_steps) = (0u32, 0u32, [false; 2], 0u64);
    let mut hist = [0u64; 12];
    for t in 0..100_000u64 {
        if t % 240 == 0 {
            scenario(&mut s, &p, &mut r, t / 240);
        }
        if t % 12 == 0 {
            for (k, inp) in inputs.iter_mut().enumerate() {
                let pl = &s.players[k];
                let chase = r.f() < 0.7;
                let (dx, dy) = (s.puck.x - pl.x, s.puck.y - pl.y);
                let d = (dx * dx + dy * dy).sqrt().max(1e-9);
                inp.move_x = if chase { dx / d } else { r.range(-1.0, 1.0) };
                inp.move_y = if chase { dy / d } else { r.range(-1.0, 1.0) };
                inp.stick_target_angle = r.range(-3.14, 3.14);
                inp.dash = r.f() < 0.3;
                inp.stick_up = r.f() < 0.25;
                inp.lift = r.f() < 0.1;
            }
        }
        step_with_stats(&mut s, &inputs, &p, &mut stats);
        max_pen = max_pen.max(stats.max_penetration);
        blocks += stats.stick_blocks as u64;
        body_blocks += stats.body_blocks as u64;
        for (k, pl) in s.players.iter().enumerate() {
            raised += pl.stick_up as u64;
            ghost_steps += pl.stick_ghost as u64;
            if pl.stick_ghost && !was_ghost[k] {
                ghosts += 1;
            }
            was_ghost[k] = pl.stick_ghost;
            if stick_reaches(pl, &s.puck, &p) && stick_puck_gap(pl, &s.puck, &p) < -1e-9 {
                end_overlaps += 1;
            }
            let d = ((s.puck.x - pl.x).powi(2) + (s.puck.y - pl.y).powi(2)).sqrt();
            if s.puck.z <= p.player_height && d < p.player_radius + p.puck_radius - 1e-9 {
                // Allowed only while the puck slides out of a body it fell into.
                if stats.body_settling > 0 { settle_steps += 1 } else { end_overlaps += 1 }
            }
            if stats.stick_contact[k] {
                contact_steps += 1;
                if run[k] == 0 {
                    hits += 1;
                }
                run[k] += 1;
                // A run is pinned if the puck was against a board or net or a pinch fired at any point in it.
                let (ex, ey) = (s.puck.x.abs(), s.puck.y.abs());
                let near = ey > p.rink_width * 0.5 - p.puck_radius - 0.1 || ex > p.rink_length * 0.5 - p.puck_radius - 0.1 || (ex > p.net_x - 1.0 && ey < p.net_width * 0.5 + 1.0);
                pinned[k] |= near || stats.stick_blocks > 0 || stats.body_blocks > 0;
                if pinned[k] { longest_pinned = longest_pinned.max(run[k]) } else { longest = longest.max(run[k]) }
            } else {
                if run[k] > 0 {
                    hist[(run[k] as usize).min(11)] += 1;
                }
                run[k] = 0;
                pinned[k] = false;
            }
        }
        let (hx, hy) = (p.rink_length * 0.5, p.rink_width * 0.5);
        if s.puck.x.abs() > hx || s.puck.y.abs() > hy {
            outside += 1;
        }
    }
    println!("contact stress, 100000 steps: {hits} contacts, {contact_steps} contact steps, raised stick-steps {raised}");
    println!("max penetration before resolve {max_pen:.4} m; end-of-step overlaps {end_overlaps}; longest contact run {longest} steps in open ice, {longest_pinned} against a board, net or pinch; body settling {settle_steps} player-steps; run length histogram {hist:?}");
    println!("stick placed on the puck without a swing (ghost, no push): {ghosts} cases, {ghost_steps} player-steps");
    println!("sticks stopped by a pinch {blocks}, players pushed off {body_blocks}, puck outside the rink {outside}");
    assert_eq!(end_overlaps, 0);
    assert_eq!(outside, 0);
    assert!(hits > 500, "the run must actually hit the puck");
    // A stick starting next to the puck pushes it on every step of its spin-up: stick_max_ang_speed / accel = 6.1 steps.
    // Those runs end with the puck leaving at full speed. The margin covers a spin-up that begins after a pinch.
    assert!(longest <= 7, "puck stuck to the blade: {longest} consecutive contact steps in open ice");
    // Against a board or net the stick can lean on a pinned puck: the rebounds break it up, but not at once.
    assert!(longest_pinned <= 20, "{longest_pinned} consecutive pinned contact steps");
    // Sticks and bodies are swept; what is left is the rest of a substep after a contact, and a second surface a pinch adds.
    assert!(max_pen <= 0.2, "max penetration {max_pen}");
}

/// The hardest swing at a resting puck cannot pass through at any phase, including the fastest head speed.
#[test]
fn hard_swing_never_tunnels() {
    let p = Params::default();
    for k in 0..400 {
        for radius in [0.9, 1.0, 1.1] {
            let mut s = State::new_default(&p, 1);
            let dir = k as f64 * 0.0157;
            s.players[0].x = 0.0;
            s.players[0].y = 0.0;
            s.puck.x = radius * dir.cos();
            s.puck.y = radius * dir.sin();
            s.players[0].stick_angle = dir + 1.4;
            s.players[0].stick_omega = -p.stick_max_ang_speed;
            let inp = [PlayerInput { stick_target_angle: dir - 1.2, ..Default::default() }];
            let mut st = ContactStats::default();
            for _ in 0..16 {
                step_with_stats(&mut s, &inp, &p, &mut st);
            }
            assert!(s.puck.vx.hypot(s.puck.vy) > 5.0, "swing at dir {dir} r {radius} missed the puck");
        }
    }
}

/// Puck pinched between a blade and the board: it stays in the rink, out of the blade, and the stick stops.
#[test]
fn pinch_against_board() {
    let p = Params::default();
    let hy = p.rink_width * 0.5;
    let mut s = State::new_default(&p, 1);
    s.puck.x = 0.0;
    s.puck.y = hy - p.puck_radius - 0.001;
    s.players[0].x = 0.0;
    s.players[0].y = hy - 1.4;
    s.players[0].stick_angle = 0.0;
    let inp = [PlayerInput { stick_target_angle: 1.57, move_y: 1.0, ..Default::default() }];
    let mut st = ContactStats::default();
    for _ in 0..240 {
        step_with_stats(&mut s, &inp, &p, &mut st);
        assert!(s.puck.y <= hy - p.puck_radius + 1e-6);
        assert!(stick_puck_gap(&s.players[0], &s.puck, &p) >= -1e-9);
        let d = (s.puck.x - s.players[0].x).hypot(s.puck.y - s.players[0].y);
        assert!(d >= p.player_radius + p.puck_radius - 1e-9);
    }
}

/// Printout for the session log: `cargo test -p sim --release --test contact_stress swing_printout -- --nocapture`.
#[test]
fn swing_printout() {
    let p = Params::default();
    let mut s = State::new_default(&p, 1);
    s.players[0].x = 0.0;
    s.players[0].y = 0.0;
    s.puck.x = 1.0;
    s.puck.y = 0.0;
    s.players[0].stick_angle = 1.4;
    s.players[0].stick_omega = -p.stick_max_ang_speed;
    let inp = [PlayerInput { stick_target_angle: -1.2, ..Default::default() }];
    let mut st = ContactStats::default();
    println!("step angle   omega   puck_x  puck_y   speed   gap     contact");
    for t in 0..16 {
        step_with_stats(&mut s, &inp, &p, &mut st);
        let g = stick_puck_gap(&s.players[0], &s.puck, &p);
        println!("{t:3} {:6.3} {:7.2} {:7.3} {:7.3} {:7.2} {:7.3} {}", s.players[0].stick_angle, s.players[0].stick_omega, s.puck.x, s.puck.y, s.puck.vx.hypot(s.puck.vy), g, st.stick_contact[0]);
    }
}

/// A stick lowered onto a resting puck does not knock it; it collides again once they have separated.
#[test]
fn lowered_stick_does_not_teleport() {
    let p = Params::default();
    let mut s = State::new_default(&p, 1);
    s.players[0].x = 0.0;
    s.players[0].y = 0.0;
    s.players[0].stick_angle = 0.0;
    s.puck.x = 0.95;
    s.puck.y = 0.0;
    let mut st = ContactStats::default();
    let up = [PlayerInput { stick_target_angle: 0.0, stick_up: true, ..Default::default() }];
    let down = [PlayerInput { stick_target_angle: 0.0, ..Default::default() }];
    for _ in 0..5 { step_with_stats(&mut s, &up, &p, &mut st); }
    for _ in 0..30 {
        step_with_stats(&mut s, &down, &p, &mut st);
        assert_eq!((s.puck.x, s.puck.y), (0.95, 0.0), "the puck must not move");
        assert!(s.players[0].stick_ghost);
    }
    // Skate away: the flag clears at separation.
    let away = [PlayerInput { stick_target_angle: 0.0, move_x: -1.0, ..Default::default() }];
    for _ in 0..60 { step_with_stats(&mut s, &away, &p, &mut st); }
    assert!(!s.players[0].stick_ghost);
}
