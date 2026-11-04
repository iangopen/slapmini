//! Parameters. Every physics constant lives in `params.json` with a provenance tag.
//! `Params` is a flat struct of f64 so the fitter can address any field by name.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

macro_rules! define_params {
    ($($name:ident),* $(,)?) => {
        /// All sim constants. Field names match the keys in `params.json`.
        #[derive(Clone, Debug, PartialEq)]
        pub struct Params { $(pub $name: f64,)* }

        impl Params {
            /// Every parameter name, in declaration order.
            pub const NAMES: &'static [&'static str] = &[$(stringify!($name)),*];

            pub fn get(&self, name: &str) -> Option<f64> {
                match name { $(stringify!($name) => Some(self.$name),)* _ => None }
            }

            /// Returns false if `name` is not a parameter.
            pub fn set(&mut self, name: &str, value: f64) -> bool {
                match name { $(stringify!($name) => { self.$name = value; true })* _ => false }
            }

            fn zeroed() -> Self { Params { $($name: 0.0,)* } }
        }
    };
}

define_params!(
    timestep, substeps, gravity,
    rink_length, rink_width, rink_corner_radius, board_restitution, board_friction,
    net_x, net_width, net_depth, net_height, net_restitution,
    puck_radius, puck_mass, ice_friction_decel, ice_linear_drag, air_drag, ice_restitution,
    player_radius, player_height, player_mass, player_accel, player_max_speed, player_ice_drag,
    player_turn_rate, player_restitution,
    stick_inner, stick_length, stick_thickness, stick_mass, stick_max_ang_speed,
    stick_max_ang_accel, stick_restitution, stick_z_min, stick_z_max,
    max_puck_speed_air, max_puck_speed_ice_normal, max_puck_speed_ice_dash,
    dash_speed, dash_duration, dash_cooldown, lift_vz_factor, lift_max_vz, chip_max_speed,
    blue_line_x, faceoff_x, faceoff_y, faceoff_circle_radius, speed_display_scale,
    // Match rules (sim/src/rules.rs). Only read when State::match_state is Some.
    match_length_s, mercy_lead, countdown_s, goal_pause_s, player_spawn_x,
);

/// One entry of `params.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ParamEntry {
    pub value: f64,
    /// "placeholder" (invented), "fitted" (from calib), "measured"/"measured-shape", or "rule" (a match rule Ian set).
    pub provenance: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub unit: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

/// The on-disk params file.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ParamFile {
    #[serde(rename = "_comment", default, skip_serializing_if = "String::is_empty")]
    pub comment: String,
    pub params: BTreeMap<String, ParamEntry>,
}

/// The repo's `params.json`, embedded at compile time (no file I/O in sim).
pub const DEFAULT_PARAMS_JSON: &str = include_str!("../../params.json");

impl ParamFile {
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("params json: {e}"))
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("ParamFile serializes")
    }

    /// Strict conversion: every name must be present, no unknown names.
    pub fn to_params(&self) -> Result<Params, String> {
        let mut p = Params::zeroed();
        for name in Params::NAMES {
            let e = self.params.get(*name).ok_or_else(|| format!("params: missing '{name}'"))?;
            p.set(name, e.value);
        }
        if let Some(k) = self.params.keys().find(|k| p.get(k).is_none()) {
            return Err(format!("params: unknown key '{k}'"));
        }
        Ok(p)
    }

    /// Write values from `p` back into the entries, marking `fitted` names as provenance "fitted".
    pub fn update_from(&mut self, p: &Params, fitted: &[&str]) {
        for (k, e) in self.params.iter_mut() {
            if let Some(v) = p.get(k) {
                e.value = v;
            }
            if fitted.contains(&k.as_str()) {
                e.provenance = "fitted".into();
            }
        }
    }
}

impl Params {
    pub fn from_json(s: &str) -> Result<Self, String> {
        ParamFile::from_json(s)?.to_params()
    }

    pub fn substeps_n(&self) -> u32 {
        let n = libm::round(self.substeps);
        if n < 1.0 { 1 } else { n as u32 }
    }
}

impl Default for Params {
    fn default() -> Self {
        Params::from_json(DEFAULT_PARAMS_JSON).expect("embedded params.json is valid")
    }
}
