#![forbid(unsafe_code)]
//! Evaluate Quill QG artifacts against a committed pass-over-pass baseline.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use frankensearch_quill_gauntlet::{
    PerfEvidenceFile, PerfGate, PerfGateArtifact, PerfGateDecision, PerfRatchetMode,
    PerfRatchetRequest, evaluate_perf_ratchet,
};
use sha2::{Digest, Sha256};

const USAGE: &str = "\
Usage:
  quill-perf-ratchet \\
    --manifest <docs/contracts/quill-perf-gates.toml> \\
    --baseline <.bench-history/QG-N.machine.latest.json> \\
    --candidate <QG-N.json> \\
    [--rerun <QG-N.json>] \\
    --output <ratchet.json> \\
    --mode <promotion|regression-alarm> \\
    [--promote-dir <.bench-history> --machine-class <label> --date <YYYY-MM-DD>]

Exit status: 0=Allow, 1=Block, 2=Quarantine, 64=invalid invocation.";

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    baseline: PathBuf,
    candidate: PathBuf,
    rerun: Option<PathBuf>,
    output: PathBuf,
    mode: PerfRatchetMode,
    promote_dir: Option<PathBuf>,
    machine_class: Option<String>,
    date: Option<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(decision) => match decision {
            PerfGateDecision::Allow => ExitCode::SUCCESS,
            PerfGateDecision::Block => ExitCode::from(1),
            PerfGateDecision::Quarantine => ExitCode::from(2),
        },
        Err(error) => {
            eprintln!("quill-perf-ratchet: {error}");
            eprintln!("{USAGE}");
            ExitCode::from(64)
        }
    }
}

fn run() -> Result<PerfGateDecision, Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let manifest_bytes = fs::read(&args.manifest)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let manifest_text = std::str::from_utf8(&manifest_bytes)?;
    let manifest = toml::from_str::<toml::Value>(manifest_text)?;

    let (baseline, baseline_bytes) = read_artifact(&args.baseline)?;
    let (candidate, candidate_bytes) = read_artifact(&args.candidate)?;
    let rerun = args.rerun.as_deref().map(read_artifact).transpose()?;
    let activated = gate_activated(&manifest, candidate.gate)?;

    let mut evidence_files = vec![
        evidence("manifest", &args.manifest, &manifest_bytes),
        evidence("baseline", &args.baseline, &baseline_bytes),
        evidence("candidate", &args.candidate, &candidate_bytes),
    ];
    if let (Some(rerun_path), Some((_, rerun_bytes))) = (args.rerun.as_deref(), rerun.as_ref()) {
        evidence_files.push(evidence("rerun", rerun_path, rerun_bytes));
    }

    let mut evaluation = evaluate_perf_ratchet(PerfRatchetRequest {
        baseline: Some(&baseline),
        candidate: &candidate,
        rerun: rerun.as_ref().map(|(artifact, _)| artifact),
        gate_activated: activated,
        mode: args.mode,
        expected_manifest_sha256: &manifest_sha256,
        evidence: evidence_files,
    });

    if evaluation.decision == PerfGateDecision::Allow {
        promote_history_if_requested(
            &args,
            candidate.gate,
            &candidate_bytes,
            &mut evaluation.history_updates,
        )?;
    }

    let output = serde_json::to_string_pretty(&evaluation)?;
    write_file(&args.output, format!("{output}\n").as_bytes())?;
    println!(
        "{} {}: {} (evidence {})",
        evaluation.gate,
        mode_label(evaluation.mode),
        evaluation.decision,
        args.output.display()
    );
    for reason in &evaluation.reasons {
        println!("{}: {}", reason.code, reason.message);
    }
    Ok(evaluation.decision)
}

fn parse_args<I>(mut values: I) -> Result<Args, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let mut manifest = None;
    let mut baseline = None;
    let mut candidate = None;
    let mut rerun = None;
    let mut output = None;
    let mut mode = None;
    let mut promote_dir = None;
    let mut machine_class = None;
    let mut date = None;

    while let Some(flag) = values.next() {
        match flag.to_string_lossy().as_ref() {
            "-h" | "--help" => return Err(USAGE.into()),
            "--manifest" => manifest = Some(PathBuf::from(next_value(&mut values, "--manifest")?)),
            "--baseline" => baseline = Some(PathBuf::from(next_value(&mut values, "--baseline")?)),
            "--candidate" => {
                candidate = Some(PathBuf::from(next_value(&mut values, "--candidate")?));
            }
            "--rerun" => rerun = Some(PathBuf::from(next_value(&mut values, "--rerun")?)),
            "--output" => output = Some(PathBuf::from(next_value(&mut values, "--output")?)),
            "--mode" => {
                let value = next_value(&mut values, "--mode")?;
                mode = Some(match value.to_string_lossy().as_ref() {
                    "promotion" => PerfRatchetMode::Promotion,
                    "regression-alarm" => PerfRatchetMode::RegressionAlarm,
                    other => return Err(format!("invalid --mode {other:?}").into()),
                });
            }
            "--promote-dir" => {
                promote_dir = Some(PathBuf::from(next_value(&mut values, "--promote-dir")?));
            }
            "--machine-class" => {
                machine_class = Some(
                    next_value(&mut values, "--machine-class")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--date" => {
                date = Some(
                    next_value(&mut values, "--date")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let promotion_fields = [
        promote_dir.is_some(),
        machine_class.is_some(),
        date.is_some(),
    ];
    if promotion_fields.iter().any(|present| *present)
        && !promotion_fields.iter().all(|present| *present)
    {
        return Err("--promote-dir, --machine-class, and --date must be supplied together".into());
    }
    if let Some(label) = machine_class.as_deref() {
        validate_component(label, "machine class")?;
    }
    if let Some(value) = date.as_deref() {
        validate_component(value, "date")?;
    }

    Ok(Args {
        manifest: manifest.ok_or("missing --manifest")?,
        baseline: baseline.ok_or("missing --baseline")?,
        candidate: candidate.ok_or("missing --candidate")?,
        rerun,
        output: output.ok_or("missing --output")?,
        mode: mode.ok_or("missing --mode")?,
        promote_dir,
        machine_class,
        date,
    })
}

fn next_value<I>(values: &mut I, flag: &str) -> Result<OsString, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn validate_component(value: &str, field: &str) -> Result<(), Box<dyn Error>> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("{field} {value:?} is not a safe filename component").into());
    }
    Ok(())
}

fn read_artifact(path: &Path) -> Result<(PerfGateArtifact, Vec<u8>), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let artifact = serde_json::from_slice::<PerfGateArtifact>(&bytes)?;
    Ok((artifact, bytes))
}

fn gate_activated(manifest: &toml::Value, gate: PerfGate) -> Result<bool, Box<dyn Error>> {
    manifest
        .get("gate")
        .and_then(|gates| gates.get(gate.label()))
        .and_then(|policy| policy.get("activated"))
        .and_then(toml::Value::as_bool)
        .ok_or_else(|| format!("manifest does not define gate.{}.activated", gate.label()).into())
}

fn evidence(role: &str, path: &Path, bytes: &[u8]) -> PerfEvidenceFile {
    PerfEvidenceFile {
        role: role.to_owned(),
        path: path.to_string_lossy().into_owned(),
        sha256: sha256_hex(bytes),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn promote_history_if_requested(
    args: &Args,
    gate: PerfGate,
    candidate_bytes: &[u8],
    updates: &mut Vec<PerfEvidenceFile>,
) -> Result<(), Box<dyn Error>> {
    let (Some(history_dir), Some(machine_class), Some(date)) = (
        args.promote_dir.as_deref(),
        args.machine_class.as_deref(),
        args.date.as_deref(),
    ) else {
        return Ok(());
    };

    let stem = format!("{}.{}", gate.label(), machine_class);
    let latest = history_dir.join(format!("{stem}.latest.json"));
    let rolling = history_dir.join(format!("{stem}.{date}.json"));
    write_file(&latest, candidate_bytes)?;
    write_file(&rolling, candidate_bytes)?;
    updates.push(evidence("history_latest", &latest, candidate_bytes));
    updates.push(evidence("history_window", &rolling, candidate_bytes));
    Ok(())
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

const fn mode_label(mode: PerfRatchetMode) -> &'static str {
    match mode {
        PerfRatchetMode::Promotion => "promotion",
        PerfRatchetMode::RegressionAlarm => "regression-alarm",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_options_are_all_or_nothing() {
        let result = parse_args(
            [
                "--manifest",
                "manifest.toml",
                "--baseline",
                "baseline.json",
                "--candidate",
                "candidate.json",
                "--output",
                "out.json",
                "--mode",
                "promotion",
                "--promote-dir",
                ".bench-history",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(result.is_err());
    }

    #[test]
    fn history_components_reject_path_traversal() {
        assert!(validate_component("../worker", "machine class").is_err());
        assert!(validate_component("github-ubuntu", "machine class").is_ok());
        assert!(validate_component("2026-07-23", "date").is_ok());
    }
}
