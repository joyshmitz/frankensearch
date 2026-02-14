use std::path::PathBuf;
use std::str::FromStr;

use frankensearch_core::{SearchError, SearchResult};

use crate::config::{CliOverrides, PressureProfile, TuiTheme};

/// Top-level fsfs command entrypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CliCommand {
    Search,
    Index,
    #[default]
    Status,
    Explain,
    Config,
}

/// Parsed CLI input including command and high-priority overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliInput {
    pub command: CliCommand,
    pub overrides: CliOverrides,
}

/// Parse fsfs CLI args.
///
/// The first non-flag token selects the command. Supported flags implement
/// the precedence contract from `bd-2hz.13`.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` if the command/flags are malformed.
pub fn parse_cli_args<I, S>(args: I) -> SearchResult<CliInput>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let tokens: Vec<String> = args.into_iter().map(Into::into).collect();
    let (command, mut idx) = extract_command(&tokens)?;
    let mut overrides = CliOverrides::default();

    while idx < tokens.len() {
        let flag = tokens[idx].as_str();
        match flag {
            "--roots" => {
                let value = expect_value(&tokens, idx, "--roots")?;
                overrides.roots = Some(split_csv(value)?);
                idx += 2;
            }
            "--exclude" => {
                let value = expect_value(&tokens, idx, "--exclude")?;
                overrides.exclude_patterns = Some(split_csv(value)?);
                idx += 2;
            }
            "--limit" => {
                let value = expect_value(&tokens, idx, "--limit")?;
                overrides.limit = Some(parse_usize(value, "search.default_limit")?);
                idx += 2;
            }
            "--fast-only" => {
                overrides.fast_only = Some(true);
                idx += 1;
            }
            "--no-fast-only" => {
                overrides.fast_only = Some(false);
                idx += 1;
            }
            "--explain" => {
                overrides.explain = Some(true);
                idx += 1;
            }
            "--profile" => {
                let value = expect_value(&tokens, idx, "--profile")?;
                overrides.profile = Some(PressureProfile::from_str(value).map_err(|()| {
                    SearchError::InvalidConfig {
                        field: "pressure.profile".into(),
                        value: value.into(),
                        reason: "expected strict|performance|degraded".into(),
                    }
                })?);
                idx += 2;
            }
            "--theme" => {
                let value = expect_value(&tokens, idx, "--theme")?;
                overrides.theme =
                    Some(
                        TuiTheme::from_str(value).map_err(|()| SearchError::InvalidConfig {
                            field: "tui.theme".into(),
                            value: value.into(),
                            reason: "expected auto|light|dark".into(),
                        })?,
                    );
                idx += 2;
            }
            "--config" => {
                let value = expect_value(&tokens, idx, "--config")?;
                overrides.config_path = Some(PathBuf::from(value));
                idx += 2;
            }
            _ => {
                return Err(SearchError::InvalidConfig {
                    field: "cli.flag".into(),
                    value: flag.into(),
                    reason: "unsupported flag for fsfs scaffold".into(),
                });
            }
        }
    }

    Ok(CliInput { command, overrides })
}

fn extract_command(tokens: &[String]) -> SearchResult<(CliCommand, usize)> {
    if let Some(token) = tokens.first()
        && !token.starts_with('-')
    {
        return Ok((parse_command(token)?, 1));
    }
    Ok((CliCommand::default(), 0))
}

fn parse_command(token: &str) -> SearchResult<CliCommand> {
    match token {
        "search" => Ok(CliCommand::Search),
        "index" => Ok(CliCommand::Index),
        "status" => Ok(CliCommand::Status),
        "explain" => Ok(CliCommand::Explain),
        "config" => Ok(CliCommand::Config),
        _ => Err(SearchError::InvalidConfig {
            field: "cli.command".into(),
            value: token.into(),
            reason: "expected search|index|status|explain|config".into(),
        }),
    }
}

fn expect_value<'a>(tokens: &'a [String], idx: usize, flag: &str) -> SearchResult<&'a str> {
    tokens
        .get(idx + 1)
        .map(String::as_str)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "cli.flag".into(),
            value: flag.into(),
            reason: "missing value".into(),
        })
}

fn split_csv(value: &str) -> SearchResult<Vec<String>> {
    let parts: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect();

    if parts.is_empty() {
        return Err(SearchError::InvalidConfig {
            field: "cli.csv".into(),
            value: value.into(),
            reason: "expected at least one comma-separated value".into(),
        });
    }

    Ok(parts)
}

fn parse_usize(value: &str, field: &str) -> SearchResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected unsigned integer".into(),
        })
}

#[cfg(test)]
mod tests {
    use super::{CliCommand, parse_cli_args};

    #[test]
    fn parse_command_and_overrides() {
        let input = parse_cli_args([
            "search",
            "--roots",
            "/repo,/notes",
            "--exclude",
            "target,node_modules",
            "--limit",
            "25",
            "--fast-only",
            "--explain",
            "--profile",
            "degraded",
            "--theme",
            "light",
        ])
        .expect("parse");

        assert_eq!(input.command, CliCommand::Search);
        assert_eq!(input.overrides.limit, Some(25));
        assert_eq!(input.overrides.fast_only, Some(true));
        assert_eq!(input.overrides.explain, Some(true));
        assert_eq!(
            input.overrides.roots.expect("roots"),
            vec!["/repo".to_string(), "/notes".to_string()]
        );
    }

    #[test]
    fn unknown_flag_returns_error() {
        let err = parse_cli_args(["status", "--wat"]).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("unsupported flag"));
    }

    #[test]
    fn default_command_is_status() {
        let input = parse_cli_args(["--limit", "10"]).expect("parse");
        assert_eq!(input.command, CliCommand::Status);
    }
}
