# frankensearch-ops

Operations control-plane TUI for frankensearch fleet monitoring.

## Overview

`frankensearch-ops` is an operations console that discovers running frankensearch instances, displays real-time metrics, and provides fleet-wide observability. It builds on the shared `frankensearch-tui` framework and uses FrankenSQLite for telemetry persistence. It can run in demo mode with simulated data or connect to live instances.

```text
+-------------------------------------------+
| frankensearch-ops (this crate)            |
|   OpsApp, screens, discovery, simulator   |
+-------------------------------------------+
| frankensearch-tui (shared framework)      |
|   Screen, ScreenRegistry, AppShell, ...   |
+-------------------------------------------+
| FrankenTUI (ftui-*)                       |
+-------------------------------------------+
```

## Key Types

### Application

- `OpsApp` - main application entry point and event loop
- `AppState` - shared application state for async bridge between data sources and UI

### Data Sources

- `DataSource` - trait for pluggable telemetry data providers
- `StorageDataSource` - production data source backed by FrankenSQLite
- `MockDataSource` - mock data source for testing and demo mode

### Discovery

- `DiscoveryEngine` - discovers running frankensearch instances
- `DiscoveredInstance` / `InstanceSighting` - instance discovery records
- `DiscoveryConfig` / `DiscoverySource` - discovery configuration and source types
- `StaticDiscoverySource` - static list of known instances

### Storage

- `OpsStorage` / `OpsStorageConfig` - ops telemetry database management
- `SloHealth` / `SloMaterializationConfig` - SLO tracking and materialization
- `AnomalyMaterializationSnapshot` - anomaly detection snapshots

### Screens

- `ScreenCategory` - screen groupings for navigation (Fleet, Search, Index, Resource)
- Screens for fleet overview, search metrics, index health, and resource monitoring

### Instance Lifecycle

- `ProjectLifecycleTracker` - tracks instance lifecycle transitions
- `InstanceAttribution` / `InstanceLifecycle` - instance metadata and lifecycle state
- `ControlPlaneHealth` / `ControlPlaneMetrics` - control plane health aggregation

### Simulation

- `TelemetrySimulator` / `TelemetrySimulatorConfig` - generates realistic telemetry data
- `SimulatedProject` / `SimulatedSearchEvent` - simulated project and event models
- `WorkloadProfile` - configurable workload simulation profiles

### Preferences and Theming

- `DisplayPreferences` / `ContrastMode` / `MotionPreference` - accessibility preferences
- `SemanticPalette` / `ThemeVariant` - semantic color palette and theme variants
- `ViewPreset` / `Density` - view density presets (compact, default, comfortable)

### Accessibility

- `FrameQualityTracker` - monitors frame rendering performance
- `KeyboardParityAudit` - ensures all actions are keyboard-accessible

## Usage

```bash
# Launch in demo mode with simulated data
frankensearch-ops --demo

# Connect to a specific database
frankensearch-ops --db-path /path/to/ops.db

# Environment variables
FRANKENSEARCH_OPS_DEMO=1 frankensearch-ops
FRANKENSEARCH_OPS_DB_PATH=/path/to/ops.db frankensearch-ops
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-tui
  ^
  |
frankensearch-ops (binary -- leaf crate)
```

This is a leaf crate. No other crate depends on it.

## License

MIT
