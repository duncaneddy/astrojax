"""Generate cross-validation fixtures from Brahe v1.1.2.

Produces ``access_brahe_trajectory.npz`` containing:
- ECEF trajectory (positions + velocities) at 1-second grid over 24 hours
- Brahe access windows for multiple constraint scenarios

Orbit: SGP4 polar orbit (i=90, e=0.001, n=15.22 rev/day)
Station: Svalbard (15.396518 E, 78.230306 N, 0m)
Duration: 24h from 2020-01-01 00:00:00 UTC

Usage:
    uv run python tests/fixtures/generate_access_brahe_fixtures.py
"""

from pathlib import Path

import brahe as bh
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LINE1 = "1 00001U          20001.00000000  .00000000  00000-0  00000-0 0    07"
LINE2 = "2 00001  90.0000   0.0000 0010000   0.0000   0.0000 15.21936719    07"

STATION_LON_DEG = 15.396518
STATION_LAT_DEG = 78.230306
STATION_ALT_M = 0.0

DURATION_S = 86400.0  # 24 hours
DT_GRID = 1.0  # 1-second trajectory grid

OUTPUT_PATH = Path(__file__).parent / "access_brahe_trajectory.npz"


def main():
    bh.initialize_eop()

    # ---------------------------------------------------------------------------
    # Set up propagator and station
    # ---------------------------------------------------------------------------

    prop = bh.SGPPropagator.from_tle(LINE1, LINE2)
    loc = bh.PointLocation(STATION_LON_DEG, STATION_LAT_DEG, STATION_ALT_M)
    search_start = bh.Epoch(2020, 1, 1, 0, 0, 0)
    search_end = search_start + DURATION_S

    # ---------------------------------------------------------------------------
    # Generate ECEF trajectory at 1-second intervals
    # ---------------------------------------------------------------------------

    n_points = int(DURATION_S / DT_GRID) + 1
    times_s = np.arange(n_points, dtype=np.float64) * DT_GRID
    positions_ecef = np.zeros((n_points, 3), dtype=np.float64)
    velocities_ecef = np.zeros((n_points, 3), dtype=np.float64)

    print(f"Generating trajectory: {n_points} points at {DT_GRID}s intervals...")

    for i in range(n_points):
        epc = search_start + float(times_s[i])
        state_eci = prop.state_eci(epc)
        rot = bh.rotation_eci_to_ecef(epc)

        positions_ecef[i] = rot @ state_eci[:3]
        # Approximate ECEF velocity: rotate ECI velocity (ignores dR/dt*r term,
        # but at 1s grid this is fine for Hermite interpolation).
        velocities_ecef[i] = rot @ state_eci[3:]

    print(
        f"Trajectory done. Pos range: {np.linalg.norm(positions_ecef, axis=1).min():.0f} "
        f"- {np.linalg.norm(positions_ecef, axis=1).max():.0f} m"
    )

    # ---------------------------------------------------------------------------
    # Access search configuration
    # ---------------------------------------------------------------------------

    config = bh.AccessSearchConfig(initial_time_step=10.0, adaptive_step=False)
    time_tolerance = 0.01

    # ---------------------------------------------------------------------------
    # Scenario 1: Elevation >= 0 deg
    # ---------------------------------------------------------------------------

    print("\nScenario 1: el >= 0 deg")
    c_el0 = bh.ElevationConstraint(min_elevation_deg=0.0)
    wins_el0 = bh.location_accesses(
        loc,
        prop,
        search_start,
        search_end,
        c_el0,
        config=config,
        time_tolerance=time_tolerance,
    )
    print(f"  Found {len(wins_el0)} windows")
    el0_t_rise = np.array([w.start - search_start for w in wins_el0], dtype=np.float64)
    el0_t_set = np.array([w.end - search_start for w in wins_el0], dtype=np.float64)
    el0_duration = np.array([w.duration for w in wins_el0], dtype=np.float64)
    for i, _w in enumerate(wins_el0):
        print(
            f"  Window {i:2d}: rise={el0_t_rise[i]:9.2f}s  set={el0_t_set[i]:9.2f}s  "
            f"dur={el0_duration[i]:7.1f}s"
        )

    # ---------------------------------------------------------------------------
    # Scenario 2: Elevation >= 5 deg
    # ---------------------------------------------------------------------------

    print("\nScenario 2: el >= 5 deg")
    c_el5 = bh.ElevationConstraint(min_elevation_deg=5.0)
    wins_el5 = bh.location_accesses(
        loc,
        prop,
        search_start,
        search_end,
        c_el5,
        config=config,
        time_tolerance=time_tolerance,
    )
    print(f"  Found {len(wins_el5)} windows")
    el5_t_rise = np.array([w.start - search_start for w in wins_el5], dtype=np.float64)
    el5_t_set = np.array([w.end - search_start for w in wins_el5], dtype=np.float64)
    el5_duration = np.array([w.duration for w in wins_el5], dtype=np.float64)
    for i, _w in enumerate(wins_el5):
        print(
            f"  Window {i:2d}: rise={el5_t_rise[i]:9.2f}s  set={el5_t_set[i]:9.2f}s  "
            f"dur={el5_duration[i]:7.1f}s"
        )

    # ---------------------------------------------------------------------------
    # Scenario 3: Off-nadir <= 60 deg
    # ---------------------------------------------------------------------------

    print("\nScenario 3: off-nadir <= 60 deg")
    c_on60 = bh.OffNadirConstraint(max_off_nadir_deg=60.0)
    wins_on60 = bh.location_accesses(
        loc,
        prop,
        search_start,
        search_end,
        c_on60,
        config=config,
        time_tolerance=time_tolerance,
    )
    print(f"  Found {len(wins_on60)} windows")
    on60_t_rise = np.array([w.start - search_start for w in wins_on60], dtype=np.float64)
    on60_t_set = np.array([w.end - search_start for w in wins_on60], dtype=np.float64)
    on60_duration = np.array([w.duration for w in wins_on60], dtype=np.float64)
    for i, _w in enumerate(wins_on60):
        print(
            f"  Window {i:2d}: rise={on60_t_rise[i]:9.2f}s  set={on60_t_set[i]:9.2f}s  "
            f"dur={on60_duration[i]:7.1f}s"
        )

    # ---------------------------------------------------------------------------
    # Scenario 4: Elevation >= 5 deg AND off-nadir <= 60 deg
    # ---------------------------------------------------------------------------

    print("\nScenario 4: el >= 5 deg AND off-nadir <= 60 deg")
    c_combined = bh.ConstraintAll([c_el5, c_on60])
    wins_combined = bh.location_accesses(
        loc,
        prop,
        search_start,
        search_end,
        c_combined,
        config=config,
        time_tolerance=time_tolerance,
    )
    print(f"  Found {len(wins_combined)} windows")
    combined_t_rise = np.array([w.start - search_start for w in wins_combined], dtype=np.float64)
    combined_t_set = np.array([w.end - search_start for w in wins_combined], dtype=np.float64)
    combined_duration = np.array([w.duration for w in wins_combined], dtype=np.float64)
    for i, _w in enumerate(wins_combined):
        print(
            f"  Window {i:2d}: rise={combined_t_rise[i]:9.2f}s  set={combined_t_set[i]:9.2f}s  "
            f"dur={combined_duration[i]:7.1f}s"
        )

    # ---------------------------------------------------------------------------
    # Save to .npz
    # ---------------------------------------------------------------------------

    np.savez(
        OUTPUT_PATH,
        # Trajectory
        times_s=times_s,
        positions_ecef=positions_ecef,
        velocities_ecef=velocities_ecef,
        # Station
        station_lon_deg=STATION_LON_DEG,
        station_lat_deg=STATION_LAT_DEG,
        # Scenario 1: el >= 0
        el0_t_rise=el0_t_rise,
        el0_t_set=el0_t_set,
        el0_duration=el0_duration,
        # Scenario 2: el >= 5
        el5_t_rise=el5_t_rise,
        el5_t_set=el5_t_set,
        el5_duration=el5_duration,
        # Scenario 3: off-nadir <= 60
        on60_t_rise=on60_t_rise,
        on60_t_set=on60_t_set,
        on60_duration=on60_duration,
        # Scenario 4: el >= 5 AND off-nadir <= 60
        combined_t_rise=combined_t_rise,
        combined_t_set=combined_t_set,
        combined_duration=combined_duration,
    )

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
