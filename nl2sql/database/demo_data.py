"""Deterministic demo dataset for the bundled observability warehouse."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

RANDOM_SEED = 20240501

OBSERVATION_COUNT = 6000
OBSERVATION_WINDOW_DAYS = 30

FAILURE_RATE = 0.16
DEGRADED_RATE = 0.06
TIMEOUT_RATE = 0.04

_INTERFACE_TEMPLATES = ("Ethernet1/{n}", "Ethernet2/{n}", "xe-0/0/{n}", "et-0/0/{n}")


@dataclass(slots=True)
class DemoDataset:
    """Generated rows, keyed by table name in insertion order."""

    tables: dict[str, list[dict[str, Any]]]

    def row_count(self) -> int:
        """Total number of generated rows across all tables."""
        return sum(len(rows) for rows in self.tables.values())


def _iso(moment: datetime) -> str:
    """Render a timestamp in the format SQLite's datetime functions compare against."""
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def build_demo_dataset(now: datetime | None = None) -> DemoDataset:
    """Generate the full demo dataset.

    Args:
        now: Reference point for all relative timestamps. Defaults to the current
            UTC time.
    """
    rng = random.Random(RANDOM_SEED)
    now = now or datetime.now(UTC)
    tables: dict[str, list[dict[str, Any]]] = {}

    created = _iso(now - timedelta(days=365))

    # --- Reference dimensions -------------------------------------------------
    tables["regions"] = [
        {"region_id": 1, "region_code": "us-east", "region_name": "US East",
         "continent": "North America", "created_at": created},
        {"region_id": 2, "region_code": "us-west", "region_name": "US West",
         "continent": "North America", "created_at": created},
        {"region_id": 3, "region_code": "eu-west", "region_name": "EU West",
         "continent": "Europe", "created_at": created},
        {"region_id": 4, "region_code": "ap-south", "region_name": "Asia Pacific South",
         "continent": "Asia", "created_at": created},
    ]

    tables["environments"] = [
        {"environment_id": 1, "environment_code": "PROD", "environment_name": "Production",
         "tier": "tier_1", "is_production": 1,
         "description": "Live customer-facing infrastructure.", "created_at": created},
        {"environment_id": 2, "environment_code": "STAGE", "environment_name": "Staging",
         "tier": "tier_2", "is_production": 0,
         "description": "Pre-production validation environment.", "created_at": created},
        {"environment_id": 3, "environment_code": "DEV", "environment_name": "Development",
         "tier": "tier_3", "is_production": 0,
         "description": "Engineering sandbox.", "created_at": created},
        {"environment_id": 4, "environment_code": "QA", "environment_name": "Quality Assurance",
         "tier": "tier_3", "is_production": 0,
         "description": "Automated test environment.", "created_at": created},
    ]

    tables["sites"] = [
        {"site_id": 1, "site_code": "IAD1", "site_name": "Ashburn 1", "region_id": 1,
         "address": "21715 Filigree Ct, Ashburn VA", "latitude": 39.0438,
         "longitude": -77.4874, "is_active": 1, "created_at": created},
        {"site_id": 2, "site_code": "IAD2", "site_name": "Ashburn 2", "region_id": 1,
         "address": "44060 Digital Loudoun Plaza, Ashburn VA", "latitude": 39.0169,
         "longitude": -77.5390, "is_active": 1, "created_at": created},
        {"site_id": 3, "site_code": "SFO2", "site_name": "San Jose 2", "region_id": 2,
         "address": "1500 Space Park Dr, Santa Clara CA", "latitude": 37.3541,
         "longitude": -121.9552, "is_active": 1, "created_at": created},
        {"site_id": 4, "site_code": "DUB1", "site_name": "Dublin 1", "region_id": 3,
         "address": "Unit 4 Grange Castle, Dublin", "latitude": 53.3169,
         "longitude": -6.4306, "is_active": 1, "created_at": created},
        {"site_id": 5, "site_code": "SIN1", "site_name": "Singapore 1", "region_id": 4,
         "address": "29 Tai Seng Ave, Singapore", "latitude": 1.3336,
         "longitude": 103.8885, "is_active": 1, "created_at": created},
    ]

    tables["teams"] = [
        {"team_id": 1, "team_name": "Core Network", "team_email": "core-net@example.com",
         "escalation_policy": "always_on", "created_at": created},
        {"team_id": 2, "team_name": "Edge Platform", "team_email": "edge@example.com",
         "escalation_policy": "follow_the_sun", "created_at": created},
        {"team_id": 3, "team_name": "Site Reliability", "team_email": "sre@example.com",
         "escalation_policy": "always_on", "created_at": created},
        {"team_id": 4, "team_name": "Security Engineering", "team_email": "secops@example.com",
         "escalation_policy": "business_hours", "created_at": created},
    ]

    tables["users"] = [
        {"user_id": index, "username": username, "full_name": full_name,
         "email": f"{username}@example.com", "team_id": team_id, "role": role,
         "is_active": 1, "created_at": created}
        for index, (username, full_name, team_id, role) in enumerate(
            [
                ("a.okafor", "Amara Okafor", 1, "admin"),
                ("l.svensson", "Linnea Svensson", 1, "engineer"),
                ("r.mehta", "Rohan Mehta", 2, "engineer"),
                ("j.tanaka", "Jun Tanaka", 2, "analyst"),
                ("m.duarte", "Mariana Duarte", 3, "engineer"),
                ("k.novak", "Karel Novak", 3, "admin"),
                ("s.haddad", "Sami Haddad", 4, "engineer"),
                ("p.lindqvist", "Petra Lindqvist", 4, "viewer"),
            ],
            start=1,
        )
    ]

    tables["vendors"] = [
        {"vendor_id": 1, "vendor_name": "Northbridge Systems", "support_tier": "platinum",
         "contract_expiry_date": "2027-03-31", "created_at": created},
        {"vendor_id": 2, "vendor_name": "Helix Networks", "support_tier": "gold",
         "contract_expiry_date": "2026-11-30", "created_at": created},
        {"vendor_id": 3, "vendor_name": "Cirrus Fabric", "support_tier": "silver",
         "contract_expiry_date": "2026-08-15", "created_at": created},
    ]

    tables["device_models"] = [
        {"model_id": 1, "vendor_id": 1, "model_name": "NB-9500 Core Router",
         "device_category": "router", "end_of_support_date": "2029-01-31",
         "created_at": created},
        {"model_id": 2, "vendor_id": 1, "model_name": "NB-4200 Aggregation Switch",
         "device_category": "switch", "end_of_support_date": "2028-06-30",
         "created_at": created},
        {"model_id": 3, "vendor_id": 2, "model_name": "HX-770 Edge Router",
         "device_category": "router", "end_of_support_date": "2027-12-31",
         "created_at": created},
        {"model_id": 4, "vendor_id": 2, "model_name": "HX-310 Access Switch",
         "device_category": "switch", "end_of_support_date": "2027-09-30",
         "created_at": created},
        {"model_id": 5, "vendor_id": 3, "model_name": "CF-100 Firewall",
         "device_category": "firewall", "end_of_support_date": "2028-02-28",
         "created_at": created},
    ]

    tables["interface_types"] = [
        {"interface_type_id": 1, "type_name": "1000BASE-T", "media": "copper",
         "nominal_speed_mbps": 1000},
        {"interface_type_id": 2, "type_name": "10GBASE-SR", "media": "fibre",
         "nominal_speed_mbps": 10000},
        {"interface_type_id": 3, "type_name": "40GBASE-SR4", "media": "fibre",
         "nominal_speed_mbps": 40000},
        {"interface_type_id": 4, "type_name": "100GBASE-LR4", "media": "fibre",
         "nominal_speed_mbps": 100000},
    ]

    tables["observation_types"] = [
        {"observation_type_id": 1, "type_code": "LINK_STATE", "type_name": "Link state poll",
         "category": "availability",
         "description": "Polls the operational state of an interface."},
        {"observation_type_id": 2, "type_code": "LATENCY_PROBE", "type_name": "Latency probe",
         "category": "performance",
         "description": "Measures round trip time across the link."},
        {"observation_type_id": 3, "type_code": "PACKET_LOSS", "type_name": "Packet loss sample",
         "category": "performance",
         "description": "Samples the proportion of probes that were dropped."},
        {"observation_type_id": 4, "type_code": "THROUGHPUT", "type_name": "Throughput sample",
         "category": "capacity",
         "description": "Measures bits per second carried by the interface."},
        {"observation_type_id": 5, "type_code": "ERROR_COUNTER", "type_name": "Error counter read",
         "category": "availability",
         "description": "Reads interface error counters from the device."},
        {"observation_type_id": 6, "type_code": "BGP_SESSION", "type_name": "BGP session check",
         "category": "routing",
         "description": "Verifies that the BGP session is established."},
    ]

    tables["failure_reasons"] = [
        {"failure_reason_id": 1, "reason_code": "TIMEOUT", "reason_name": "Poll timed out",
         "severity": "high", "is_actionable": 1,
         "description": "The device did not answer within the poll interval."},
        {"failure_reason_id": 2, "reason_code": "AUTH_FAILED",
         "reason_name": "Authentication failed", "severity": "critical", "is_actionable": 1,
         "description": "Credentials were rejected by the device."},
        {"failure_reason_id": 3, "reason_code": "LINK_DOWN", "reason_name": "Link is down",
         "severity": "critical", "is_actionable": 1,
         "description": "The interface reported an operationally down state."},
        {"failure_reason_id": 4, "reason_code": "HIGH_LATENCY", "reason_name": "Latency above threshold",
         "severity": "medium", "is_actionable": 1,
         "description": "Round trip time exceeded the configured threshold."},
        {"failure_reason_id": 5, "reason_code": "PACKET_LOSS", "reason_name": "Packet loss detected",
         "severity": "high", "is_actionable": 1,
         "description": "A material proportion of probes were dropped."},
        {"failure_reason_id": 6, "reason_code": "SNMP_ERROR", "reason_name": "SNMP error returned",
         "severity": "medium", "is_actionable": 0,
         "description": "The device returned an SNMP protocol error."},
        {"failure_reason_id": 7, "reason_code": "UNREACHABLE", "reason_name": "Device unreachable",
         "severity": "critical", "is_actionable": 1,
         "description": "No network path to the management address."},
    ]

    # --- Devices, interfaces and collectors -----------------------------------
    tables["devices"] = _build_devices(rng, tables["sites"], now)
    tables["interfaces"] = _build_interfaces(rng, tables["devices"], now)
    tables["collectors"] = _build_collectors(tables["sites"], now)

    # --- Facts ----------------------------------------------------------------
    observations, metrics = _build_observations(
        rng, tables["interfaces"], tables["devices"], tables["collectors"], now
    )
    tables["observations"] = observations
    tables["observation_metrics"] = metrics

    tables["alert_rules"] = _build_alert_rules(now)
    tables["incidents"] = _build_incidents(rng, now)
    tables["alerts"] = _build_alerts(rng, observations, tables["incidents"], now)
    tables["incident_events"] = _build_incident_events(tables["incidents"])
    tables["maintenance_windows"] = _build_maintenance_windows(tables["devices"], now)
    tables["slo_targets"] = _build_slo_targets()
    tables["audit_logs"] = _build_audit_logs(rng, now)
    tables["purchase_orders"] = _build_purchase_orders(rng, tables["vendors"], now)

    dataset = DemoDataset(tables=tables)
    logger.info("Demo dataset generated: %d rows", dataset.row_count())
    return dataset


def _build_devices(
    rng: random.Random, sites: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    """Generate a device fleet spread across sites, environments and models."""
    devices: list[dict[str, Any]] = []
    device_id = 1

    environment_weights = [(1, 5), (2, 2), (3, 1), (4, 1)]
    roles = ("core", "edge", "agg", "fw")

    for site in sites:
        for environment_id, device_count in environment_weights:
            for index in range(device_count):
                role = roles[index % len(roles)]
                model_id = rng.choice([1, 2, 3, 4, 5])
                devices.append(
                    {
                        "device_id": device_id,
                        "device_name": (
                            f"{site['site_code'].lower()}-{role}-"
                            f"{environment_id}{index + 1:02d}"
                        ),
                        "model_id": model_id,
                        "site_id": site["site_id"],
                        "environment_id": environment_id,
                        "owner_team_id": rng.choice([1, 2, 3, 4]),
                        "serial_number": f"SN{rng.randrange(10**9, 10**10)}",
                        "management_ip": (
                            f"10.{site['site_id']}.{environment_id}.{index + 10}"
                        ),
                        "device_status": (
                            "DECOMMISSIONED" if device_id % 37 == 0 else "ACTIVE"
                        ),
                        "commissioned_at": _iso(
                            now - timedelta(days=rng.randrange(90, 900))
                        ),
                        "created_at": _iso(now - timedelta(days=365)),
                    }
                )
                device_id += 1

    return devices


def _build_interfaces(
    rng: random.Random, devices: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    """Generate interfaces for each device, a few of which are uplinks."""
    interfaces: list[dict[str, Any]] = []
    interface_id = 1

    for device in devices:
        for port_number in range(1, rng.randrange(4, 7)):
            template = _INTERFACE_TEMPLATES[interface_id % len(_INTERFACE_TEMPLATES)]
            interface_type_id = rng.choice([1, 2, 2, 3, 4])
            is_uplink = port_number <= 2
            oper_status = "DOWN" if interface_id % 23 == 0 else "UP"

            interfaces.append(
                {
                    "interface_id": interface_id,
                    "device_id": device["device_id"],
                    "interface_type_id": interface_type_id,
                    "interface_name": template.format(n=port_number),
                    "description": (
                        "Uplink to core fabric" if is_uplink else "Access port"
                    ),
                    "admin_status": "UP",
                    "oper_status": oper_status,
                    "speed_mbps": {1: 1000, 2: 10000, 3: 40000, 4: 100000}[
                        interface_type_id
                    ],
                    "mtu": 9000 if is_uplink else 1500,
                    "is_uplink": int(is_uplink),
                    "created_at": _iso(now - timedelta(days=300)),
                }
            )
            interface_id += 1

    return interfaces


def _build_collectors(sites: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Generate one collector agent per site."""
    return [
        {
            "collector_id": site["site_id"],
            "collector_name": f"collector-{site['site_code'].lower()}-01",
            "site_id": site["site_id"],
            "collector_version": "4.2.1",
            "poll_interval_seconds": 60,
            "collector_status": "OFFLINE" if site["site_id"] == 5 else "HEALTHY",
            "last_heartbeat_at": _iso(
                now - timedelta(hours=6 if site["site_id"] == 5 else 0, minutes=2)
            ),
            "created_at": _iso(now - timedelta(days=300)),
        }
        for site in sites
    ]


def _build_observations(
    rng: random.Random,
    interfaces: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    collectors: list[dict[str, Any]],
    now: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate the observations fact table and its attached metric samples."""
    devices_by_id = {device["device_id"]: device for device in devices}
    collectors_by_site = {
        collector["site_id"]: collector["collector_id"] for collector in collectors
    }

    # A minority of interfaces are made unreliable, so rankings are non-uniform.
    unreliable = {
        interface["interface_id"]
        for interface in rng.sample(interfaces, k=max(len(interfaces) // 12, 1))
    }

    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    metric_id = 1

    for observation_id in range(1, OBSERVATION_COUNT + 1):
        interface = rng.choice(interfaces)
        device = devices_by_id[interface["device_id"]]

        # Weighted towards the recent past so that short windows still return rows.
        age_hours = rng.triangular(0, OBSERVATION_WINDOW_DAYS * 24, 0)
        observed_at = now - timedelta(hours=age_hours)

        failure_chance = FAILURE_RATE * (3.0 if interface["interface_id"] in unreliable else 1.0)
        roll = rng.random()
        if roll < failure_chance:
            status = "FAILED"
        elif roll < failure_chance + DEGRADED_RATE:
            status = "DEGRADED"
        elif roll < failure_chance + DEGRADED_RATE + TIMEOUT_RATE:
            status = "TIMEOUT"
        else:
            status = "SUCCESS"

        observation_type_id = rng.choice([1, 1, 2, 2, 3, 4, 5, 6])
        failure_reason_id = (
            rng.choice([1, 2, 3, 3, 4, 5, 6, 7]) if status == "FAILED" else None
        )

        observations.append(
            {
                "observation_id": observation_id,
                "interface_id": interface["interface_id"],
                "device_id": device["device_id"],
                "environment_id": device["environment_id"],
                "observation_type_id": observation_type_id,
                "collector_id": collectors_by_site.get(device["site_id"]),
                "status": status,
                "failure_reason_id": failure_reason_id,
                "observed_at": _iso(observed_at),
                "duration_ms": rng.randrange(5, 2500),
                "created_at": _iso(observed_at + timedelta(seconds=3)),
            }
        )

        # Performance checks emit a numeric sample; availability checks do not.
        if observation_type_id in (2, 3, 4):
            metric_name, unit, value = {
                2: ("latency_ms", "ms", round(rng.uniform(0.4, 320.0), 2)),
                3: ("packet_loss_pct", "percent", round(rng.uniform(0.0, 12.0), 2)),
                4: ("throughput_mbps", "mbps", round(rng.uniform(50.0, 9500.0), 2)),
            }[observation_type_id]

            metrics.append(
                {
                    "metric_id": metric_id,
                    "observation_id": observation_id,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "metric_unit": unit,
                    "recorded_at": _iso(observed_at),
                }
            )
            metric_id += 1

    return observations, metrics


def _build_alert_rules(now: datetime) -> list[dict[str, Any]]:
    """Generate the alert rule catalogue."""
    created = _iso(now - timedelta(days=200))
    definitions = [
        (1, "Interface down", 1, "critical", 1.0, "==", 5, 1),
        (2, "Latency above 200ms", 2, "high", 200.0, ">", 5, 1),
        (3, "Packet loss above 2%", 3, "high", 2.0, ">", 10, 1),
        (4, "Throughput below 100 Mbps", 4, "medium", 100.0, "<", 15, 1),
        (5, "Rising error counters", 5, "medium", 50.0, ">", 30, 1),
        (6, "BGP session down", 6, "critical", 1.0, "==", 5, 1),
        (7, "Legacy latency rule", 2, "low", 500.0, ">", 60, 0),
    ]

    return [
        {
            "rule_id": rule_id,
            "rule_name": name,
            "observation_type_id": observation_type_id,
            "severity": severity,
            "threshold_value": threshold,
            "comparison_operator": operator,
            "evaluation_window_minutes": window,
            "is_enabled": enabled,
            "created_at": created,
        }
        for rule_id, name, observation_type_id, severity, threshold, operator, window, enabled
        in definitions
    ]


def _build_incidents(rng: random.Random, now: datetime) -> list[dict[str, Any]]:
    """Generate incidents, roughly two thirds of which are already resolved."""
    incidents: list[dict[str, Any]] = []

    for incident_id in range(1, 25):
        opened_at = now - timedelta(hours=rng.randrange(1, 24 * 25))
        is_resolved = incident_id % 3 != 0
        resolved_at = (
            opened_at + timedelta(minutes=rng.randrange(20, 900)) if is_resolved else None
        )

        incidents.append(
            {
                "incident_id": incident_id,
                "incident_key": f"INC-2024-{incident_id:04d}",
                "title": rng.choice(
                    [
                        "Elevated packet loss on aggregation fabric",
                        "Edge router unreachable after maintenance",
                        "BGP session flapping between regions",
                        "Latency regression on transatlantic link",
                        "Collector outage causing observation gap",
                    ]
                ),
                "severity": rng.choice(["sev1", "sev2", "sev2", "sev3", "sev4"]),
                "incident_status": "RESOLVED" if is_resolved else "INVESTIGATING",
                "environment_id": rng.choice([1, 1, 1, 2, 3]),
                "owner_team_id": rng.choice([1, 2, 3, 4]),
                "opened_at": _iso(opened_at),
                "resolved_at": _iso(resolved_at) if resolved_at else None,
                "resolution_summary": (
                    "Replaced the faulty optic and confirmed the link was stable."
                    if is_resolved
                    else None
                ),
            }
        )

    return incidents


def _build_alerts(
    rng: random.Random,
    observations: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    """Generate alerts from the subset of observations that would breach a rule."""
    breaching = [
        observation
        for observation in observations
        if observation["status"] in {"FAILED", "TIMEOUT"}
    ]
    sampled = rng.sample(breaching, k=min(len(breaching), 400))
    incident_ids = [incident["incident_id"] for incident in incidents]

    alerts: list[dict[str, Any]] = []
    for alert_id, observation in enumerate(sampled, start=1):
        triggered_at = datetime.strptime(
            observation["observed_at"], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=UTC) + timedelta(seconds=30)

        severity = rng.choice(["critical", "high", "high", "medium", "low"])
        status = rng.choice(["OPEN", "ACKNOWLEDGED", "RESOLVED", "RESOLVED", "SUPPRESSED"])

        acknowledged_at = (
            triggered_at + timedelta(minutes=rng.randrange(1, 90))
            if status in {"ACKNOWLEDGED", "RESOLVED"}
            else None
        )
        resolved_at = (
            triggered_at + timedelta(minutes=rng.randrange(30, 600))
            if status == "RESOLVED"
            else None
        )

        alerts.append(
            {
                "alert_id": alert_id,
                "rule_id": rng.choice([1, 2, 3, 4, 5, 6]),
                "observation_id": observation["observation_id"],
                "device_id": observation["device_id"],
                "incident_id": rng.choice(incident_ids) if alert_id % 6 == 0 else None,
                "severity": severity,
                "alert_status": status,
                "triggered_at": _iso(triggered_at),
                "acknowledged_at": _iso(acknowledged_at) if acknowledged_at else None,
                "acknowledged_by_user_id": (
                    rng.randrange(1, 9) if acknowledged_at else None
                ),
                "resolved_at": _iso(resolved_at) if resolved_at else None,
            }
        )

    return alerts


def _build_incident_events(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate a short timeline for every incident."""
    events: list[dict[str, Any]] = []
    event_id = 1

    for incident in incidents:
        opened_at = datetime.strptime(
            incident["opened_at"], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=UTC)

        timeline = [
            ("status_change", "Incident declared and response started.", 0),
            ("comment", "Correlated failing observations to a single site.", 12),
            ("mitigation", "Traffic drained away from the affected path.", 35),
        ]
        if incident["resolved_at"]:
            timeline.append(("resolution", "Root cause fixed and service restored.", 90))

        for event_type, description, offset_minutes in timeline:
            events.append(
                {
                    "incident_event_id": event_id,
                    "incident_id": incident["incident_id"],
                    "event_type": event_type,
                    "event_description": description,
                    "actor_user_id": (event_id % 8) + 1,
                    "occurred_at": _iso(opened_at + timedelta(minutes=offset_minutes)),
                }
            )
            event_id += 1

    return events


def _build_purchase_orders(
    rng: random.Random, vendors: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    """Generate hardware purchase orders spread across vendors and the last year."""
    cost_by_tier = {"platinum": (48_000, 260_000), "gold": (22_000, 120_000)}
    statuses = ["OPEN", "APPROVED", "DELIVERED", "DELIVERED", "CANCELLED"]
    orders: list[dict[str, Any]] = []

    for index in range(1, 121):
        vendor = vendors[index % len(vendors)]
        low, high = cost_by_tier.get(vendor["support_tier"], (8_000, 45_000))

        orders.append(
            {
                "purchase_order_id": index,
                "vendor_id": vendor["vendor_id"],
                "order_status": rng.choice(statuses),
                "total_cost": rng.randrange(low, high, 500),
                "ordered_at": _iso(now - timedelta(days=rng.randint(0, 365))),
            }
        )

    return orders


def _build_maintenance_windows(
    devices: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    """Generate a handful of past, active and scheduled maintenance windows."""
    windows: list[dict[str, Any]] = []

    for index, offset_days in enumerate([-14, -7, -1, 0, 3, 10], start=1):
        device = devices[index * 5 % len(devices)]
        starts_at = now + timedelta(days=offset_days)

        if offset_days < 0:
            status = "COMPLETED"
        elif offset_days == 0:
            status = "IN_PROGRESS"
        else:
            status = "SCHEDULED"

        windows.append(
            {
                "maintenance_window_id": index,
                "site_id": device["site_id"],
                "device_id": device["device_id"] if index % 2 == 0 else None,
                "title": "Line card firmware upgrade",
                "window_status": status,
                "starts_at": _iso(starts_at),
                "ends_at": _iso(starts_at + timedelta(hours=4)),
                "requested_by_user_id": (index % 8) + 1,
                "created_at": _iso(now - timedelta(days=30)),
            }
        )

    return windows


def _build_slo_targets() -> list[dict[str, Any]]:
    """Generate availability targets for each environment and check type."""
    targets: list[dict[str, Any]] = []
    target_id = 1

    target_by_environment = {1: 99.9, 2: 99.0, 3: 95.0, 4: 95.0}

    for environment_id, target_rate in target_by_environment.items():
        for observation_type_id in (1, 2, 3):
            targets.append(
                {
                    "slo_target_id": target_id,
                    "environment_id": environment_id,
                    "observation_type_id": observation_type_id,
                    "target_success_rate": target_rate,
                    "measurement_window_days": 30,
                    "is_active": 1,
                }
            )
            target_id += 1

    return targets


def _build_audit_logs(rng: random.Random, now: datetime) -> list[dict[str, Any]]:
    """Generate an audit trail of recent configuration changes."""
    entity_types = ("device", "interface", "alert_rule", "maintenance_window")
    actions = ("create", "update", "disable", "enable")

    return [
        {
            "audit_log_id": audit_id,
            "actor_user_id": rng.randrange(1, 9),
            "entity_type": rng.choice(entity_types),
            "entity_id": rng.randrange(1, 60),
            "action": rng.choice(actions),
            "changed_at": _iso(now - timedelta(hours=rng.randrange(1, 24 * 60))),
            "details": '{"before": "…", "after": "…"}',
        }
        for audit_id in range(1, 121)
    ]
