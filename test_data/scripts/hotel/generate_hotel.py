"""Simulate a hotel with M rooms, on/off room heaters and one temperature sensor per room.

Building   F floors; on each floor M/F rooms sit on the north and south side of a central corridor.
           Every zone (room or corridor) is a two-node RC model in the spirit of ISO 13790: a fast
           air node (air + furniture) and a slow mass node (slabs, walls). Air nodes exchange heat
           with neighbouring rooms (partitions), the corridor (wall + door), the rooms above/below
           (slabs) and the outdoors (windows + infiltration); mass nodes lose heat through the
           opaque facade, the roof (top floor) and the ground slab (ground floor).
Weather    outdoor temperature = mean + cloud-dependent diurnal cycle + Ornstein-Uhlenbeck weather;
           winter solar irradiance on the south / north facade with random daily cloudiness.
Guests     random bookings (check-in 14-21h, check-out 7-11:30h), guests go out during the day and
           some evenings, internal gains while present. Each guest picks a setpoint or switches the
           heating off ("heated or not").
Heating    --policy occupancy: hysteresis thermostats (guest setpoint when checked in, setback when
           vacant, frost protection always). --policy random: room heaters switch as random telegraph
           processes regardless of temperature (open-loop excitation for system identification).
           Corridors are always thermostat-controlled.
Sensors    one per room: air temperature + fixed per-sensor bias + white noise, quantised.

The linear network is integrated exactly (zero-order hold) on 1-minute steps; thermostats act every
step. A burn-in period is simulated and discarded. Everything is stored at the sensor interval.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import expm

DATA = Path(__file__).resolve().parents[2] / "data" / "hotel"

RHO_CP_AIR = 1200.0  # J/(m^3 K)
ROOM_W, ROOM_D, STOREY = 4.0, 6.0, 2.7  # room width along the corridor, depth, storey height (m)
CORRIDOR_W = 2.0
T_GROUND = 10.0
G_WINDOW = 0.5  # solar energy transmittance of the glazing
SOLAR_TO_AIR = 0.3  # share of solar gains released to the air node (rest heats the mass)
CORRIDOR_SETPOINT = 18.0
RANDOM_MEAN_ON_MIN, RANDOM_MEAN_OFF_MIN = 30.0, 90.0


def layout(M, F):
    """Room -> (floor, side, position along the corridor); side 0 = north, 1 = south."""
    if M % F:
        raise SystemExit(f"--rooms ({M}) must be divisible by --floors ({F})")
    per_floor = M // F
    n_north = (per_floor + 1) // 2
    rows = [(f, s, p) for f in range(F) for s, n in ((0, n_north), (1, per_floor - n_north)) for p in range(n)]
    floor, side, pos = map(np.array, zip(*rows))
    return floor, side, pos, n_north


def build_network(floor, side, pos, F, n_cols, rng):
    """Zones 0..M-1 are rooms, M..M+F-1 the corridors of floors 0..F-1. State = [T_air (Z), T_mass (Z)]."""
    M = len(floor)
    Z = M + F
    zfloor = np.r_[floor, np.arange(F)]

    size = rng.uniform(0.85, 1.25, M)  # room size factor
    area = np.r_[ROOM_W * ROOM_D * size, np.full(F, CORRIDOR_W * ROOM_W * n_cols)]
    window = np.r_[rng.uniform(2.2, 3.8, M), np.zeros(F)]
    volume = area * STOREY

    C_air = RHO_CP_AIR * volume * np.r_[np.full(M, 5.0), np.full(F, 3.0)]  # furniture: x5 the air capacity
    C_mass = 165e3 * area  # ISO 13790 'medium' class
    H_am = 9.0 * area  # air <-> internal surfaces / mass

    ach = np.r_[rng.uniform(0.3, 0.7, M), np.full(F, 1.0)]  # infiltration, air changes per hour
    H_ao = ach * volume * RHO_CP_AIR / 3600 + 1.3 * window  # air -> outdoors (infiltration + glazing U=1.3)
    H_ao[M:] += 15.0  # corridor end doors, stairwell leakage

    n_side = {(f, s): np.sum((floor == f) & (side == s)) for f in range(F) for s in (0, 1)}
    corner = np.array([pos[i] in (0, n_side[floor[i], side[i]] - 1) for i in range(M)])
    facade = np.r_[ROOM_W * np.sqrt(size) * STOREY - window[:M] + corner * ROOM_D * STOREY,
                   np.full(F, 2 * CORRIDOR_W * STOREY)]
    H_mo = 0.5 * facade + 0.25 * area * (zfloor == F - 1)  # opaque facade (U=0.5) + roof (U=0.25)
    H_mg = 0.35 * area * (zfloor == 0)  # ground slab

    G = np.zeros((Z, Z))  # air <-> air conductances

    def link(i, j, h):
        G[i, j] += h
        G[j, i] += h

    room = {(floor[i], side[i], pos[i]): i for i in range(M)}
    for i in range(M):
        f, s, p = floor[i], side[i], pos[i]
        if (f, s, p + 1) in room:
            link(i, room[f, s, p + 1], 1.0 * ROOM_D * STOREY)  # partition wall, U=1.0
        if (f + 1, s, p) in room:
            link(i, room[f + 1, s, p], 1.2 * area[i])  # floor slab, U=1.2
        link(i, M + f, 1.0 * (ROOM_W * STOREY - 2.0) + 2.0 * 2.0)  # corridor wall + door
    for f in range(F - 1):
        link(M + f, M + f + 1, 40.0 + 1.2 * area[M + f])  # stairwell + slab

    A = np.zeros((2 * Z, 2 * Z))
    A[:Z, :Z] = -(np.diag(G.sum(1)) - G + np.diag(H_am + H_ao)) / C_air[:, None]
    A[:Z, Z:] = np.diag(H_am / C_air)
    A[Z:, :Z] = np.diag(H_am / C_mass)
    A[Z:, Z:] = -np.diag((H_am + H_mo + H_mg) / C_mass)

    power = np.r_[np.round(rng.uniform(1000, 1600, M), -1), np.full(F, 400.0 * n_cols)]
    return dict(A=A, G=G, C_air=C_air, C_mass=C_mass, H_am=H_am, H_ao=H_ao, H_mo=H_mo, H_mg=H_mg,
                area=area, window=window, power=power)


def weather(t_h, rng, T_mean):
    """Outdoor temperature (C) and solar irradiance on the south / north facade (W/m^2)."""
    days = np.arange(np.floor(t_h[0] / 24) - 1, np.ceil(t_h[-1] / 24) + 2)
    clear = np.interp(t_h, days * 24 + 12, rng.beta(1.0, 1.4, days.size))  # 0 overcast .. 1 clear sky

    dt, tau, sigma = t_h[1] - t_h[0], 72.0, 3.0  # Ornstein-Uhlenbeck weather anomaly
    a = np.exp(-dt / tau)
    kicks = sigma * np.sqrt(1 - a**2) * rng.standard_normal(t_h.size)
    ou = np.empty_like(t_h)
    ou[0] = sigma * rng.standard_normal()
    for k in range(1, t_h.size):
        ou[k] = a * ou[k - 1] + kicks[k]

    hour = t_h % 24
    T_out = T_mean + ou + (1.5 + 3.0 * clear) * np.cos(2 * np.pi * (hour - 14.5) / 24)
    sun = np.clip(np.sin(np.pi * (hour - 8.0) / 8.5), 0, None)  # winter day, 8:00-16:30
    diffuse = 80 * (1 - 0.5 * clear) * sun
    return T_out, 450 * clear * sun**1.3 + diffuse, diffuse


def bookings(M, t_h, rng, mean_vacancy, p_heating_off):
    """Per-minute occupancy: checked_in, occupants present, guest setpoint (NaN = heating off)."""
    dt = t_h[1] - t_h[0]
    K = t_h.size

    def idx(h):
        return int(np.clip(round((h - t_h[0]) / dt), 0, K))

    checked_in = np.zeros((K, M), bool)
    occupants = np.zeros((K, M), np.int8)
    guest_sp = np.full((K, M), np.nan)
    for i in range(M):
        d = int(np.floor(t_h[0] / 24)) - int(rng.integers(0, 4))
        while d * 24 < t_h[-1]:
            nights = int(rng.choice([1, 2, 3, 4, 5, 7], p=[0.3, 0.25, 0.18, 0.12, 0.08, 0.07]))
            ci = d * 24 + rng.uniform(14, 21)
            co = (d + nights) * 24 + rng.uniform(7, 11.5)
            heats = rng.random() >= p_heating_off
            sp = np.clip(np.round(rng.normal(21.5, 1.0) * 2) / 2, 19.0, 24.5) if heats else np.nan
            a, b = idx(ci), idx(co)
            checked_in[a:b, i] = True
            guest_sp[a:b, i] = sp
            occupants[a:b, i] = rng.choice([1, 2], p=[0.4, 0.6])
            for day in range(d, d + nights):
                if day > d and rng.random() < 0.7:  # out during the day
                    occupants[idx(max(day * 24 + rng.uniform(8, 11), ci)):idx(day * 24 + rng.uniform(15, 20)), i] = 0
                if rng.random() < 0.3:  # out for dinner
                    occupants[idx(max(day * 24 + rng.uniform(18.5, 20), ci)):idx(day * 24 + rng.uniform(21, 23.5)), i] = 0
            d += nights + int(rng.geometric(1 / (1 + mean_vacancy))) - 1  # 0 vacant nights = same-day turnover
    return checked_in, occupants, guest_sp


def simulate(net, M, F, side, T_out, I_south, I_north, setpoint, gains, policy, rng, dt_s, frost, hyst,
             rec_every, T0=17.0):
    Z = M + F
    A = net["A"]
    aug = np.zeros((4 * Z, 4 * Z))
    aug[:2 * Z, :2 * Z] = A
    aug[:2 * Z, 2 * Z:] = np.eye(2 * Z)
    E = expm(aug * dt_s)
    Ad, Phi = E[:2 * Z, :2 * Z], E[:2 * Z, 2 * Z:]  # x+ = Ad x + Phi f  for f held constant over the step

    # Exogenous forcing (weather, solar, internal gains) for every step, pushed through Phi in one go.
    solar = G_WINDOW * net["window"] * np.where(np.r_[side, np.zeros(F)] == 1, I_south[:, None], I_north[:, None])
    q_air = np.c_[gains, np.zeros((len(T_out), F))] + SOLAR_TO_AIR * solar
    f_air = (q_air + net["H_ao"] * T_out[:, None]) / net["C_air"]
    f_mass = ((1 - SOLAR_TO_AIR) * solar + net["H_mo"] * T_out[:, None] + net["H_mg"] * T_GROUND) / net["C_mass"]
    x_exo = np.c_[f_air, f_mass] @ Phi.T
    B_heat = Phi[:, :Z] / net["C_air"]

    K = len(T_out)
    N = K // rec_every
    X = np.empty((N, 2 * Z))
    duty = np.zeros((N, Z))
    x = np.full(2 * Z, T0)
    on = np.zeros(Z, bool)
    sp_corridor = np.full(F, CORRIDOR_SETPOINT)
    p_switch_on, p_switch_off = dt_s / 60 / RANDOM_MEAN_OFF_MIN, dt_s / 60 / RANDOM_MEAN_ON_MIN
    for k in range(K):
        r, j = divmod(k, rec_every)
        if j == 0:
            X[r] = x
        Ta = x[:Z]
        sp = np.r_[setpoint[k], sp_corridor]
        sp = np.where(np.isnan(sp), frost, np.maximum(sp, frost))
        thermostat = (Ta < sp - hyst) | (on & (Ta < sp + hyst))
        if policy == "random":
            flip = rng.random(M) < np.where(on[:M], p_switch_off, p_switch_on)
            thermostat[:M] = on[:M] ^ flip
        on = thermostat
        duty[r] += on
        x = Ad @ x + x_exo[k] + B_heat @ (on * net["power"])
    return X, duty / rec_every


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DATA / "hotel.h5")
    p.add_argument("--rooms", type=int, default=40, help="number of rooms M (divisible by --floors)")
    p.add_argument("--floors", type=int, default=4)
    p.add_argument("--days", type=float, default=28.0)
    p.add_argument("--burn-in-days", type=float, default=5.0, help="simulated first, then discarded")
    p.add_argument("--dt-sim", type=float, default=60.0, help="simulation / thermostat step (s)")
    p.add_argument("--dt-sample", type=float, default=300.0, help="sensor sampling interval (s)")
    p.add_argument("--t-mean", type=float, default=2.0, help="mean outdoor temperature (C)")
    p.add_argument("--policy", choices=["occupancy", "random"], default="occupancy")
    p.add_argument("--vacant-setpoint", type=float, default=16.0, help="setback for vacant rooms (C)")
    p.add_argument("--frost", type=float, default=10.0, help="frost-protection setpoint (C)")
    p.add_argument("--hysteresis", type=float, default=0.3, help="thermostat half-band (C)")
    p.add_argument("--p-heating-off", type=float, default=0.15, help="probability a guest switches heating off")
    p.add_argument("--mean-vacancy", type=float, default=1.0, help="mean vacant nights between stays")
    p.add_argument("--sensor-noise", type=float, default=0.1, help="white-noise std (C)")
    p.add_argument("--sensor-bias", type=float, default=0.3, help="std of the fixed per-sensor bias (C)")
    p.add_argument("--quantization", type=float, default=0.1, help="sensor resolution (C), 0 = none")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rec_every = round(args.dt_sample / args.dt_sim)
    if not np.isclose(rec_every * args.dt_sim, args.dt_sample):
        raise SystemExit("--dt-sample must be a multiple of --dt-sim")
    rng = np.random.default_rng(args.seed)
    M, F = args.rooms, args.floors
    floor, side, pos, n_cols = layout(M, F)
    net = build_network(floor, side, pos, F, n_cols, rng)

    dt_h = args.dt_sim / 3600
    n_burn = round(args.burn_in_days * 24 / dt_h / rec_every) * rec_every
    K = n_burn + round(args.days * 24 / dt_h / rec_every) * rec_every
    t_h = (np.arange(K) - n_burn) * dt_h

    T_out, I_south, I_north = weather(t_h, rng, args.t_mean)
    checked_in, occupants, guest_sp = bookings(M, t_h, rng, args.mean_vacancy, args.p_heating_off)
    setpoint = np.where(checked_in, guest_sp, args.vacant_setpoint)
    gains = 15.0 + 15.0 * checked_in + 70.0 * occupants + 40.0 * (occupants > 0)  # W

    print(f"Simulating {M} rooms on {F} floors for {args.days:g} days (+{args.burn_in_days:g} burn-in), policy={args.policy} ...")
    X, duty = simulate(net, M, F, side, T_out, I_south, I_north, setpoint, gains, args.policy, rng,
                       args.dt_sim, args.frost, args.hysteresis, rec_every)

    keep = slice(n_burn // rec_every, None)
    samp = slice(n_burn, None, rec_every)
    Z = M + F
    T_air, T_mass, duty = X[keep, :Z], X[keep, Z:], duty[keep]
    t = t_h[samp]
    bias = args.sensor_bias * rng.standard_normal(M)
    sensor = T_air[:, :M] + bias + args.sensor_noise * rng.standard_normal((len(t), M))
    if args.quantization > 0:
        sensor = np.round(sensor / args.quantization) * args.quantization

    n = len(t)
    split = np.zeros(n, np.int8)
    split[int(args.train_frac * n):] = 1
    split[int((args.train_frac + args.val_frac) * n):] = 2
    room_number = (floor + 1) * 100 + np.array([np.sum((floor == floor[i]) & (np.arange(M) < i)) + 1 for i in range(M)])

    f32 = np.float32
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f["t"] = t
        f["t"].attrs["units"] = "hours since start (00:00 of day 0)"
        f["sensor_temp"] = sensor.astype(f32)
        f["room_temp"] = T_air[:, :M].astype(f32)
        f["room_mass_temp"] = T_mass[:, :M].astype(f32)
        f["heater_duty"] = duty[:, :M].astype(f32)
        f["setpoint"] = setpoint[samp].astype(f32)
        f["checked_in"] = checked_in[samp]
        f["occupants"] = occupants[samp]
        f["corridor_temp"] = T_air[:, M:].astype(f32)
        f["corridor_heater_duty"] = duty[:, M:].astype(f32)
        f["t_out"] = T_out[samp].astype(f32)
        f["solar_south"] = I_south[samp].astype(f32)
        f["solar_north"] = I_north[samp].astype(f32)
        f["split"] = split
        g = f.create_group("rooms")
        for name, arr in [("number", room_number), ("floor", floor), ("side", side), ("position", pos),
                          ("sensor_bias", bias), ("heater_power", net["power"][:M]), ("window_area", net["window"][:M]),
                          ("floor_area", net["area"][:M])]:
            g[name] = arr
        g = f.create_group("network")
        for name in ["A", "G", "C_air", "C_mass", "H_am", "H_ao", "H_mo", "H_mg", "power"]:
            g[name] = net[name]
        g.attrs["note"] = ("zones 0..M-1 rooms, M..M+F-1 corridors (by floor); continuous-time state "
                           "x = [T_air, T_mass], dx/dt = A x + forcing; units J/K, W/K, W, 1/s")
        f.attrs.update(
            description=__doc__,
            layout=("arrays are (time, room); side 0 = north, 1 = south; heater_duty[k] = fraction of "
                    "[t_k, t_k + dt_sample) the heater was on; setpoint NaN = guest switched heating off"),
            split_labels="0 = train, 1 = val, 2 = test (chronological)",
            n_floors=F, n_cols=n_cols, t_ground=T_GROUND, corridor_setpoint=CORRIDOR_SETPOINT,
            **{k: v for k, v in vars(args).items() if k != "out"},
        )

    on_frac = duty[:, :M].mean()
    kwh_day = (duty[:, :M] * net["power"][:M]).sum(1).mean() * 24 / 1000
    print(f"Wrote {args.out}: {n} samples every {args.dt_sample / 60:g} min")
    print(f"  outdoor {T_out[samp].min():.1f}..{T_out[samp].max():.1f} C, rooms {T_air[:, :M].min():.1f}..{T_air[:, :M].max():.1f} C")
    print(f"  occupancy {checked_in[samp].mean():.0%}, heaters on {on_frac:.0%} of the time, "
          f"room heating {kwh_day:.0f} kWh/day ({kwh_day / M:.1f} per room)")


if __name__ == "__main__":
    main()
