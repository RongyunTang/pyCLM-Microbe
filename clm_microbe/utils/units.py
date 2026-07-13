# utils/units.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict
import re
import numpy as np
import pandas as pd 

Number = float

# ---- Constants ----
AVOGADRO = 6.02214076e23  # 1/mol
R_J_per_molK = 8.314462618  # J mol^-1 K^-1
K_B_eV_per_K = 8.617333262145e-5  # eV K^-1 (Boltzmann)
J_PER_kcal = 4184.0
J_PER_eV_PER_MOL = 96485.33212  # (1 eV per molecule) * Na = 96.485 kJ/mol

MOLAR_MASS: Dict[str, float] = {
    # g / mol
    "C": 12.01,
    "H": 1.008,
    "O": 16.00,
    "N": 14.01,
    # common species
    "CH4": 16.04,
    "CO2": 44.01,
    "O2": 32.00,
    "H2O": 18.01528,
}

# ---- Scalar factors (dimension only) ----
def mass_factor(frm: Literal["mg","g","kg"], to: Literal["mg","g","kg"]) :
    table = {"mg": 1e-3, "g": 1.0, "kg": 1e3}  # to grams
    return table[frm] / table[to]

def amount_factor(frm: Literal["pmol","nmol","umol","mmol","mol"],
                  to:   Literal["pmol","nmol","umol","mmol","mol"]) :
    table = {"pmol":1e-12, "nmol":1e-9, "umol":1e-6, "mmol":1e-3, "mol":1.0}  # to mol
    return table[frm] / table[to]

def time_factor(frm: Literal["s","min","hr","day"], to: Literal["s","min","hr","day"]) :
    table = {"s":1.0, "min":60.0, "hr":3600.0, "day":86400.0}  # to seconds
    return table[frm] / table[to]

def length_factor(frm: Literal["mm","cm","m"], to: Literal["mm","cm","m"]) :
    table = {"mm":1e-3, "cm":1e-2, "m":1.0}  # to meters
    return table[frm] / table[to]

def area_factor(frm: Literal["cm2","m2"], to: Literal["cm2","m2"]) :
    table = {"cm2":1e-4, "m2":1.0}  # to m^2
    return table[frm] / table[to]

def volume_factor(frm: Literal["cm3","m3"], to: Literal["cm3","m3"]) :
    table = {"cm3":1e-6, "m3":1.0}  # to m^3
    return table[frm] / table[to]

# ---- Temperature ----
def C_to_K(TC) : return TC + 273.15
def K_to_C(TK) : return TK - 273.15

# ---- Mass/Amount conversions using molar mass ----
def mass_to_moles(mass, mass_unit: Literal["mg","g","kg"],
                  species: str) :
    """mass unit -> mol (uses MOLAR_MASS in g/mol)."""
    g = mass * mass_factor(mass_unit, "g")
    M = MOLAR_MASS[species]
    return g / M  # mol

def moles_to_mass(moles, mass_unit: Literal["mg","g","kg"],
                  species: str) :
    g = moles * MOLAR_MASS[species]
    return g * mass_factor("g", mass_unit)

def amount_to_amount(x, frm: Literal["pmol","nmol","umol","mmol","mol"],
                     to:   Literal["pmol","nmol","umol","mmol","mol"]) :
    return x * amount_factor(frm, to)

# ---- Concentration helpers ----
def mg_per_cm3_to_kg_per_m3(x) :
    # 1 mg/cm^3 == 1 kg/m^3
    return x
def mg_per_cm3_to_g_per_m3(x) :
    # 1 mg/cm^3 == 1000 g/m^3.
    # Derivation: 1 mg = 1e-3 g, 1 cm^3 = 1e-6 m^3.
    # → 1 mg/cm^3 = 1e-3 g / 1e-6 m^3 = 1000 g/m^3.
    return x * 1e3
def g_per_cm3_to_kg_per_m3(x) :
    # 1 g/cm^3 == 1000 kg/m^3
    return x * 1e3
def g_per_m3_to_kg_per_m3(x) :
    # 1 g/m^3 == 0.001 kg/m^3
    return x * 1e-3
def umol_per_cm3_to_mol_per_m3(x) :
    # 1 µmol/cm^3 == 1 mol/m^3
    return x

# ---- Energy per mole ----
def Jmol_to_kJmol(E) :   return E / 1e3
def kJmol_to_Jmol(E) :   return E * 1e3
def Jmol_to_kcalmol(E) : return E / J_PER_kcal
def kcalmol_to_Jmol(E) : return E * J_PER_kcal
def kJmol_to_kcalmol(E) : return Jmol_to_kcalmol(kJmol_to_Jmol(E))
def kcalmol_to_kJmol(E) : return Jmol_to_kJmol(kcalmol_to_Jmol(E))
def Jmol_to_eV(E) :      return E / J_PER_eV_PER_MOL
def eV_to_Jmol(E) :      return E * J_PER_eV_PER_MOL

# ---- Arrhenius / Q10 relations ----
def Ea_to_Q10(Ea_J_mol, T_ref_K, deltaT = 10.0) :
    """Compute Q10 from activation energy at reference temp (Kelvin)."""
    R = R_J_per_molK
    return float((2.718281828)**(Ea_J_mol * (1.0/R) * (1.0/(T_ref_K - 0.0) - 1.0/(T_ref_K + deltaT))))

def Q10_to_Ea(Q10, T_ref_K, deltaT = 10.0) :
    """Activation energy from Q10 at reference temp."""
    from math import log
    R = R_J_per_molK
    return R * log(Q10) / (1.0/(T_ref_K - 0.0) - 1.0/(T_ref_K + deltaT))


_NUMERIC_KEEP = re.compile(r"[^0-9eE\.\+\-]")  # allow digits, dot, sign, exponent

def _coerce_numeric_ndarray(x, *, name="value"):
    """
    Accepts pd.Series, ndarray, list, or scalar.
    Returns a float NumPy array with junk stripped (commas, spaces, unit suffixes).
    Raises if everything becomes NaN.
    """
    if isinstance(x, pd.Series):
        s = x.astype(str).str.strip()
        s = s.str.replace(r"\s+", "", regex=True).str.replace(",", "", regex=False)
        s = s.str.replace(_NUMERIC_KEEP, "", regex=True)
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    else:
        # array-like or scalar
        try:
            arr = np.asarray(x, dtype=float)
        except Exception:
            s = pd.Series(x, dtype="object").astype(str).str.strip()
            s = s.str.replace(r"\s+", "", regex=True).str.replace(",", "", regex=False)
            s = s.str.replace(_NUMERIC_KEEP, "", regex=True)
            arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    if np.isnan(arr).all():
        raise TypeError(f"{name} is non-numeric after cleaning.")
    return arr


# ---- Flux / rate helpers ----
def flux_massC_to_umol_CH4_per_m2_s(flux_mgC_m2_hr, carbon_per_molecule):
    """
    Convert mg C m^-2 hr^-1 to µmol CH4 m^-2 s^-1.
    By default assumes 1 C atom per molecule (CH4, CO2).

    The per-hour -> per-second step DIVIDES by time_factor("hr", "s") (= 3600),
    since hours are in the denominator of the rate.
    """
    # mgC -> gC (mg in numerator: multiply by 1e-3)
    gC_m2_hr = flux_mgC_m2_hr * mass_factor("mg", "g")
    # /hr -> /s  (hr in denominator: divide by 3600)
    gC_m2_s = gC_m2_hr / time_factor("hr", "s")
    # gC -> mol C
    molC_m2_s = gC_m2_s / MOLAR_MASS["C"]
    # mol C == mol molecule when 1 C per molecule
    mol_molecule_m2_s = molC_m2_s / carbon_per_molecule
    # mol -> µmol
    umol_molecule_m2_s = mol_molecule_m2_s * amount_factor("mol", "umol")
    return umol_molecule_m2_s

def flux_massC_to_umol_CO2_per_m2_s(flux_mgC_m2_hr, carbon_per_molecule):
    """
    Convert mg C m^-2 hr^-1 to µmol CO2 m^-2 s^-1.
    By default assumes 1 C atom per molecule (CH4, CO2).

    The per-hour -> per-second step DIVIDES by time_factor("hr", "s") (= 3600),
    since hours are in the denominator of the rate.
    """
    flux_mgC_m2_hr = _coerce_numeric_ndarray(flux_mgC_m2_hr, name="flux_mgC_m2_hr")

    # mgC -> gC (numerator)
    gC_m2_hr = flux_mgC_m2_hr * mass_factor("mg", "g")
    # /hr -> /s (denominator: divide, do not multiply)
    gC_m2_s  = gC_m2_hr / time_factor("hr", "s")
    # gC -> mol C
    molC_m2_s = gC_m2_s / MOLAR_MASS["C"]
    # mol C == mol molecule when 1 C per molecule
    mol_CO2_m2_s = molC_m2_s / carbon_per_molecule
    # mol -> µmol
    umol_CO2_m2_s = mol_CO2_m2_s * amount_factor("mol", "umol")
    return umol_CO2_m2_s

def flux_mass_to_molar(
    flux_value,
    mass_unit: Literal["mg","g","kg"],
    species: str,
    per_area_from: Literal["cm2","m2"],
    per_time_from: Literal["s","min","hr","day"],
    per_area_to: Literal["cm2","m2"] = "m2",
    per_time_to: Literal["s","min","hr","day"] = "s",
    out_amount_unit: Literal["pmol","nmol","umol","mmol","mol"] = "umol",
):
    """
    Generic: (mass of species) * area^-1 * time^-1  -> (amount) * area^-1 * time^-1.
    Example: mg CH4 m^-2 hr^-1  -> µmol CH4 m^-2 s^-1
    """
    # mass -> mol
    mol_per_area_time = (mass_to_moles(flux_value, mass_unit, species))
    # per area/time unit factors
    A = area_factor(per_area_from, per_area_to)
    T = time_factor(per_time_from, per_time_to)
    mol_per_area_time *= (1.0 / A) * (1.0 / T)
    # mol -> requested amount unit
    return mol_per_area_time * amount_factor("mol", out_amount_unit)

# Shorthands you’ll use often:
def mgC_m2_hr_to_umol_CH4_m2_s(x):
    # mg C m^-2 hr^-1 -> µmol CH4 m^-2 s^-1 (1 C per CH4)
    x = _coerce_numeric_ndarray(x, name="CH4.flux.mgCm-2hr-1")
    return flux_massC_to_umol_CH4_per_m2_s(x, carbon_per_molecule=1)

def mgC_m2_hr_to_umol_CO2_m2_s(x):
    # mg C m^-2 hr^-1 -> µmol CO2 m^-2 s^-1 (1 C per CO2)
    x = _coerce_numeric_ndarray(x, name="CO2.flux.mgCm-2hr-1")
    results = flux_massC_to_umol_CO2_per_m2_s(x, carbon_per_molecule=1)
    return results

def mgCH4_m2_hr_to_umol_m2_s(x):
    # mg CH4 m^-2 hr^-1 -> µmol CH4 m^-2 s^-1
    x = _coerce_numeric_ndarray(x, name="CH4.flux.mgCm-2hr-1")
    return flux_mass_to_molar(x, mass_unit="mg", species="CH4",
                              per_area_from="m2", per_time_from="hr",
                              per_area_to="m2", per_time_to="s",
                              out_amount_unit="umol")

def ppm_to_nmol_m3(x):
    """
    Convert ppm to nmol/m^3 for any gas.
    ideal gas law: PV = nRT  =>  n/V = P/RT
    P: Pa
    V: m^3
    n: mol
    T: K
    R: 8.314 J/(mol K)
    """
    mol_fraction = 1e-6  # 1ppm = 1e-6 fraction
    P = 101325  # Pa (not used)
    T = 298  # K at 25 degC (not used)
    R = 8.314  # J/(mol K) (not used)
    mol_per_m3 = (P) / (R * T)  # mol/m^3, around 40.9 mol/m^3 at 25 degC and 1 atm
    nmol_per_m3 = mol_per_m3 * 1e9  # nmol/m^3
    return x * mol_fraction * nmol_per_m3

def gC_per_kg_to_mol_per_m3(x, BD):
    """
    x:  gC per kg of soil  [gC / kg_soil]
    BD: bulk density       [g_soil / cm^3_soil]
    returns mol C per m^3 of soil  [mol C / m^3_soil]

    cm^3 appears in the DENOMINATOR of BD, so the volume conversion uses the
    inverse factor volume_factor("m3","cm3") = 1e6 (1 m^3 contains 1e6 cm^3,
    so 1/cm^3 = 1e6 / m^3).  The mass step uses mass_factor("g","kg") = 1e-3,
    expressing the cancellation g_soil/kg_soil = 1/1000 when BD's g_soil meets
    x's per-kg_soil denominator.

    Derivation:
        x [gC/kg_soil] * BD [g_soil/cm^3_soil]
          -> cancel kg_soil with g_soil:        multiply by 1e-3
                                                  (mass_factor("g","kg"))
          -> convert cm^3 denominator to m^3:   multiply by 1e6
                                                  (volume_factor("m3","cm3"))
          -> result is gC/m^3_soil
          -> divide by MOLAR_MASS["C"] (= 12.01) to get mol C/m^3.
        Net unit factor: 1e-3 * 1e6 = 1e3.
    """
    gC_per_m3   = x * BD * mass_factor("g", "kg") * volume_factor("m3", "cm3")
    molC_per_m3 = gC_per_m3 / MOLAR_MASS["C"]
    return molC_per_m3

