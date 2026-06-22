"""
Feature engineering functions for CERN electron collision data.

This module calculates derived features from raw detector measurements.
"""

import math
from typing import Dict
from schemas import CERNElectronPair


def calculate_cern_features(electron_pair: CERNElectronPair) -> Dict[str, float]:
    """
    Calculate derived features from raw CERN detector measurements.
    
    Takes raw measurements (pt, eta, phi, charge) and calculates:
    - E_total: Total energy of the system
    - delta_eta, delta_phi, delta_R: Angular separations
    - pt_product, pt_ratio: Transverse momentum relations
    - is_os: Opposite sign charge indicator
    
    Args:
        electron_pair: Raw detector measurements
        
    Returns:
        Dictionary with all 9 features needed for the model
    """
    
    # Extract raw measurements
    pt1, eta1, phi1 = electron_pair.pt1, electron_pair.eta1, electron_pair.phi1
    pt2, eta2, phi2 = electron_pair.pt2, electron_pair.eta2, electron_pair.phi2
    charge1, charge2 = electron_pair.charge1, electron_pair.charge2
    
    # Calculate energies (assuming electron mass ≈ 0.000511 GeV)
    # E = sqrt(pt² * cosh(eta)² + m²) ≈ pt * cosh(eta) for particles
    E1 = pt1 * math.cosh(eta1)
    E2 = pt2 * math.cosh(eta2)
    E_total = E1 + E2
    
    # Calculate angular separations
    delta_eta = abs(eta1 - eta2)
    delta_phi = abs(phi1 - phi2)
    
    # Normalize delta_phi to [0, π]
    if delta_phi > math.pi:
        delta_phi = 2 * math.pi - delta_phi
    
    # Calculate delta_R (angular distance in eta-phi space)
    delta_R = math.sqrt(delta_eta**2 + delta_phi**2)
    
    # Calculate transverse momentum relations
    pt_product = pt1 * pt2
    pt_ratio = pt1 / pt2 if pt2 > 0 else 0.0
    
    # Check if opposite sign charges
    is_os = 1.0 if charge1 != charge2 else 0.0
    
    return {
        'pt1': pt1,
        'pt2': pt2,
        'E_total': E_total,
        'delta_eta': delta_eta,
        'delta_phi': delta_phi,
        'delta_R': delta_R,
        'pt_product': pt_product,
        'pt_ratio': pt_ratio,
        'is_os': is_os
    }
