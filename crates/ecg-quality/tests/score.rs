use ecg_quality::{score, QualityConfig, QualityFeatures};

/// A window with healthy values for every feature must score near 1. An earlier
/// scorer returned 0.02 on pristine signal, which condemned whole records; this
/// pins the arithmetic independently of any corpus.
#[test]
fn clean_window_scores_high() {
    let cfg = QualityConfig::new(360.0);
    let f = QualityFeatures {
        kurtosis: 6.3,
        skewness: -0.7,
        hf_ratio: 0.013,
        base_ratio: 0.083,
        qrs_ratio: 0.56,
        p2p: 2.0,
        p2p_rel: 1.0,
        kurtosis_rel: 1.0,
        qrs_ratio_rel: 1.0,
        sat_frac: 0.0,
        flat_frac: 0.14,
    };
    let s = score(&f, &cfg);
    assert!(s > 0.80, "clean window scored {s}");
}

#[test]
fn noisy_window_scores_low() {
    let cfg = QualityConfig::new(360.0);
    let f = QualityFeatures {
        kurtosis: 0.2,
        skewness: 0.1,
        hf_ratio: 0.02,
        base_ratio: 0.40,
        qrs_ratio: 0.09,
        p2p: 6.0,
        p2p_rel: 3.0,
        kurtosis_rel: 0.05,
        qrs_ratio_rel: 0.15,
        sat_frac: 0.0,
        flat_frac: 0.02,
    };
    let s = score(&f, &cfg);
    assert!(s < 0.25, "noisy window scored {s}");
}

/// A clean recording of a rhythm with wide, frequent complexes - bigeminy,
/// paced - is far less leptokurtic than clean sinus. It must still pass.
#[test]
fn clean_bigeminy_is_not_condemned() {
    let cfg = QualityConfig::new(128.0);
    let f = QualityFeatures {
        kurtosis: 0.9,
        skewness: -0.3,
        hf_ratio: 0.009,
        base_ratio: 0.16,
        qrs_ratio: 0.21,
        p2p: 3.9,
        p2p_rel: 1.0,
        kurtosis_rel: 1.0,
        qrs_ratio_rel: 1.0,
        sat_frac: 0.0,
        flat_frac: 0.01,
    };
    let s = score(&f, &cfg);
    assert!(s > 0.80, "clean bigeminy scored {s}");
}

/// A twentyfold amplitude collapse is an electrode coming off. It must be
/// vetoed outright, not averaged against shape statistics that still look
/// plausible on whatever is left.
#[test]
fn lead_off_is_vetoed() {
    let cfg = QualityConfig::new(128.0);
    let f = QualityFeatures {
        kurtosis: 4.0,
        skewness: -0.5,
        hf_ratio: 0.02,
        base_ratio: 0.10,
        qrs_ratio: 0.40,
        p2p: 0.10,
        p2p_rel: 0.05,
        kurtosis_rel: 1.0,
        qrs_ratio_rel: 1.0,
        sat_frac: 0.0,
        flat_frac: 0.10,
    };
    let s = score(&f, &cfg);
    assert!(s <= cfg.score_bad, "lead-off scored {s}");
}
