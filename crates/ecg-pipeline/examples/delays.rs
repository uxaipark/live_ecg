use ecg_pipeline::{PreprocessConfig, Preprocessor};
fn main() {
    for fs in [250.0, 500.0] {
        let pre = Preprocessor::new(PreprocessConfig::new(fs));
        println!("fs {fs}");
        for f in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0] {
            println!(
                "  {f:5.1} Hz  clean {:6.2} sa {:6.1} ms | pt {:6.2} sa {:6.1} ms | qrs {:6.2} sa {:6.1} ms",
                pre.group_delay_samples(f),
                pre.group_delay_samples(f) * 1000.0 / fs,
                pre.pt_group_delay_samples(f),
                pre.pt_group_delay_samples(f) * 1000.0 / fs,
                pre.qrs_group_delay_samples(f),
                pre.qrs_group_delay_samples(f) * 1000.0 / fs,
            );
        }
    }
}
