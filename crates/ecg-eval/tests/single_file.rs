//! `dist/ecg_engine.rs` is generated from the workspace and committed, so a
//! host can take the one file. It must be what the workspace generates now:
//! an engine change that was not regenerated would ship the old engine under
//! a stale identity.

use std::path::Path;

#[test]
fn the_single_file_engine_is_up_to_date() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let id = ecg_eval::amalgamate::identity(&root).unwrap();
    let fresh = ecg_eval::amalgamate::generate(&root, &id).unwrap();
    let on_disk = std::fs::read_to_string(root.join("dist/ecg_engine.rs")).unwrap();
    assert!(
        fresh == on_disk,
        "dist/ecg_engine.rs is stale: run tools/build_engine.sh and commit it"
    );
    let header = std::fs::read_to_string(root.join("crates/ecg-ffi/include/ecg.h")).unwrap();
    let shipped = std::fs::read_to_string(root.join("dist/ecg.h")).unwrap();
    assert!(
        header == shipped,
        "dist/ecg.h differs from crates/ecg-ffi/include/ecg.h"
    );
}
